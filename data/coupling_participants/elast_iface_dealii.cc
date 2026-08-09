/* Plane-strain linear elasticity  -div(sigma(u)) = 0  on ONE rectangular
 * subdomain of the OASiS VECTOR coupling problem (two subdomains sharing a
 * straight interface at x = iface_x).  The vector counterpart of
 * heat_iface_dealii.cc.
 *
 * Domain [x0,x1] x [y0,y1].  The WHOLE non-interface boundary (the outer
 * x-face and both y-faces) carries a prescribed displacement given as a
 * polynomial in (x, y):
 *     u_x = cx[0] + cx[1]*x + cx[2]*y + cx[3]*y*y
 *     u_y = cy[0] + cy[1]*x + cy[2]*y + cy[3]*y*y
 * On the interface x = iface_x this participant is either
 *   side = 0 (DIRICHLET): u = u_if(y)  (piecewise linear from the samples), or
 *   side = 1 (NEUMANN)  : + \int g(y) . v ds  added to the weak-form RHS
 *                         (g = the partner's exported traction, UNCHANGED).
 *
 * THE TWO INTERFACE CORNERS (iface_x, y0) and (iface_x, y1) belong to the
 * OUTER boundary on BOTH sides: they sit on a y-face, which is Dirichlet in
 * the un-split problem. The outer constraint therefore WINS there — see the
 * std::map merge below, which relies on std::map::insert not overwriting.
 * Handing them to the interface leaves them unconstrained on the Neumann side;
 * that subproblem is still well posed, still converges, and lands a few
 * percent off with a residual of 1e-10 and a balanced interface.
 *
 * SIGN CONVENTION for the exported traction, identical to the shipped Python
 * vector participants and to the scalar (heat) ones:
 *     q_out = -(sigma . n_own),   n_own = s_out * e_x
 * so the two sides' exports cancel and the Neumann side applies the partner's
 * numbers unchanged.
 *
 * Input file (argv[1], whitespace separated):
 *   side E nu x0 x1 y0 y1 iface_x nx ny degree
 *   cx0 cx1 cx2 cx3
 *   cy0 cy1 cy2 cy3
 *   n_samples
 *   y_0 vx_0 vy_0
 *   ...
 * Output file (argv[2]): one line "y ux uy qx qy" per interface node,
 *   sorted by y.
 *
 * All problem numbers come from the input file — nothing is hardcoded.
 */
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

using namespace dealii;

/* Piecewise-linear interpolant of vector samples v(y), clamped outside the
 * sample range — identical semantics to numpy.interp applied per component. */
class Sampled2D
{
public:
  Sampled2D(std::vector<double> ys, std::vector<double> vx,
            std::vector<double> vy)
    : ys_(std::move(ys)), vx_(std::move(vx)), vy_(std::move(vy))
  {}

  double operator()(const double y, const unsigned int c) const
  {
    const std::vector<double> &v = (c == 0 ? vx_ : vy_);
    if (ys_.size() == 1 || y <= ys_.front())
      return v.front();
    if (y >= ys_.back())
      return v.back();
    const auto        it = std::upper_bound(ys_.begin(), ys_.end(), y);
    const std::size_t i  = std::distance(ys_.begin(), it);
    const double      w  = (y - ys_[i - 1]) / (ys_[i] - ys_[i - 1]);
    return (1.0 - w) * v[i - 1] + w * v[i];
  }

private:
  std::vector<double> ys_, vx_, vy_;
};

/* The prescribed displacement on the non-interface boundary. */
class PolyBC : public Function<2>
{
public:
  PolyBC(const std::array<double, 4> &cx, const std::array<double, 4> &cy)
    : Function<2>(2)
    , cx_(cx)
    , cy_(cy)
  {}
  virtual double value(const Point<2>  &p,
                       const unsigned int component = 0) const override
  {
    const std::array<double, 4> &c = (component == 0 ? cx_ : cy_);
    return c[0] + c[1] * p[0] + c[2] * p[1] + c[3] * p[1] * p[1];
  }

private:
  const std::array<double, 4> cx_, cy_;
};

/* Sampled2D as a 2-component deal.II Function of y. */
class InterfaceFunction : public Function<2>
{
public:
  explicit InterfaceFunction(const Sampled2D &s)
    : Function<2>(2)
    , s_(s)
  {}
  virtual double value(const Point<2>  &p,
                       const unsigned int component = 0) const override
  {
    return s_(p[1], component);
  }

private:
  const Sampled2D &s_;
};

int main(int argc, char *argv[])
{
  if (argc < 3)
    {
      std::cerr << "usage: elast_iface_dealii <input.txt> <output.txt>\n";
      return 1;
    }

  std::ifstream in(argv[1]);
  if (!in)
    {
      std::cerr << "cannot open input file " << argv[1] << "\n";
      return 1;
    }

  unsigned int side;
  double       E, nu, x0, x1, y0, y1, iface_x;
  unsigned int nx, ny, degree, n_samples;
  in >> side >> E >> nu >> x0 >> x1 >> y0 >> y1 >> iface_x >> nx >> ny >>
    degree;
  std::array<double, 4> cx{}, cy{};
  for (unsigned int i = 0; i < 4; ++i)
    in >> cx[i];
  for (unsigned int i = 0; i < 4; ++i)
    in >> cy[i];
  in >> n_samples;
  std::vector<double> sy(n_samples), svx(n_samples), svy(n_samples);
  for (unsigned int i = 0; i < n_samples; ++i)
    in >> sy[i] >> svx[i] >> svy[i];
  if (!in || n_samples == 0)
    {
      std::cerr << "malformed input file\n";
      return 1;
    }
  const Sampled2D samples(sy, svx, svy);

  /* Plane strain. */
  const double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  const double mu     = E / (2.0 * (1.0 + nu));

  /* colorize=true gives 0: x = x0, 1: x = x1, 2: y = y0, 3: y = y1. */
  const bool         iface_at_x1 = std::abs(iface_x - x1) < std::abs(iface_x - x0);
  const unsigned int iface_id    = iface_at_x1 ? 1 : 0;
  const unsigned int outer_id    = iface_at_x1 ? 0 : 1;
  const double       s_out       = iface_at_x1 ? 1.0 : -1.0; // n_own = s * e_x

  Triangulation<2> tria;
  GridGenerator::subdivided_hyper_rectangle(
    tria, {nx, ny}, Point<2>(x0, y0), Point<2>(x1, y1), /*colorize=*/true);

  const FE_Q<2>     base(degree);
  const FESystem<2> fe(base, 2);
  DoFHandler<2>     dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  /* Constraints: the OUTER boundary first, then the interface, merged with
   * std::map::insert so that a dof already fixed by the outer boundary — the
   * two interface corners — keeps that value. */
  const PolyBC                             outer_fun(cx, cy);
  std::map<types::global_dof_index, double> fixed;
  for (const unsigned int id : {outer_id, 2u, 3u})
    VectorTools::interpolate_boundary_values(dof_handler, id, outer_fun, fixed);
  if (side == 0)
    {
      const InterfaceFunction                   iface_fun(samples);
      std::map<types::global_dof_index, double> iface_vals;
      VectorTools::interpolate_boundary_values(dof_handler, iface_id, iface_fun,
                                               iface_vals);
      for (const auto &kv : iface_vals)
        fixed.insert(kv); // does NOT overwrite: the outer boundary wins
    }

  AffineConstraints<double> constraints;
  for (const auto &kv : fixed)
    {
      constraints.add_line(kv.first);
      constraints.set_inhomogeneity(kv.first, kv.second);
    }
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityPattern sparsity;
  sparsity.copy_from(dsp);

  SparseMatrix<double> system_matrix(sparsity);
  Vector<double>       solution(dof_handler.n_dofs());
  Vector<double>       rhs(dof_handler.n_dofs());

  const QGauss<2> quadrature(degree + 1);
  FEValues<2>     fe_values(fe, quadrature,
                            update_values | update_gradients | update_JxW_values);
  const QGauss<1> face_quadrature(degree + 2);
  FEFaceValues<2> fe_face_rhs(fe, face_quadrature,
                              update_values | update_quadrature_points |
                                update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dofs(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0.;
      cell_rhs    = 0.;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            const SymmetricTensor<2, 2> eps_i =
              fe_values[FEValuesExtractors::Vector(0)].symmetric_gradient(i, q);
            const double div_i =
              fe_values[FEValuesExtractors::Vector(0)].divergence(i, q);
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              {
                const SymmetricTensor<2, 2> eps_j =
                  fe_values[FEValuesExtractors::Vector(0)].symmetric_gradient(j,
                                                                              q);
                const double div_j =
                  fe_values[FEValuesExtractors::Vector(0)].divergence(j, q);
                cell_matrix(i, j) +=
                  (2.0 * mu * (eps_i * eps_j) + lambda * div_i * div_j) *
                  fe_values.JxW(q);
              }
          }

      /* Neumann side: + \int g(y) . v ds on the interface faces. */
      if (side == 1)
        for (const unsigned int f : cell->face_indices())
          if (cell->face(f)->at_boundary() &&
              cell->face(f)->boundary_id() == iface_id)
            {
              fe_face_rhs.reinit(cell, f);
              for (unsigned int q = 0; q < face_quadrature.size(); ++q)
                {
                  const double y = fe_face_rhs.quadrature_point(q)[1];
                  for (unsigned int i = 0; i < dofs_per_cell; ++i)
                    {
                      const unsigned int c =
                        fe.system_to_component_index(i).first;
                      cell_rhs(i) += samples(y, c) *
                                     fe_face_rhs.shape_value(i, q) *
                                     fe_face_rhs.JxW(q);
                    }
                }
            }

      cell->get_dof_indices(local_dofs);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dofs,
                                             system_matrix, rhs);
    }

  SolverControl            control(50000, 1e-14 * rhs.l2_norm() + 1e-18);
  SolverCG<Vector<double>> solver(control);
  PreconditionSSOR<SparseMatrix<double>> precond;
  precond.initialize(system_matrix, 1.2);
  solver.solve(system_matrix, solution, rhs, precond);
  constraints.distribute(solution);

  /* Interface displacement and outward traction export at the FE support
   * points of the interface faces, averaged over the adjacent cells and keyed
   * by y. Same recovery as heat_iface_dealii.cc, one component at a time. */
  const Quadrature<1> face_support(base.get_unit_face_support_points());
  FEFaceValues<2>     fe_face(fe, face_support,
                              update_values | update_gradients |
                                update_quadrature_points);
  std::vector<Vector<double>> face_u(face_support.size(), Vector<double>(2));
  std::vector<std::vector<Tensor<1, 2>>> face_grad(
    face_support.size(), std::vector<Tensor<1, 2>>(2));
  std::map<double, std::array<double, 5>> iface; // y -> (ux, uy, qx, qy, count)

  for (const auto &cell : dof_handler.active_cell_iterators())
    for (const unsigned int f : cell->face_indices())
      if (cell->face(f)->at_boundary() &&
          cell->face(f)->boundary_id() == iface_id)
        {
          fe_face.reinit(cell, f);
          fe_face.get_function_values(solution, face_u);
          fe_face.get_function_gradients(solution, face_grad);
          for (unsigned int q = 0; q < face_support.size(); ++q)
            {
              const double y   = fe_face.quadrature_point(q)[1];
              const double key = std::round(y * 1e10) / 1e10;
              const double exx = face_grad[q][0][0];
              const double eyy = face_grad[q][1][1];
              const double exy = 0.5 * (face_grad[q][0][1] + face_grad[q][1][0]);
              const double sxx = 2.0 * mu * exx + lambda * (exx + eyy);
              const double sxy = 2.0 * mu * exy;
              auto        &e   = iface[key];
              e[0] += face_u[q][0];
              e[1] += face_u[q][1];
              e[2] += -s_out * sxx;
              e[3] += -s_out * sxy;
              e[4] += 1.0;
            }
        }

  std::ofstream out(argv[2]);
  out.precision(16);
  for (const auto &[y, e] : iface)
    out << y << " " << e[0] / e[4] << " " << e[1] / e[4] << " " << e[2] / e[4]
        << " " << e[3] / e[4] << "\n";
  return 0;
}
