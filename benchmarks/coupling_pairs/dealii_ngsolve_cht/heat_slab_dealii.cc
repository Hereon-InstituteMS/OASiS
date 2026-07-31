/* Steady heat conduction on a rectangular slab (deal.II participant of the
 * deal.II <-> NGSolve conjugate-heat-transfer coupling pair).
 *
 * Domain: [x_min,x_max] x [y_min,y_max]. Constant conductivity k.
 * BCs: Dirichlet T = T_dirichlet on y = y_min (outer boundary),
 *      Dirichlet T = T_if(x)    on y = y_max (coupling interface,
 *                                piecewise-linear from sampled data),
 *      insulated (natural) on the lateral sides.
 *
 * Input file (argv[1], whitespace separated):
 *   k x_min x_max y_min y_max T_dirichlet nx ny degree
 *   n_samples
 *   x_0 T_0
 *   ...
 * Output file (argv[2]): one line "x T q" per interface point, where
 *   q = -k dT/dy at (x, y_max)  (heat flux through the interface in +y).
 *
 * All problem numbers come from the input file — nothing is hardcoded.
 */
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
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
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

using namespace dealii;

/* Piecewise-linear interpolant of sampled interface temperature T(x),
 * clamped outside the sample range. */
class InterfaceTemperature : public Function<2>
{
public:
  InterfaceTemperature(std::vector<double> xs, std::vector<double> Ts)
    : xs_(std::move(xs)), Ts_(std::move(Ts))
  {}

  virtual double value(const Point<2> &p, const unsigned int = 0) const override
  {
    const double x = p[0];
    if (x <= xs_.front())
      return Ts_.front();
    if (x >= xs_.back())
      return Ts_.back();
    const auto it = std::upper_bound(xs_.begin(), xs_.end(), x);
    const std::size_t i = std::distance(xs_.begin(), it);
    const double w = (x - xs_[i - 1]) / (xs_[i] - xs_[i - 1]);
    return (1.0 - w) * Ts_[i - 1] + w * Ts_[i];
  }

private:
  std::vector<double> xs_, Ts_;
};

int main(int argc, char *argv[])
{
  if (argc < 3)
    {
      std::cerr << "usage: heat_slab_dealii <input.txt> <output.txt>\n";
      return 1;
    }

  std::ifstream in(argv[1]);
  if (!in)
    {
      std::cerr << "cannot open input file " << argv[1] << "\n";
      return 1;
    }

  double k, x_min, x_max, y_min, y_max, T_dirichlet;
  unsigned int nx, ny, degree, n_samples;
  in >> k >> x_min >> x_max >> y_min >> y_max >> T_dirichlet >> nx >> ny >>
    degree >> n_samples;
  std::vector<double> sx(n_samples), sT(n_samples);
  for (unsigned int i = 0; i < n_samples; ++i)
    in >> sx[i] >> sT[i];
  if (!in)
    {
      std::cerr << "malformed input file\n";
      return 1;
    }

  Triangulation<2> tria;
  GridGenerator::subdivided_hyper_rectangle(
    tria, {nx, ny}, Point<2>(x_min, y_min), Point<2>(x_max, y_max),
    /*colorize=*/true); // boundary ids: 0 left, 1 right, 2 bottom, 3 top

  const FE_Q<2> fe(degree);
  DoFHandler<2> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  InterfaceTemperature     interface_temp(sx, sT);
  VectorTools::interpolate_boundary_values(
    dof_handler, 2, Functions::ConstantFunction<2>(T_dirichlet), constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 3, interface_temp,
                                           constraints);
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
                            update_gradients | update_JxW_values);
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
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) += k * fe_values.shape_grad(i, q) *
                                 fe_values.shape_grad(j, q) *
                                 fe_values.JxW(q);
      cell->get_dof_indices(local_dofs);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dofs,
                                             system_matrix, rhs);
    }

  SolverControl            control(10000, 1e-12 * rhs.l2_norm() + 1e-14);
  SolverCG<Vector<double>> solver(control);
  PreconditionSSOR<SparseMatrix<double>> precond;
  precond.initialize(system_matrix, 1.2);
  solver.solve(system_matrix, solution, rhs, precond);
  constraints.distribute(solution);

  /* Interface temperature and normal heat flux q = -k dT/dy at the FE
   * support points of the top-boundary faces (dedup by x-coordinate). */
  const Quadrature<1> face_quad(fe.get_unit_face_support_points());
  FEFaceValues<2>     fe_face(fe, face_quad,
                              update_values | update_gradients |
                                update_quadrature_points);
  std::vector<double>         face_T(face_quad.size());
  std::vector<Tensor<1, 2>>   face_grad(face_quad.size());
  std::map<double, std::pair<double, double>> interface; // x -> (T, q)

  for (const auto &cell : dof_handler.active_cell_iterators())
    for (const unsigned int f : cell->face_indices())
      if (cell->face(f)->at_boundary() &&
          cell->face(f)->boundary_id() == 3)
        {
          fe_face.reinit(cell, f);
          fe_face.get_function_values(solution, face_T);
          fe_face.get_function_gradients(solution, face_grad);
          for (unsigned int q = 0; q < face_quad.size(); ++q)
            {
              const double x = fe_face.quadrature_point(q)[0];
              const double key =
                std::round(x * 1e10) / 1e10; // dedupe shared vertices
              interface[key] = {face_T[q], -k * face_grad[q][1]};
            }
        }

  std::ofstream out(argv[2]);
  out.precision(16);
  for (const auto &[x, Tq] : interface)
    out << x << " " << Tq.first << " " << Tq.second << "\n";
  return 0;
}
