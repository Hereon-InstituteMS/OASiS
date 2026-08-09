// Shared translation unit for the convection-diffusion Signal family.
// One compile serves several fixture directories; each fixture runs one probe.
//
// usage: convdiff_family <probe>
//   supg_tau | dg_face_terms | upwind_direction | supg_high_pe
//   | anisotropic_layer
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace dealii;

static bool mutate()
{
  const char *m = std::getenv("T2_MUTATE");
  return m != nullptr && std::string(m) == "1";
}

// ===========================================================================
// convection_diffusion#0 -- the SUPG stabilisation parameter.
// 1D transport, b = 1, inflow 0 at x = 0 and outflow value 1 at x = 1, no
// source. For P1 in 1D the SUPG term collapses to an artificial diffusion
// tau*b^2, and the DOUBLY-ASYMPTOTIC tau is the one that makes the scheme
// nodally exact. The bare h/(2|b|) is not.
// ===========================================================================
enum TauKind
{
  TAU_NONE,
  TAU_BARE,
  TAU_DOUBLY_ASYMPTOTIC
};

static double exact_1d(double x, double eps)
{
  // (exp(x/eps) - 1)/(exp(1/eps) - 1), written so it does not overflow
  return (std::exp((x - 1.0) / eps) - std::exp(-1.0 / eps)) /
         (1.0 - std::exp(-1.0 / eps));
}

static void solve_1d(double eps, unsigned int ncells, TauKind kind,
                     double &worst_nodal_error, double &umin, double &umax,
                     double &pe_h, double &tau_used)
{
  const double       b = 1.0;
  Triangulation<1>   tria;
  GridGenerator::subdivided_hyper_cube(tria, ncells, 0.0, 1.0);
  const double h = 1.0 / ncells;
  pe_h           = b * h / (2.0 * eps);
  switch (kind)
    {
      case TAU_NONE:
        tau_used = 0.0;
        break;
      case TAU_BARE:
        tau_used = h / (2.0 * b);
        break;
      default:
        tau_used = h / (2.0 * b) *
                   (1.0 / std::tanh(pe_h) - 1.0 / pe_h);
        break;
    }

  FE_Q<1>       fe(1);
  DoFHandler<1> dof(tria);
  dof.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(
    dof, 0, Functions::ZeroFunction<1>(), constraints);
  VectorTools::interpolate_boundary_values(
    dof, 1, Functions::ConstantFunction<1>(1.0), constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof.n_dofs());
  DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A(sp);
  Vector<double>       rhs(dof.n_dofs()), sol(dof.n_dofs());

  QGauss<1>   quad(3);
  FEValues<1> fev(fe, quad,
                  update_values | update_gradients | update_JxW_values);
  const unsigned int                   n = fe.dofs_per_cell;
  FullMatrix<double>                   cm(n, n);
  Vector<double>                       cv(n);
  std::vector<types::global_dof_index> local(n);
  for (const auto &cell : dof.active_cell_iterators())
    {
      fev.reinit(cell);
      cm = 0.0;
      cv = 0.0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            cm(i, j) += ((eps + tau_used * b * b) * fev.shape_grad(i, q)[0] *
                           fev.shape_grad(j, q)[0] +
                         b * fev.shape_grad(j, q)[0] * fev.shape_value(i, q)) *
                        fev.JxW(q);
      cell->get_dof_indices(local);
      constraints.distribute_local_to_global(cm, cv, local, A, rhs);
    }
  SparseDirectUMFPACK direct;
  direct.initialize(A);
  direct.vmult(sol, rhs);
  constraints.distribute(sol);

  std::map<types::global_dof_index, Point<1>> support;
  DoFTools::map_dofs_to_support_points(MappingQ1<1>(), dof, support);
  worst_nodal_error = 0.0;
  umin              = 1e300;
  umax              = -1e300;
  for (types::global_dof_index i = 0; i < dof.n_dofs(); ++i)
    {
      const double x = support[i][0];
      worst_nodal_error =
        std::max(worst_nodal_error, std::abs(sol(i) - exact_1d(x, eps)));
      umin = std::min(umin, sol(i));
      umax = std::max(umax, sol(i));
    }
}

static int supg_tau()
{
  const unsigned int ncells = 20;
  const TauKind      under_test = mutate() ? TAU_DOUBLY_ASYMPTOTIC : TAU_BARE;
  std::cout << "tau_under_test="
            << (mutate() ? "h_over_2b_times_coth_Pe_minus_1_over_Pe"
                         : "bare_h_over_2b")
            << std::endl;
  std::cout << "n_cells=" << ncells << " b=1" << std::endl;

  const double pe_target[3] = {0.05, 0.5, 5.0};
  double       worst_test = 0.0, worst_galerkin_undershoot = 0.0;
  for (int k = 0; k < 3; ++k)
    {
      const double h   = 1.0 / ncells;
      const double eps = h / (2.0 * pe_target[k]);
      double       e_t, min_t, max_t, pe_t, tau_t;
      double       e_g, min_g, max_g, pe_g, tau_g;
      solve_1d(eps, ncells, under_test, e_t, min_t, max_t, pe_t, tau_t);
      solve_1d(eps, ncells, TAU_NONE, e_g, min_g, max_g, pe_g, tau_g);
      std::cout << "cell_peclet=" << pe_t << " eps=" << eps
                << " tau_under_test=" << tau_t
                << " max_nodal_error_under_test=" << e_t
                << " min_under_test=" << min_t << " max_under_test=" << max_t
                << std::endl;
      std::cout << "cell_peclet=" << pe_g
                << " plain_galerkin_max_nodal_error=" << e_g
                << " plain_galerkin_min=" << min_g
                << " plain_galerkin_max=" << max_g << std::endl;
      worst_test = std::max(worst_test, e_t);
      worst_galerkin_undershoot =
        std::max(worst_galerkin_undershoot, -min_g);
    }
  std::cout << "worst_max_nodal_error_under_test=" << worst_test << std::endl;
  std::cout << "worst_plain_galerkin_undershoot=" << worst_galerkin_undershoot
            << std::endl;
  // the exact solution of this inflow/outflow problem lives in [0, 1]
  const bool galerkin_leaves_the_range = worst_galerkin_undershoot > 0.1;
  std::cout << "plain_galerkin_leaves_the_zero_one_range="
            << (galerkin_leaves_the_range ? "true" : "false") << std::endl;
  const bool nodally_exact = worst_test < 1e-10;
  std::cout << "tau_under_test_is_nodally_exact="
            << (nodally_exact ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (nodally_exact
                  ? "doubly_asymptotic_tau_is_nodally_exact_at_every_peclet"
                  : "bare_tau_misses_the_exact_solution_at_the_nodes")
            << std::endl;
  return 0;
}

// ===========================================================================
// The upwind DG machinery shared by convection_diffusion#1, #2 and #3.
// Pure advection b.grad(u) = 0 with inflow data g, FE_DGQ.
// ===========================================================================
constexpr int dim = 2;

struct DGTransport
{
  Triangulation<dim>   tria;
  FE_DGQ<dim>          fe;
  DoFHandler<dim>      dof;
  SparsityPattern      sp;
  SparseMatrix<double> A;
  Vector<double>       rhs, sol;
  Tensor<1, dim>       b;

  DGTransport(unsigned int degree)
    : fe(degree)
    , dof(tria)
  {
    b[0] = 1.0;
    b[1] = 0.0;
  }

  // inflow datum: 1 below y = 0.5, 0 above
  static double g(const Point<dim> &p) { return (p[1] < 0.5) ? 1.0 : 0.0; }

  void setup(unsigned int refine)
  {
    GridGenerator::hyper_cube(tria, 0.0, 1.0, false);
    tria.refine_global(refine);
    dof.distribute_dofs(fe);
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof, dsp);
    sp.copy_from(dsp);
    A.reinit(sp);
    rhs.reinit(dof.n_dofs());
    sol.reinit(dof.n_dofs());
  }

  // with_interior_faces=false reproduces "FEValues alone, no FEInterfaceValues"
  // flip_upwind=true takes the DOWNSTREAM value in the numerical flux
  void assemble(bool with_interior_faces, bool flip_upwind,
                double reaction = 0.0)
  {
    A   = 0.0;
    rhs = 0.0;
    const unsigned int n = fe.dofs_per_cell;
    QGauss<dim>        quad(fe.degree + 2);
    QGauss<dim - 1>    fquad(fe.degree + 2);
    FEValues<dim>      fev(fe, quad,
                           update_values | update_gradients | update_JxW_values);
    FEFaceValues<dim>  ffv(fe, fquad,
                           update_values | update_normal_vectors |
                             update_quadrature_points | update_JxW_values);
    FEInterfaceValues<dim> fiv(fe, fquad,
                               update_values | update_normal_vectors |
                                 update_JxW_values);
    FullMatrix<double>                   cm(n, n);
    Vector<double>                       cv(n);
    std::vector<types::global_dof_index> local(n), nlocal(n);

    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        cv = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
              cm(i, j) += (-fev.shape_value(j, q) *
                             (b * fev.shape_grad(i, q)) +
                           reaction * fev.shape_value(i, q) *
                             fev.shape_value(j, q)) *
                          fev.JxW(q);
        cell->get_dof_indices(local);
        A.add(local, cm);
        rhs.add(local, cv);

        // boundary faces: outflow into the matrix, inflow datum into the rhs
        for (const auto f : cell->face_indices())
          if (cell->face(f)->at_boundary())
            {
              ffv.reinit(cell, f);
              cm = 0.0;
              cv = 0.0;
              for (unsigned int q = 0; q < fquad.size(); ++q)
                {
                  const double bn = b * ffv.normal_vector(q);
                  if (bn > 0)
                    for (unsigned int i = 0; i < n; ++i)
                      for (unsigned int j = 0; j < n; ++j)
                        cm(i, j) += bn * ffv.shape_value(j, q) *
                                    ffv.shape_value(i, q) * ffv.JxW(q);
                  else
                    for (unsigned int i = 0; i < n; ++i)
                      cv(i) -= bn * g(ffv.quadrature_point(q)) *
                               ffv.shape_value(i, q) * ffv.JxW(q);
                }
              A.add(local, cm);
              rhs.add(local, cv);
            }

        if (!with_interior_faces)
          continue;

        for (const auto f : cell->face_indices())
          {
            if (cell->face(f)->at_boundary())
              continue;
            const auto ncell = cell->neighbor(f);
            if (ncell->active_cell_index() < cell->active_cell_index())
              continue;   // visit each interior face once
            const unsigned int nf = cell->neighbor_of_neighbor(f);
            fiv.reinit(cell, f, numbers::invalid_unsigned_int, ncell, nf,
                       numbers::invalid_unsigned_int);
            const unsigned int ni = fiv.n_current_interface_dofs();
            FullMatrix<double> fm(ni, ni);
            fm = 0.0;
            for (unsigned int q = 0; q < fquad.size(); ++q)
              {
                const double bn = b * fiv.normal(q);
                // upwind: take the value from the cell the flow comes FROM
                const bool take_here = flip_upwind ? (bn <= 0) : (bn > 0);
                for (unsigned int i = 0; i < ni; ++i)
                  for (unsigned int j = 0; j < ni; ++j)
                    fm(i, j) += bn * fiv.shape_value(take_here, j, q) *
                                fiv.jump_in_shape_values(i, q) * fiv.JxW(q);
              }
            const auto idx = fiv.get_interface_dof_indices();
            A.add(idx, fm);
          }
      }
  }

  // how many nonzero matrix entries connect dofs living on DIFFERENT cells
  unsigned int inter_cell_couplings() const
  {
    std::vector<unsigned int> owner(dof.n_dofs(), 0);
    std::vector<types::global_dof_index> local(fe.dofs_per_cell);
    for (const auto &cell : dof.active_cell_iterators())
      {
        cell->get_dof_indices(local);
        for (auto i : local)
          owner[i] = cell->active_cell_index();
      }
    unsigned int cnt = 0;
    for (types::global_dof_index i = 0; i < dof.n_dofs(); ++i)
      for (auto it = A.begin(i); it != A.end(i); ++it)
        if (owner[it->column()] != owner[i] && std::abs(it->value()) > 1e-14)
          ++cnt;
    return cnt;
  }

  double mean_absolute_interior_jump()
  {
    QGauss<dim - 1>        fquad(fe.degree + 2);
    FEInterfaceValues<dim> fiv(fe, fquad,
                               update_values | update_normal_vectors |
                                 update_JxW_values);
    double total = 0.0, length = 0.0;
    for (const auto &cell : dof.active_cell_iterators())
      for (const auto f : cell->face_indices())
        {
          if (cell->face(f)->at_boundary())
            continue;
          const auto ncell = cell->neighbor(f);
          if (ncell->active_cell_index() < cell->active_cell_index())
            continue;
          const unsigned int nf = cell->neighbor_of_neighbor(f);
          fiv.reinit(cell, f, numbers::invalid_unsigned_int, ncell, nf,
                     numbers::invalid_unsigned_int);
          const auto idx = fiv.get_interface_dof_indices();
          for (unsigned int q = 0; q < fquad.size(); ++q)
            {
              double jump = 0.0;
              for (unsigned int i = 0; i < idx.size(); ++i)
                jump += sol(idx[i]) * fiv.jump_in_shape_values(i, q);
              total += std::abs(jump) * fiv.JxW(q);
              length += fiv.JxW(q);
            }
        }
    return total / length;
  }
};

// ---------------------------------------------------------------------------
// convection_diffusion#1 -- FEValues alone gives no interface terms.
// ---------------------------------------------------------------------------
static int dg_face_terms()
{
  DGTransport t(1);
  t.setup(4);
  const bool with_faces = mutate();
  std::cout << "interior_face_terms="
            << (with_faces ? "FEInterfaceValues" : "none_FEValues_only")
            << std::endl;
  t.assemble(with_faces, false);
  std::cout << "n_dofs=" << t.dof.n_dofs()
            << " n_active_cells=" << t.tria.n_active_cells() << std::endl;
  const unsigned int couplings = t.inter_cell_couplings();
  std::cout << "n_inter_cell_matrix_couplings=" << couplings << std::endl;
  std::cout << "cells_are_coupled=" << (couplings > 0 ? "true" : "false")
            << std::endl;

  std::string outcome = "succeeded";
  try
    {
      SparseDirectUMFPACK direct;
      direct.initialize(t.A);
      direct.vmult(t.sol, t.rhs);
    }
  catch (const std::exception &e)
    {
      const std::string w(e.what());
      outcome = "threw";
      std::cout << "solver_exception="
                << w.substr(0, std::min<size_t>(w.size(), 900)) << std::endl;
    }
  std::cout << "direct_solve=" << outcome << std::endl;
  if (outcome == "succeeded")
    {
      std::cout << "solution_min=" << *std::min_element(t.sol.begin(),
                                                        t.sol.end())
                << " solution_max="
                << *std::max_element(t.sol.begin(), t.sol.end()) << std::endl;
      std::cout << "mean_absolute_interior_jump="
                << t.mean_absolute_interior_jump() << std::endl;
    }
  std::cout << "VERDICT="
            << ((couplings == 0)
                  ? "no_interface_terms_leaves_the_cells_uncoupled"
                  : "interface_terms_couple_the_cells")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// convection_diffusion#2 -- the upwind direction.
// ---------------------------------------------------------------------------
// Cell-averaged solution in four windows: near the inflow (x < 0.1) and near
// the outflow (x > 0.9), each split at y = 0.5.
static void windows(const DGTransport &t, double &in_lo, double &in_hi,
                    double &out_lo, double &out_hi)
{
  double        sum[4] = {0, 0, 0, 0}, area[4] = {0, 0, 0, 0};
  QGauss<dim>   quad(3);
  FEValues<dim> fev(t.fe, quad,
                    update_values | update_quadrature_points |
                      update_JxW_values);
  std::vector<double> vals(quad.size());
  for (const auto &cell : t.dof.active_cell_iterators())
    {
      fev.reinit(cell);
      fev.get_function_values(t.sol, vals);
      for (unsigned int q = 0; q < quad.size(); ++q)
        {
          const Point<dim> &p = fev.quadrature_point(q);
          int               w = -1;
          if (p[0] < 0.1)
            w = (p[1] < 0.5) ? 0 : 1;
          else if (p[0] > 0.9)
            w = (p[1] < 0.5) ? 2 : 3;
          if (w >= 0)
            {
              sum[w] += vals[q] * fev.JxW(q);
              area[w] += fev.JxW(q);
            }
        }
    }
  in_lo  = sum[0] / area[0];
  in_hi  = sum[1] / area[1];
  out_lo = sum[2] / area[2];
  out_hi = sum[3] / area[3];
}

static bool try_solve(DGTransport &t, const char *tag)
{
  try
    {
      SparseDirectUMFPACK direct;
      direct.initialize(t.A);
      direct.vmult(t.sol, t.rhs);
    }
  catch (const std::exception &e)
    {
      const std::string w(e.what());
      std::cout << tag << "_direct_solve=threw" << std::endl;
      std::cout << tag << "_solver_exception="
                << w.substr(0, std::min<size_t>(w.size(), 900)) << std::endl;
      return false;
    }
  std::cout << tag << "_direct_solve=succeeded" << std::endl;
  return true;
}

static int upwind_direction()
{
  const bool flip = !mutate();
  std::cout << "flux_takes=" << (flip ? "downstream_value" : "upstream_value")
            << std::endl;

  // (a) pure advection, exactly the operator the claim is about
  {
    DGTransport t(1);
    t.setup(4);
    t.assemble(true, flip, 0.0);
    try_solve(t, "pure_advection");
  }

  // (b) the same transport with a reaction term sigma*u so that BOTH flux
  // choices give a non-singular matrix and the question "where does the inflow
  // datum end up" actually has an answer. Exact answer for the correct upwind
  // flux: u = g(y) exp(-sigma x), largest at the INFLOW.
  DGTransport t(1);
  t.setup(4);
  t.assemble(true, flip, 1.0);
  if (!try_solve(t, "with_reaction"))
    {
      std::cout << "VERDICT=flipped_flux_makes_the_system_unsolvable"
                << std::endl;
      return 0;
    }
  double in_lo, in_hi, out_lo, out_hi;
  windows(t, in_lo, in_hi, out_lo, out_hi);
  std::cout << "mean_near_inflow_low_y=" << in_lo
            << " mean_near_inflow_high_y=" << in_hi << std::endl;
  std::cout << "mean_near_outflow_low_y=" << out_lo
            << " mean_near_outflow_high_y=" << out_hi << std::endl;
  std::cout << "solution_min=" << *std::min_element(t.sol.begin(), t.sol.end())
            << " solution_max="
            << *std::max_element(t.sol.begin(), t.sol.end()) << std::endl;
  const bool decays_downstream = out_lo < in_lo;
  std::cout << "datum_is_largest_near="
            << (decays_downstream ? "inflow" : "outflow") << std::endl;
  const bool carried = decays_downstream && out_lo > 0.05 && in_lo > 0.5 &&
                       std::abs(out_hi) < 0.05;
  std::cout << "inflow_datum_is_carried_with_the_flow="
            << (carried ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (carried ? "upwind_flux_transports_with_the_flow"
                        : "flipped_flux_does_not_transport_the_inflow_datum")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// convection_diffusion#3 -- SUPG still oscillates at high Peclet.
// Skew advection of a discontinuous inflow datum: the classic internal-layer
// test. The exact solution is bounded by its own inflow data, [0, 1].
// ---------------------------------------------------------------------------
static void supg_2d(unsigned int refine, double eps, const Tensor<1, dim> &b,
                    double &umin, double &umax, unsigned int &ndofs)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  tria.refine_global(refine);
  FE_Q<dim>       fe(1);
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);
  ndofs = dof.n_dofs();

  // inflow boundaries for b = (cos t, sin t) with t in (0, 90) are x=0 (id 0)
  // and y=0 (id 2); the datum jumps at y = 0.2 on the x=0 face
  class Inflow : public Function<dim>
  {
  public:
    double value(const Point<dim> &p, const unsigned int = 0) const override
    {
      return (p[0] < 1e-12 && p[1] > 0.2) ? 1.0 : 0.0;
    }
  };
  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(dof, 0, Inflow(), constraints);
  VectorTools::interpolate_boundary_values(dof, 2, Inflow(), constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof.n_dofs());
  DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A(sp);
  Vector<double>       rhs(dof.n_dofs()), sol(dof.n_dofs());

  const double  h  = 1.0 / (1u << refine);
  const double  bn = b.norm();
  const double  pe = bn * h / (2.0 * eps);
  const double  tau =
    h / (2.0 * bn) * (1.0 / std::tanh(pe) - 1.0 / pe);
  QGauss<dim>   quad(3);
  FEValues<dim> fev(fe, quad,
                    update_values | update_gradients | update_JxW_values);
  const unsigned int                   n = fe.dofs_per_cell;
  FullMatrix<double>                   cm(n, n);
  Vector<double>                       cv(n);
  std::vector<types::global_dof_index> local(n);
  for (const auto &cell : dof.active_cell_iterators())
    {
      fev.reinit(cell);
      cm = 0.0;
      cv = 0.0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < n; ++i)
          {
            const double bgi = b * fev.shape_grad(i, q);
            for (unsigned int j = 0; j < n; ++j)
              {
                const double bgj = b * fev.shape_grad(j, q);
                cm(i, j) += (eps * (fev.shape_grad(i, q) *
                                    fev.shape_grad(j, q)) +
                             bgj * fev.shape_value(i, q) + tau * bgi * bgj) *
                            fev.JxW(q);
              }
          }
      cell->get_dof_indices(local);
      constraints.distribute_local_to_global(cm, cv, local, A, rhs);
    }
  SparseDirectUMFPACK direct;
  direct.initialize(A);
  direct.vmult(sol, rhs);
  constraints.distribute(sol);
  umin = *std::min_element(sol.begin(), sol.end());
  umax = *std::max_element(sol.begin(), sol.end());
}

// The DG(0) upwind alternative on the same problem: monotone by construction.
static void dg0_2d(unsigned int refine, const Tensor<1, dim> &b, double &umin,
                   double &umax, unsigned int &ndofs)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0, false);
  tria.refine_global(refine);
  FE_DGQ<dim>     fe(0);
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);
  ndofs = dof.n_dofs();
  DynamicSparsityPattern dsp(dof.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dof, dsp);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A(sp);
  Vector<double>       rhs(dof.n_dofs()), sol(dof.n_dofs());

  const unsigned int     n = fe.dofs_per_cell;
  QGauss<dim>            quad(2);
  QGauss<dim - 1>        fquad(2);
  FEValues<dim>          fev(fe, quad,
                             update_values | update_gradients |
                               update_JxW_values);
  FEFaceValues<dim>      ffv(fe, fquad,
                             update_values | update_normal_vectors |
                               update_quadrature_points | update_JxW_values);
  FEInterfaceValues<dim> fiv(fe, fquad,
                             update_values | update_normal_vectors |
                               update_JxW_values);
  FullMatrix<double>                   cm(n, n);
  Vector<double>                       cv(n);
  std::vector<types::global_dof_index> local(n);
  for (const auto &cell : dof.active_cell_iterators())
    {
      fev.reinit(cell);
      cm = 0.0;
      cv = 0.0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            cm(i, j) -=
              fev.shape_value(j, q) * (b * fev.shape_grad(i, q)) * fev.JxW(q);
      cell->get_dof_indices(local);
      A.add(local, cm);
      for (const auto f : cell->face_indices())
        if (cell->face(f)->at_boundary())
          {
            ffv.reinit(cell, f);
            cm = 0.0;
            cv = 0.0;
            for (unsigned int q = 0; q < fquad.size(); ++q)
              {
                const double bn = b * ffv.normal_vector(q);
                const Point<dim> &p = ffv.quadrature_point(q);
                const double gval = (p[0] < 1e-12 && p[1] > 0.2) ? 1.0 : 0.0;
                if (bn > 0)
                  for (unsigned int i = 0; i < n; ++i)
                    for (unsigned int j = 0; j < n; ++j)
                      cm(i, j) += bn * ffv.shape_value(j, q) *
                                  ffv.shape_value(i, q) * ffv.JxW(q);
                else
                  for (unsigned int i = 0; i < n; ++i)
                    cv(i) -=
                      bn * gval * ffv.shape_value(i, q) * ffv.JxW(q);
              }
            A.add(local, cm);
            rhs.add(local, cv);
          }
      for (const auto f : cell->face_indices())
        {
          if (cell->face(f)->at_boundary())
            continue;
          const auto ncell = cell->neighbor(f);
          if (ncell->active_cell_index() < cell->active_cell_index())
            continue;
          const unsigned int nf = cell->neighbor_of_neighbor(f);
          fiv.reinit(cell, f, numbers::invalid_unsigned_int, ncell, nf,
                     numbers::invalid_unsigned_int);
          const unsigned int ni = fiv.n_current_interface_dofs();
          FullMatrix<double> fm(ni, ni);
          fm = 0.0;
          for (unsigned int q = 0; q < fquad.size(); ++q)
            {
              const double bn = b * fiv.normal(q);
              for (unsigned int i = 0; i < ni; ++i)
                for (unsigned int j = 0; j < ni; ++j)
                  fm(i, j) += bn * fiv.shape_value(bn > 0, j, q) *
                              fiv.jump_in_shape_values(i, q) * fiv.JxW(q);
            }
          const auto idx = fiv.get_interface_dof_indices();
          A.add(idx, fm);
        }
    }
  SparseDirectUMFPACK direct;
  direct.initialize(A);
  direct.vmult(sol, rhs);
  umin = *std::min_element(sol.begin(), sol.end());
  umax = *std::max_element(sol.begin(), sol.end());
}

static int supg_high_pe()
{
  Tensor<1, dim> b;
  b[0]             = std::cos(numbers::PI / 6.0);
  b[1]             = std::sin(numbers::PI / 6.0);
  const double eps = 1e-6;
  std::cout << "scheme_under_test="
            << (mutate() ? "upwind_DG0" : "FE_Q1_with_SUPG") << std::endl;
  std::cout << "epsilon=" << eps << " advection=(cos30,sin30)" << std::endl;
  double       worst_under = 0.0, worst_last = 0.0, first_excursion = 0.0;
  for (unsigned int r = 4; r <= 6; ++r)
    {
      double       lo, hi;
      unsigned int nd;
      if (mutate())
        dg0_2d(r, b, lo, hi, nd);
      else
        supg_2d(r, eps, b, lo, hi, nd);
      const double excursion = std::max(-lo, hi - 1.0);
      std::cout << "refine=" << r << " n_dofs=" << nd << " min=" << lo
                << " max=" << hi << " excursion_outside_0_1=" << excursion
                << std::endl;
      if (r == 4)
        first_excursion = excursion;
      worst_under = std::max(worst_under, excursion);
      worst_last  = excursion;
    }
  std::cout << "worst_excursion_outside_0_1=" << worst_under << std::endl;
  const bool oscillates = worst_last > 0.05;
  const bool persists   = worst_last > 0.5 * first_excursion;
  std::cout << "leaves_the_zero_one_range=" << (oscillates ? "true" : "false")
            << std::endl;
  std::cout << "excursion_survives_two_refinements="
            << (persists ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << ((oscillates && persists)
                  ? "supg_still_oscillates_at_high_peclet"
                  : "scheme_stays_inside_the_data_range")
            << std::endl;
  return 0;
}

// ===========================================================================
// convection_diffusion#4 -- anisotropic vs isotropic refinement of a boundary
// layer. Measured on the nodal INTERPOLATION error of an exponential layer,
// so what is compared is how many cells each strategy needs to represent the
// layer, with no solver in the way.
// ===========================================================================
class LayerFunction : public Function<dim>
{
public:
  const double eps;
  LayerFunction(double e)
    : eps(e)
  {}
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    const double layer = (std::exp((p[0] - 1.0) / eps) - std::exp(-1.0 / eps)) /
                         (1.0 - std::exp(-1.0 / eps));
    return layer * (1.0 + 0.5 * std::sin(2.0 * numbers::PI * p[1]));
  }
};

static void layer_refinement(bool anisotropic, double eps,
                             std::vector<unsigned int> &cells,
                             std::vector<unsigned int> &dofs,
                             std::vector<double>       &err)
{
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube(tria, 4, 0.0, 1.0);
  FE_Q<dim>     fe(1);
  QGauss<dim>   quad(4);
  LayerFunction f(eps);
  for (unsigned int pass = 0; pass <= 8; ++pass)
    {
      DoFHandler<dim> dof(tria);
      dof.distribute_dofs(fe);
      Vector<double> u(dof.n_dofs());
      VectorTools::interpolate(dof, f, u);
      Vector<float> per_cell(tria.n_active_cells());
      VectorTools::integrate_difference(dof, u, f, per_cell, quad,
                                        VectorTools::L2_norm);
      const double e = VectorTools::compute_global_error(
        tria, per_cell, VectorTools::L2_norm);
      cells.push_back(tria.n_active_cells());
      dofs.push_back(dof.n_dofs());
      err.push_back(e);
      if (pass == 8)
        break;
      // refine the cells that touch the layer at x = 1
      for (const auto &cell : tria.active_cell_iterators())
        for (const auto fc : cell->face_indices())
          if (cell->face(fc)->at_boundary() &&
              std::abs(cell->face(fc)->center()[0] - 1.0) < 1e-12)
            {
              if (anisotropic)
                cell->set_refine_flag(RefinementCase<dim>::cut_axis(0));
              else
                cell->set_refine_flag();
            }
      tria.execute_coarsening_and_refinement();
    }
}

static int anisotropic_layer()
{
  const double eps = 1e-3;
  std::cout << "layer_width_eps=" << eps << std::endl;
  std::vector<unsigned int> ci, di, ca, da;
  std::vector<double>       ei, ea;
  layer_refinement(false, eps, ci, di, ei);
  layer_refinement(true, eps, ca, da, ea);
  for (unsigned int k = 0; k < ei.size(); ++k)
    std::cout << "pass=" << k << " isotropic_cells=" << ci[k]
              << " isotropic_dofs=" << di[k] << " isotropic_l2_error=" << ei[k]
              << " | anisotropic_cells=" << ca[k]
              << " anisotropic_dofs=" << da[k]
              << " anisotropic_l2_error=" << ea[k] << std::endl;

  const double target = 0.02;
  auto         first_below = [&](const std::vector<double> &e) {
    for (unsigned int k = 0; k < e.size(); ++k)
      if (e[k] < target)
        return static_cast<int>(k);
    return -1;
  };
  const int ki = first_below(ei), ka = first_below(ea);
  std::cout << "target_l2_error=" << target << std::endl;
  if (ki < 0 || ka < 0)
    {
      std::cout << "VERDICT=target_not_reached" << std::endl;
      return 0;
    }
  const unsigned int ci_at = ci[ki], ca_at = ca[ka];
  std::cout << "isotropic_cells_to_reach_target=" << ci_at
            << " anisotropic_cells_to_reach_target=" << ca_at << std::endl;
  std::cout << "isotropic_dofs_to_reach_target=" << di[ki]
            << " anisotropic_dofs_to_reach_target=" << da[ka] << std::endl;
  const double ratio = double(ci_at) / double(ca_at);
  std::cout << "cell_count_ratio_isotropic_over_anisotropic=" << ratio
            << std::endl;
  const unsigned int used = mutate() ? ca_at : ci_at;
  // "wasteful" is measured against the cheapest strategy tried, not against
  // which branch was taken
  const bool wasteful = used > 3 * ca_at;
  std::cout << "strategy_under_test="
            << (mutate() ? "anisotropic_cut_x" : "isotropic") << std::endl;
  std::cout << "cells_used_by_strategy_under_test=" << used << std::endl;
  std::cout << "strategy_under_test_wastes_cells="
            << (wasteful ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (wasteful ? "isotropic_refinement_costs_multiples_of_the_cells"
                         : "strategy_is_not_wasteful")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  if (probe == "supg_tau")
    return supg_tau();
  if (probe == "dg_face_terms")
    return dg_face_terms();
  if (probe == "upwind_direction")
    return upwind_direction();
  if (probe == "supg_high_pe")
    return supg_high_pe();
  if (probe == "anisotropic_layer")
    return anisotropic_layer();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
