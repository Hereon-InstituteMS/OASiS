// Shared translation unit for the stabilisation-parameter Signal family:
// the SIPG interior-penalty constant and the SUPG tau.
//
// usage: stabilisation_family <probe>
//   sipg_penalty | galerkin_wiggles | fixed_supg_tau
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.

#include <deal.II/base/function.h>
#include <deal.II/base/multithread_info.h>
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
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace dealii;
constexpr int dim = 2;

static bool mutate()
{
  const char *m = std::getenv("T2_MUTATE");
  return m != nullptr && std::string(m) == "1";
}
static std::string yesno(bool b)
{
  return b ? "true" : "false";
}

// ===========================================================================
// advection_dg#1 -- the SIPG interior-penalty constant.
// -laplace(u) = f on the unit square with FE_DGQ(p), u = 0 on the boundary and
// the manufactured solution sin(pi x) sin(pi y). The penalty on a face is
//   sigma_F = alpha * 0.5 * (1/h_1 + 1/h_2)
// which is deal.II's own step-74 shape with the constant pulled out, so the
// entry's rule alpha = 4 (p+1)^2 can be dialled in directly.
// ===========================================================================
class SinSin : public Function<dim>
{
public:
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  }
};

struct SipgResult
{
  double l2_error = 0.0;
  double solution_l2_norm = 0.0;
  double min_eigenvalue = 0.0;
  double max_eigenvalue = 0.0;
  unsigned int gmres_steps = 0;
  bool gmres_converged = false;
  unsigned int cg_steps = 0;
  bool cg_converged = false;
  unsigned int n_dofs = 0;
};

static SipgResult sipg_solve(unsigned int refine, unsigned int degree,
                             double alpha, bool with_spectrum)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  tria.refine_global(refine);
  FE_DGQ<dim>     fe(degree);
  MappingQ1<dim>  mapping;
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);

  DynamicSparsityPattern dsp(dof.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dof, dsp);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A;
  A.reinit(sp);
  Vector<double> rhs(dof.n_dofs()), sol(dof.n_dofs());

  QGauss<dim>     quad(degree + 2);
  QGauss<dim - 1> fquad(degree + 2);
  FEValues<dim>   fev(mapping, fe, quad,
                      update_values | update_gradients |
                        update_quadrature_points | update_JxW_values);
  FEFaceValues<dim> ffv(mapping, fe, fquad,
                        update_values | update_gradients |
                          update_normal_vectors | update_JxW_values);
  FEInterfaceValues<dim> fiv(mapping, fe, fquad,
                             update_values | update_gradients |
                               update_normal_vectors | update_JxW_values);
  const unsigned int n = fe.n_dofs_per_cell();
  FullMatrix<double> cm(n, n);
  Vector<double>     cr(n);
  std::vector<types::global_dof_index> local(n);
  const SinSin exact;

  for (const auto &cell : dof.active_cell_iterators())
    {
      fev.reinit(cell);
      cm = 0.0;
      cr = 0.0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        {
          const double f = 2.0 * numbers::PI * numbers::PI *
                           exact.value(fev.quadrature_point(q));
          for (unsigned int i = 0; i < n; ++i)
            {
              for (unsigned int j = 0; j < n; ++j)
                cm(i, j) +=
                  fev.shape_grad(i, q) * fev.shape_grad(j, q) * fev.JxW(q);
              cr(i) += f * fev.shape_value(i, q) * fev.JxW(q);
            }
        }
      cell->get_dof_indices(local);
      for (unsigned int i = 0; i < n; ++i)
        {
          for (unsigned int j = 0; j < n; ++j)
            A.add(local[i], local[j], cm(i, j));
          rhs(local[i]) += cr(i);
        }

      for (const unsigned int f : cell->face_indices())
        {
          const unsigned int nd = GeometryInfo<dim>::unit_normal_direction[f];
          const double h1 = cell->extent_in_direction(nd);
          if (cell->at_boundary(f))
            {
              const double sigma = alpha / h1;
              ffv.reinit(cell, f);
              cm = 0.0;
              for (unsigned int q = 0; q < fquad.size(); ++q)
                {
                  const Tensor<1, dim> nv = ffv.normal_vector(q);
                  for (unsigned int i = 0; i < n; ++i)
                    for (unsigned int j = 0; j < n; ++j)
                      cm(i, j) += (-(ffv.shape_grad(j, q) * nv) *
                                     ffv.shape_value(i, q) -
                                   (ffv.shape_grad(i, q) * nv) *
                                     ffv.shape_value(j, q) +
                                   sigma * ffv.shape_value(i, q) *
                                     ffv.shape_value(j, q)) *
                                  ffv.JxW(q);
                }
              for (unsigned int i = 0; i < n; ++i)
                for (unsigned int j = 0; j < n; ++j)
                  A.add(local[i], local[j], cm(i, j));
              continue;
            }
          const auto ncell = cell->neighbor(f);
          if (!(cell->id() < ncell->id()))
            continue;
          const double h2 = ncell->extent_in_direction(nd);
          const double sigma = alpha * 0.5 * (1.0 / h1 + 1.0 / h2);
          fiv.reinit(cell, f, numbers::invalid_unsigned_int, ncell,
                     cell->neighbor_of_neighbor(f),
                     numbers::invalid_unsigned_int);
          const unsigned int m = fiv.n_current_interface_dofs();
          FullMatrix<double> fm(m, m);
          fm = 0.0;
          for (unsigned int q = 0; q < fquad.size(); ++q)
            {
              const Tensor<1, dim> nv = fiv.normal_vector(q);
              for (unsigned int i = 0; i < m; ++i)
                for (unsigned int j = 0; j < m; ++j)
                  fm(i, j) +=
                    (-(fiv.average_of_shape_gradients(j, q) * nv) *
                       fiv.jump_in_shape_values(i, q) -
                     (fiv.average_of_shape_gradients(i, q) * nv) *
                       fiv.jump_in_shape_values(j, q) +
                     sigma * fiv.jump_in_shape_values(i, q) *
                       fiv.jump_in_shape_values(j, q)) *
                    fiv.get_JxW_values()[q];
            }
          const auto idx = fiv.get_interface_dof_indices();
          for (unsigned int i = 0; i < m; ++i)
            for (unsigned int j = 0; j < m; ++j)
              A.add(idx[i], idx[j], fm(i, j));
        }
    }

  SipgResult r;
  r.n_dofs = dof.n_dofs();
  {
    SparseDirectUMFPACK inv;
    inv.initialize(A);
    sol = rhs;
    inv.solve(sol);
  }
  Vector<double> diff(tria.n_active_cells());
  VectorTools::integrate_difference(mapping, dof, sol, exact, diff,
                                    QGauss<dim>(degree + 3),
                                    VectorTools::L2_norm);
  r.l2_error =
    VectorTools::compute_global_error(tria, diff, VectorTools::L2_norm);
  Vector<double> zero(tria.n_active_cells());
  VectorTools::integrate_difference(mapping, dof, sol,
                                    Functions::ZeroFunction<dim>(), zero,
                                    QGauss<dim>(degree + 3),
                                    VectorTools::L2_norm);
  r.solution_l2_norm =
    VectorTools::compute_global_error(tria, zero, VectorTools::L2_norm);

  {
    SolverControl control(5000, 1e-10 * rhs.l2_norm());
    SolverGMRES<Vector<double>> gmres(control);
    Vector<double>              x(dof.n_dofs());
    try
      {
        gmres.solve(A, x, rhs, PreconditionIdentity());
        r.gmres_converged = true;
      }
    catch (const std::exception &)
      {}
    r.gmres_steps = control.last_step();
  }
  {
    // SIPG is symmetric, so CG is the solver a user reaches for -- and CG needs
    // the form to be positive definite, which is exactly what the penalty buys.
    SolverControl control(5000, 1e-10 * rhs.l2_norm());
    SolverCG<Vector<double>> cg(control);
    Vector<double>           x(dof.n_dofs());
    try
      {
        cg.solve(A, x, rhs, PreconditionIdentity());
        r.cg_converged = true;
      }
    catch (const std::exception &)
      {}
    r.cg_steps = control.last_step();
  }

  if (with_spectrum)
    {
      LAPACKFullMatrix<double> D(dof.n_dofs(), dof.n_dofs());
      for (unsigned int row = 0; row < dof.n_dofs(); ++row)
        for (auto it = A.begin(row); it != A.end(row); ++it)
          D(row, it->column()) = it->value();
      D.compute_eigenvalues();
      double mn = 1e300, mx = -1e300;
      for (unsigned int i = 0; i < dof.n_dofs(); ++i)
        {
          mn = std::min(mn, D.eigenvalue(i).real());
          mx = std::max(mx, D.eigenvalue(i).real());
        }
      r.min_eigenvalue = mn;
      r.max_eigenvalue = mx;
    }
  return r;
}

static int sipg_penalty()
{
  const unsigned int p = 1;
  const double rule = 4.0 * (p + 1) * (p + 1); // the entry's rule: 16 for p = 1
  const double small = 0.5;                    // under-penalised
  const double huge = 1e12;                    // over-penalised
  const double alpha = mutate() ? rule : small;
  std::cout << "degree=" << p << " rule_alpha=" << rule
            << " alpha_under_test=" << alpha << std::endl;

  // Coercivity and convergence over a sweep of alpha, so that "loses
  // coercivity" and "loses the rate" are two separate measured statements.
  for (double a : {0.01, 0.1, 0.5, 2.0, rule})
    {
      const SipgResult s = sipg_solve(2, p, a, true);
      double e[3], nrm[3];
      for (unsigned int k = 0; k < 3; ++k)
        {
          const SipgResult t = sipg_solve(2 + k, p, a, false);
          e[k] = t.l2_error;
          nrm[k] = t.solution_l2_norm;
        }
      std::cout << "alpha=" << a << " min_eigenvalue=" << s.min_eigenvalue
                << " indefinite=" << yesno(s.min_eigenvalue < 0.0)
                << " l2_error_coarse=" << e[0] << " l2_error_fine=" << e[2]
                << " l2_rate=" << (std::log(e[0] / e[2]) / std::log(4.0))
                << " solution_norm_coarse=" << nrm[0]
                << " solution_norm_fine=" << nrm[2] << std::endl;
    }

  double e_test[3], e_rule[3], n_test[3];
  bool cg_test = false, cg_rule = false, indefinite_test = false;
  for (unsigned int k = 0; k < 3; ++k)
    {
      const SipgResult t = sipg_solve(2 + k, p, alpha, k == 0);
      const SipgResult g = sipg_solve(2 + k, p, rule, false);
      e_test[k] = t.l2_error;
      e_rule[k] = g.l2_error;
      n_test[k] = t.solution_l2_norm;
      if (k == 0)
        {
          cg_test = t.cg_converged;
          cg_rule = g.cg_converged;
          indefinite_test = t.min_eigenvalue < 0.0;
        }
      std::cout << "refine=" << (2 + k) << " n_dofs=" << t.n_dofs
                << " l2_error_under_test=" << t.l2_error
                << " solution_l2_norm_under_test=" << t.solution_l2_norm
                << " l2_error_rule_alpha=" << g.l2_error
                << " cg_steps_under_test=" << t.cg_steps
                << " cg_converged_under_test=" << yesno(t.cg_converged)
                << std::endl;
    }
  const double rate_test = std::log(e_test[0] / e_test[2]) / std::log(4.0);
  const double rate_rule = std::log(e_rule[0] / e_rule[2]) / std::log(4.0);
  std::cout << "l2_rate_under_test=" << rate_test
            << " l2_rate_rule_alpha=" << rate_rule << std::endl;
  std::cout << "matrix_under_test_is_indefinite=" << yesno(indefinite_test)
            << std::endl;
  std::cout << "cg_on_the_matrix_under_test_converged=" << yesno(cg_test)
            << std::endl;
  std::cout << "cg_on_the_rule_penalty_matrix_converged=" << yesno(cg_rule)
            << std::endl;
  // The cross-backend question: does the NORM run away, or only the RATE?
  const bool norm_grows = n_test[2] > 1.5 * n_test[0];
  const bool norm_settles = std::abs(n_test[2] - n_test[1]) < 0.1 * n_test[1];
  const bool error_grows = e_test[2] > e_test[0];
  std::cout << "l2_error_under_test_diverges_under_refinement="
            << yesno(error_grows) << std::endl;
  std::cout << "solution_norm_under_test_grows_under_refinement="
            << yesno(norm_grows) << std::endl;
  std::cout << "solution_norm_under_test_settles_under_refinement="
            << yesno(norm_settles) << std::endl;
  const bool rate_lost = rate_test < 1.5;
  std::cout << "l2_rate_under_test_reaches_order_one_point_five="
            << yesno(!rate_lost) << std::endl;
  std::cout << "rule_alpha_reaches_order_one_point_five="
            << yesno(rate_rule > 1.5) << std::endl;

  // The over-penalised end of the entry.
  {
    const SipgResult s = sipg_solve(2, p, huge, true);
    const double cond = s.max_eigenvalue / std::max(1e-300, s.min_eigenvalue);
    std::cout << "huge_alpha=" << huge << " min_eigenvalue=" << s.min_eigenvalue
              << " max_eigenvalue=" << s.max_eigenvalue
              << " condition_number=" << cond << std::endl;
    std::cout << "huge_alpha_gmres_steps=" << s.gmres_steps
              << " gmres_converged=" << yesno(s.gmres_converged)
              << " l2_error=" << s.l2_error << std::endl;
    std::cout << "huge_alpha_condition_number_above_1e14="
              << yesno(cond > 1e14) << std::endl;
    std::cout << "huge_alpha_gmres_stagnated=" << yesno(!s.gmres_converged)
              << std::endl;
  }
  std::cout << "VERDICT="
            << (indefinite_test
                  ? "under_penalised_sipg_loses_definiteness_but_not_the_rate"
                  : "penalty_under_test_keeps_the_form_positive_definite")
            << std::endl;
  return 0;
}

// ===========================================================================
// phase_field#0 and #1 -- the SUPG stabilisation of a 1D exponential layer.
// -eps u'' + b u' = 0 on (0,1) with u(0)=0, u(1)=1, discretised on a strip of
// the unit square with FE_Q(1) and two cells across, so the problem is 1D and
// the exact solution is known in closed form.
// ===========================================================================
enum TauMode
{
  TAU_NONE = 0, // plain Galerkin
  TAU_BARE,     // h / (2|b|)
  TAU_ASYMPT    // h / (2|b|) * (coth(Pe) - 1/Pe)
};

static const char *tau_name(TauMode m)
{
  return m == TAU_NONE ? "plain_galerkin"
                       : (m == TAU_BARE ? "bare_h_over_2b" : "doubly_asymptotic");
}

struct LayerResult
{
  double min_value = 0.0, max_value = 0.0;
  double max_nodal_error = 0.0;
  double last_element_slope = 0.0, exact_last_element_slope = 0.0;
  double kelly_total = 0.0;
  double cell_peclet = 0.0;
  unsigned int n_dofs = 0;
};

static double layer_exact(double x, double eps, double b)
{
  // (exp(b x/eps) - 1)/(exp(b/eps) - 1), written so that b/eps ~ 1000 is safe.
  const double a = b / eps;
  return (std::exp(a * (x - 1.0)) - std::exp(-a)) / (1.0 - std::exp(-a));
}

class LayerExact : public Function<dim>
{
public:
  LayerExact(double eps, double b)
    : eps(eps)
    , b(b)
  {}
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return layer_exact(p[0], eps, b);
  }

private:
  const double eps, b;
};

static LayerResult layer_solve(unsigned int nx, double eps, double b,
                               TauMode mode)
{
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_rectangle(
    tria, {nx, 2u}, Point<dim>(0.0, 0.0), Point<dim>(1.0, 1.0), true);
  FE_Q<dim>       fe(1);
  MappingQ1<dim>  mapping;
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);
  const double h = 1.0 / nx;
  const double pe = std::abs(b) * h / (2.0 * eps);
  double tau = 0.0;
  if (mode == TAU_BARE)
    tau = h / (2.0 * std::abs(b));
  else if (mode == TAU_ASYMPT)
    tau = h / (2.0 * std::abs(b)) *
          ((std::cosh(pe) / std::sinh(pe)) - 1.0 / pe);

  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(
    dof, 0, Functions::ZeroFunction<dim>(), constraints); // x = 0
  VectorTools::interpolate_boundary_values(
    dof, 1, Functions::ConstantFunction<dim>(1.0), constraints); // x = 1
  constraints.close();

  DynamicSparsityPattern dsp(dof.n_dofs());
  DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A;
  A.reinit(sp);
  Vector<double> rhs(dof.n_dofs()), sol(dof.n_dofs());

  QGauss<dim>   quad(3);
  FEValues<dim> fev(mapping, fe, quad,
                    update_values | update_gradients | update_JxW_values);
  const unsigned int n = fe.n_dofs_per_cell();
  FullMatrix<double> cm(n, n);
  Vector<double>     cr(n);
  std::vector<types::global_dof_index> local(n);
  Tensor<1, dim> beta;
  beta[0] = b;
  beta[1] = 0.0;

  for (const auto &cell : dof.active_cell_iterators())
    {
      fev.reinit(cell);
      cm = 0.0;
      cr = 0.0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            cm(i, j) += (eps * fev.shape_grad(i, q) * fev.shape_grad(j, q) +
                         fev.shape_value(i, q) *
                           (beta * fev.shape_grad(j, q)) +
                         tau * (beta * fev.shape_grad(i, q)) *
                           (beta * fev.shape_grad(j, q))) *
                        fev.JxW(q);
      cell->get_dof_indices(local);
      constraints.distribute_local_to_global(cm, cr, local, A, rhs);
    }
  {
    SparseDirectUMFPACK inv;
    inv.initialize(A);
    sol = rhs;
    inv.solve(sol);
  }
  constraints.distribute(sol);

  LayerResult r;
  r.n_dofs = dof.n_dofs();
  r.cell_peclet = pe;
  r.min_value = sol.linfty_norm() > 0 ? *std::min_element(sol.begin(), sol.end())
                                      : 0.0;
  r.max_value = *std::max_element(sol.begin(), sol.end());

  // Nodal error, and the slope of the last element, both read at the support
  // points so no quadrature smoothing hides anything.
  std::vector<Point<dim>> pts(dof.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, dof, pts);
  double worst = 0.0;
  double at_one = 0.0, at_one_minus_h = 0.0;
  for (unsigned int i = 0; i < dof.n_dofs(); ++i)
    {
      worst = std::max(worst,
                       std::abs(sol(i) - layer_exact(pts[i][0], eps, b)));
      if (std::abs(pts[i][1]) < 1e-12)
        {
          if (std::abs(pts[i][0] - 1.0) < 1e-12)
            at_one = sol(i);
          if (std::abs(pts[i][0] - (1.0 - h)) < 1e-12)
            at_one_minus_h = sol(i);
        }
    }
  r.max_nodal_error = worst;
  r.last_element_slope = (at_one - at_one_minus_h) / h;
  r.exact_last_element_slope =
    (layer_exact(1.0, eps, b) - layer_exact(1.0 - h, eps, b)) / h;

  Vector<float> est(tria.n_active_cells());
  KellyErrorEstimator<dim>::estimate(mapping, dof, QGauss<dim - 1>(3),
                                     std::map<types::boundary_id,
                                              const Function<dim> *>(),
                                     sol, est);
  r.kelly_total = est.l2_norm();
  return r;
}

// phase_field#0 -- Galerkin wiggles, and whether refinement removes them.
static int galerkin_wiggles()
{
  const double eps = 1e-3, b = 1.0;
  const TauMode mode = mutate() ? TAU_ASYMPT : TAU_NONE;
  std::cout << "scheme_under_test=" << tau_name(mode) << " eps=" << eps
            << std::endl;
  bool overshoot_above_one = false, undershoot_below_zero = false;
  bool clean_below_one = true;
  double worst_pe_above_one = 0.0;
  for (unsigned int nx : {8u, 32u, 128u, 256u, 512u, 1024u})
    {
      const LayerResult r = layer_solve(nx, eps, b, mode);
      const bool out_of_range = (r.min_value < -1e-6) || (r.max_value > 1.0 + 1e-6);
      std::cout << "cells_in_x=" << nx << " cell_peclet=" << r.cell_peclet
                << " min_value=" << r.min_value << " max_value=" << r.max_value
                << " max_nodal_error=" << r.max_nodal_error
                << " leaves_the_zero_one_range=" << yesno(out_of_range)
                << std::endl;
      if (r.cell_peclet > 1.0)
        {
          if (out_of_range)
            {
              undershoot_below_zero =
                undershoot_below_zero || (r.min_value < -1e-6);
              overshoot_above_one =
                overshoot_above_one || (r.max_value > 1.0 + 1e-6);
              worst_pe_above_one = std::max(worst_pe_above_one, r.cell_peclet);
            }
        }
      else if (out_of_range)
        clean_below_one = false;
    }
  std::cout << "oscillation_under_test_while_cell_peclet_exceeds_one="
            << yesno(undershoot_below_zero || overshoot_above_one) << std::endl;
  std::cout << "oscillation_under_test_survives_several_refinements_above_peclet_one="
            << yesno(worst_pe_above_one > 0.0) << std::endl;
  std::cout << "oscillation_under_test_gone_once_cell_peclet_drops_below_one="
            << yesno(clean_below_one) << std::endl;
  std::cout << "VERDICT="
            << ((undershoot_below_zero || overshoot_above_one)
                  ? "scheme_under_test_oscillates_until_the_cell_peclet_drops_below_one"
                  : "scheme_under_test_stays_inside_the_data_range")
            << std::endl;
  return 0;
}

// phase_field#1 -- a FIXED tau over-stabilises where the mesh already resolves.
static int fixed_supg_tau()
{
  const double b = 1.0;
  const TauMode mode = mutate() ? TAU_ASYMPT : TAU_BARE;
  std::cout << "tau_under_test=" << tau_name(mode) << std::endl;
  const unsigned int nx = 20;
  const double h = 1.0 / nx;
  bool worse_than_galerkin = false;
  bool asympt_nodally_exact = true;
  double smear_low = 0.0, smear_mid = 0.0;
  bool kelly_larger_low = false, kelly_larger_mid = false;
  for (double pe : {0.05, 0.5})
    {
      const double eps = std::abs(b) * h / (2.0 * pe);
      const LayerResult t = layer_solve(nx, eps, b, mode);
      const LayerResult g = layer_solve(nx, eps, b, TAU_NONE);
      const LayerResult a = layer_solve(nx, eps, b, TAU_ASYMPT);
      const double smear =
        1.0 - t.last_element_slope / t.exact_last_element_slope;
      std::cout << "cell_peclet=" << pe << " eps=" << eps
                << " max_nodal_error_under_test=" << t.max_nodal_error
                << " plain_galerkin_max_nodal_error=" << g.max_nodal_error
                << " doubly_asymptotic_max_nodal_error=" << a.max_nodal_error
                << std::endl;
      std::cout << "cell_peclet=" << pe
                << " last_element_slope_under_test=" << t.last_element_slope
                << " exact_last_element_slope=" << t.exact_last_element_slope
                << " relative_gradient_smearing_under_test=" << smear
                << std::endl;
      std::cout << "cell_peclet=" << pe
                << " kelly_total_under_test=" << t.kelly_total
                << " kelly_total_doubly_asymptotic=" << a.kelly_total
                << std::endl;
      if (pe < 0.1)
        {
          worse_than_galerkin = t.max_nodal_error > 2.0 * g.max_nodal_error;
          smear_low = smear;
          kelly_larger_low = t.kelly_total > a.kelly_total;
        }
      else
        {
          smear_mid = smear;
          kelly_larger_mid = t.kelly_total > a.kelly_total;
        }
      asympt_nodally_exact = asympt_nodally_exact && (a.max_nodal_error < 1e-10);
    }
  std::cout << "tau_under_test_is_worse_than_plain_galerkin_in_the_diffusive_regime="
            << yesno(worse_than_galerkin) << std::endl;
  std::cout << "smearing_reaches_twenty_percent_at_cell_peclet_one_half="
            << yesno(smear_mid > 0.20) << std::endl;
  std::cout << "smearing_reaches_twenty_percent_at_cell_peclet_one_twentieth="
            << yesno(smear_low > 0.20) << std::endl;
  std::cout << "smearing_exceeds_one_percent_at_cell_peclet_one_twentieth="
            << yesno(smear_low > 0.01) << std::endl;
  std::cout << "doubly_asymptotic_tau_is_nodally_exact="
            << yesno(asympt_nodally_exact) << std::endl;
  std::cout << "kelly_total_under_test_exceeds_the_doubly_asymptotic_one_at_either_peclet="
            << yesno(kelly_larger_low || kelly_larger_mid) << std::endl;
  std::cout << "VERDICT="
            << (worse_than_galerkin
                  ? "fixed_tau_over_stabilises_where_the_mesh_already_resolves"
                  : "tau_under_test_does_not_over_stabilise")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  MultithreadInfo::set_thread_limit(1);
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  std::cout << std::setprecision(8);
  if (probe == "sipg_penalty")
    return sipg_penalty();
  if (probe == "galerkin_wiggles")
    return galerkin_wiggles();
  if (probe == "fixed_supg_tau")
    return fixed_supg_tau();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
