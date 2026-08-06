// Shared translation unit for the nonlinear (Newton) Signal family.
// One compile serves several fixture directories; each fixture runs one probe.
//
// usage: nonlinear_family <probe>
//   cold_start | line_search | stale_linearisation
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.
//
// The model problem is the minimal surface equation of step-15,
//    -div( (1 + |grad u|^2)^{-1/2} grad u ) = 0,   u = g on the boundary,
// on the unit square with g = A sin(2 pi (x + y)). deal.II has NO Newton
// solver, so the loop below is the user's own -- which is exactly the point of
// this family of claims.

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
#include <deal.II/lac/solver_control.h>
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
constexpr int dim = 2;

static bool mutate()
{
  const char *m = std::getenv("T2_MUTATE");
  return m != nullptr && std::string(m) == "1";
}

class BoundaryValues : public Function<dim>
{
public:
  const double A;
  BoundaryValues(double a)
    : A(a)
  {}
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return A * std::sin(2.0 * numbers::PI * (p[0] + p[1]));
  }
};

struct MinimalSurface
{
  Triangulation<dim>        tria;
  FE_Q<dim>                 fe;
  DoFHandler<dim>           dof;
  AffineConstraints<double> zero_bc;   // the Newton UPDATE is zero on the bdry
  SparsityPattern           sp;
  SparseMatrix<double>      A;
  Vector<double>            sol, update, rhs, frozen;
  bool                      inner_solver_failed = false;
  std::string               inner_solver_message;

  MinimalSurface()
    : fe(1)
    , dof(tria)
  {}

  void setup(unsigned int refine)
  {
    GridGenerator::hyper_cube(tria, 0.0, 1.0, false);
    tria.refine_global(refine);
    dof.distribute_dofs(fe);
    zero_bc.clear();
    VectorTools::interpolate_boundary_values(
      dof, 0, Functions::ZeroFunction<dim>(), zero_bc);
    zero_bc.close();
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_sparsity_pattern(dof, dsp, zero_bc, false);
    sp.copy_from(dsp);
    A.reinit(sp);
    sol.reinit(dof.n_dofs());
    update.reinit(dof.n_dofs());
    rhs.reinit(dof.n_dofs());
    frozen.reinit(dof.n_dofs());
  }

  void set_boundary_values(double amplitude)
  {
    std::map<types::global_dof_index, double> bv;
    VectorTools::interpolate_boundary_values(dof, 0, BoundaryValues(amplitude),
                                             bv);
    for (const auto &p : bv)
      sol(p.first) = p.second;
  }

  double boundary_error(double amplitude) const
  {
    std::map<types::global_dof_index, double> bv;
    VectorTools::interpolate_boundary_values(dof, 0, BoundaryValues(amplitude),
                                             bv);
    double worst = 0.0;
    for (const auto &p : bv)
      worst = std::max(worst, std::abs(sol(p.first) - p.second));
    return worst;
  }

  // Assemble the Newton system at `sol`. freeze_coefficient=true uses the
  // nonlinear coefficient evaluated at `frozen` instead of at `sol` -- the
  // "linearisation was never updated" mistake.
  void assemble(bool freeze_coefficient)
  {
    A   = 0.0;
    rhs = 0.0;
    QGauss<dim>   quad(fe.degree + 2);
    FEValues<dim> fev(fe, quad,
                      update_gradients | update_quadrature_points |
                        update_JxW_values);
    const unsigned int                   n = fe.dofs_per_cell;
    FullMatrix<double>                   cm(n, n);
    Vector<double>                       cv(n);
    std::vector<types::global_dof_index> local(n);
    std::vector<Tensor<1, dim>>          gu(quad.size()), gc(quad.size());
    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        cv = 0.0;
        fev.get_function_gradients(sol, gu);
        fev.get_function_gradients(freeze_coefficient ? frozen : sol, gc);
        for (unsigned int q = 0; q < quad.size(); ++q)
          {
            const double c = 1.0 / std::sqrt(1.0 + gc[q] * gc[q]);
            for (unsigned int i = 0; i < n; ++i)
              {
                for (unsigned int j = 0; j < n; ++j)
                  {
                    cm(i, j) += (fev.shape_grad(i, q) * c *
                                 fev.shape_grad(j, q)) *
                                fev.JxW(q);
                    if (!freeze_coefficient)
                      cm(i, j) -= (fev.shape_grad(i, q) * c * c * c *
                                   (gu[q] * fev.shape_grad(j, q)) * gu[q]) *
                                  fev.JxW(q);
                  }
                cv(i) -= (fev.shape_grad(i, q) * c * gu[q]) * fev.JxW(q);
              }
          }
        cell->get_dof_indices(local);
        zero_bc.distribute_local_to_global(cm, cv, local, A, rhs);
      }
  }

  // ||F(sol + alpha*update)|| with the boundary rows removed, exactly as
  // step-15's compute_residual does.
  double residual_at(double alpha, bool freeze_coefficient)
  {
    Vector<double> eval(sol);
    eval.add(alpha, update);
    Vector<double> res(dof.n_dofs());
    QGauss<dim>    quad(fe.degree + 2);
    FEValues<dim>  fev(fe, quad,
                       update_gradients | update_quadrature_points |
                         update_JxW_values);
    const unsigned int                   n = fe.dofs_per_cell;
    Vector<double>                       cv(n);
    std::vector<types::global_dof_index> local(n);
    std::vector<Tensor<1, dim>>          gu(quad.size()), gc(quad.size());
    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cv = 0.0;
        fev.get_function_gradients(eval, gu);
        fev.get_function_gradients(freeze_coefficient ? frozen : eval, gc);
        for (unsigned int q = 0; q < quad.size(); ++q)
          {
            const double c = 1.0 / std::sqrt(1.0 + gc[q] * gc[q]);
            for (unsigned int i = 0; i < n; ++i)
              cv(i) -= (fev.shape_grad(i, q) * c * gu[q]) * fev.JxW(q);
          }
        cell->get_dof_indices(local);
        for (unsigned int i = 0; i < n; ++i)
          res(local[i]) += cv(i);
      }
    for (const auto &line : zero_bc.get_lines())
      res(line.index) = 0.0;
    return res.l2_norm();
  }

  bool solve_update()
  {
    inner_solver_failed = false;
    SolverControl ctrl(2000, 1e-10 * std::max(1e-30, rhs.l2_norm()));
    SolverCG<Vector<double>>               cg(ctrl);
    PreconditionSSOR<SparseMatrix<double>> prec;
    try
      {
        prec.initialize(A, 1.2);
        cg.solve(A, update, rhs, prec);
      }
    catch (const std::exception &e)
      {
        inner_solver_failed = true;
        const std::string w(e.what());
        inner_solver_message = w.substr(0, std::min<size_t>(w.size(), 800));
        return false;
      }
    zero_bc.distribute(update);
    return true;
  }

  // Returns the number of Newton steps taken; `converged` says whether the
  // residual reached the tolerance.
  unsigned int newton(unsigned int max_steps, bool line_search,
                      bool freeze_coefficient, double tol,
                      std::vector<double> &history, bool &converged,
                      bool &diverged)
  {
    converged = false;
    diverged  = false;
    unsigned int step = 0;
    double       r    = residual_at(0.0, freeze_coefficient);
    history.push_back(r);
    if (r < tol)
      {
        converged = true;
        return 0;
      }
    for (step = 1; step <= max_steps; ++step)
      {
        assemble(freeze_coefficient);
        if (!solve_update())
          {
            diverged = true;
            return step;
          }
        double alpha = 1.0;
        if (line_search)
          {
            const double r0 = r;
            while (alpha > 1e-5 &&
                   residual_at(alpha, freeze_coefficient) >= r0)
              alpha *= 0.5;
          }
        sol.add(alpha, update);
        r = residual_at(0.0, freeze_coefficient);
        history.push_back(r);
        if (!std::isfinite(r) || r > 1e8 * history[0])
          {
            diverged = true;
            return step;
          }
        if (r < tol)
          {
            converged = true;
            return step;
          }
      }
    return step;
  }
};

static void print_history(const char *tag, const std::vector<double> &h)
{
  std::cout << tag << "_residual_history=";
  for (unsigned int i = 0; i < h.size() && i < 12; ++i)
    std::cout << (i ? "," : "") << h[i];
  std::cout << std::endl;
}

// ---------------------------------------------------------------------------
// nonlinear#0 -- the initial guess.
// ---------------------------------------------------------------------------
static int cold_start()
{
  const double amplitude = 8.0;
  const double tol       = 1e-8;
  std::cout << "boundary_amplitude=" << amplitude << std::endl;
  // Three starting points are run in EVERY invocation, so the comparison the
  // claim is about is visible whichever one is under test.

  struct Result
  {
    unsigned int        steps = 0;
    bool                converged = false, diverged = false;
    double              final_residual = 0.0, bnd_err = 0.0;
    std::vector<double> hist;
    std::string         inner;
  };

  // (a) the literal cold start: an all-zero vector, boundary values never put
  //     into the guess
  Result cold;
  {
    MinimalSurface m;
    m.setup(5);
    cold.steps = m.newton(25, true, false, tol, cold.hist, cold.converged,
                          cold.diverged);
    cold.final_residual = cold.hist.back();
    cold.bnd_err        = m.boundary_error(amplitude);
    cold.inner          = m.inner_solver_message;
  }

  // (b) the same zero interior, but WITH the boundary values interpolated onto
  //     the guess -- the claim's first fix
  Result withbv;
  {
    MinimalSurface m;
    m.setup(5);
    m.set_boundary_values(amplitude);
    withbv.steps = m.newton(25, true, false, tol, withbv.hist,
                            withbv.converged, withbv.diverged);
    withbv.final_residual = withbv.hist.back();
    withbv.bnd_err        = m.boundary_error(amplitude);
    withbv.inner          = m.inner_solver_message;
  }

  // (c) continuation: ramp the boundary amplitude in load steps, each starting
  //     from the previous step's solution -- the claim's second fix
  Result cont;
  {
    MinimalSurface m;
    m.setup(5);
    const unsigned int nload = 5;
    for (unsigned int L = 1; L <= nload; ++L)
      {
        m.set_boundary_values(amplitude * L / nload);
        std::vector<double> h;
        bool                cv, dv;
        cont.steps += m.newton(25, true, false, tol, h, cv, dv);
        cont.converged = cv;
        cont.diverged  = dv;
        if (L == nload)
          cont.hist = h;
      }
    cont.final_residual = cont.hist.back();
    cont.bnd_err        = m.boundary_error(amplitude);
    cont.inner          = m.inner_solver_message;
  }

  auto report = [&](const char *tag, const Result &r) {
    std::cout << tag << "_newton_steps=" << r.steps
              << " converged=" << (r.converged ? "true" : "false")
              << " diverged=" << (r.diverged ? "true" : "false")
              << " final_residual=" << r.final_residual
              << " max_boundary_violation=" << r.bnd_err << std::endl;
    print_history(tag, r.hist);
    if (!r.inner.empty())
      std::cout << tag << "_inner_solver_message=" << r.inner << std::endl;
  };
  report("zero_guess_without_boundary_values", cold);
  report("zero_guess_with_boundary_values", withbv);
  report("continuation", cont);

  const Result &under = mutate() ? cont : cold;
  std::cout << "initial_guess_under_test="
            << (mutate() ? "continuation_from_the_previous_load_step"
                         : "all_zero_vector_no_boundary_values")
            << std::endl;
  const bool bc_ok = under.bnd_err < 1e-10;
  std::cout << "boundary_condition_satisfied=" << (bc_ok ? "true" : "false")
            << std::endl;
  std::cout << "reported_convergence=" << (under.converged ? "true" : "false")
            << std::endl;
  // the claim's own Signal: a residual ABOVE 1e3 at step 1 that grows further
  const bool claimed_divergence =
    under.hist.size() > 2 && under.hist[1] > 1e3 &&
    under.hist[2] > under.hist[1];
  std::cout << "residual_grows_past_1e3_as_the_claim_says="
            << (claimed_divergence ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << ((under.converged && !bc_ok)
                  ? "cold_start_reports_convergence_without_the_boundary_data"
                  : (bc_ok ? "guess_leads_to_the_boundary_data"
                           : "newton_failed"))
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// nonlinear#1 -- the line search. Both variants use the SAME initial guess, so
// the only difference is whether the step length is backtracked.
// ---------------------------------------------------------------------------
static int line_search_probe()
{
  const double amplitude = 32.0;
  const double tol       = 1e-8;
  const bool   use_ls    = mutate();
  std::cout << "boundary_amplitude=" << amplitude << std::endl;
  std::cout << "step_length=" << (use_ls ? "backtracking_line_search"
                                         : "full_newton_step_alpha_1")
            << std::endl;
  MinimalSurface m;
  m.setup(5);
  m.set_boundary_values(amplitude);
  std::vector<double> hist;
  bool                converged = false, diverged = false;
  const unsigned int  steps =
    m.newton(30, use_ls, false, tol, hist, converged, diverged);
  std::cout << "newton_steps=" << steps
            << " converged=" << (converged ? "true" : "false")
            << " diverged=" << (diverged ? "true" : "false") << std::endl;
  print_history("newton", hist);
  if (!m.inner_solver_message.empty())
    std::cout << "inner_solver_message=" << m.inner_solver_message
              << std::endl;
  bool monotone = true;
  for (unsigned int i = 1; i < hist.size(); ++i)
    if (!(hist[i] < hist[i - 1]))
      monotone = false;
  std::cout << "residual_decreased_monotonically="
            << (monotone ? "true" : "false") << std::endl;
  std::cout << "final_residual=" << hist.back() << std::endl;
  std::cout << "VERDICT="
            << ((converged && monotone)
                  ? "line_search_keeps_the_residual_decreasing"
                  : "full_step_newton_does_not_converge")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// nonlinear#2 -- a linearisation point that is never updated.
// ---------------------------------------------------------------------------
static int stale_linearisation()
{
  const double amplitude = 32.0;
  const double tol       = 1e-8;
  const bool   freeze    = !mutate();
  std::cout << "boundary_amplitude=" << amplitude << std::endl;
  std::cout << "linearisation_point="
            << (freeze ? "frozen_at_the_initial_guess"
                       : "updated_every_newton_step")
            << std::endl;

  // the properly converged answer, computed in the same run as the reference
  MinimalSurface ref;
  ref.setup(5);
  ref.set_boundary_values(amplitude);
  std::vector<double> rh;
  bool                rc, rd;
  const unsigned int  rsteps = ref.newton(40, true, false, tol, rh, rc, rd);
  std::cout << "reference_newton_steps=" << rsteps
            << " reference_converged=" << (rc ? "true" : "false")
            << " reference_final_residual=" << rh.back() << std::endl;

  MinimalSurface m;
  m.setup(5);
  m.set_boundary_values(amplitude);
  m.frozen = m.sol;   // the linearisation point the mistake never leaves
  std::vector<double> hist;
  bool                converged = false, diverged = false;
  const unsigned int  steps =
    m.newton(40, true, freeze, tol, hist, converged, diverged);
  std::cout << "newton_steps=" << steps
            << " converged=" << (converged ? "true" : "false") << std::endl;
  print_history("newton", hist);
  std::cout << "reported_final_residual=" << hist.back() << std::endl;
  // the residual of the REAL nonlinear problem at the answer that was accepted
  const double true_residual = m.residual_at(0.0, false);
  std::cout << "true_nonlinear_residual_at_the_accepted_answer="
            << true_residual << std::endl;
  Vector<double> diff(m.sol);
  diff -= ref.sol;
  std::cout << "l2_difference_from_the_reference_solution=" << diff.l2_norm()
            << " reference_l2_norm=" << ref.sol.l2_norm() << std::endl;
  const double rel = diff.l2_norm() / ref.sol.l2_norm();
  std::cout << "relative_difference_from_the_reference=" << rel << std::endl;
  const bool looks_converged = converged && hist.back() < tol;
  const bool wrong_answer    = rel > 1e-3;
  std::cout << "reports_convergence=" << (looks_converged ? "true" : "false")
            << std::endl;
  std::cout << "answer_differs_from_the_reference="
            << (wrong_answer ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << ((looks_converged && wrong_answer)
                  ? "stale_linearisation_converges_to_the_wrong_problem"
                  : "answer_matches_the_reference")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  if (probe == "cold_start")
    return cold_start();
  if (probe == "line_search")
    return line_search_probe();
  if (probe == "stale_linearisation")
    return stale_linearisation();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
