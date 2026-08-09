// Shared translation unit for the geometric-multigrid Signal family, in the
// shape of step-16: FE_Q(2) Laplace on the unit square with a level hierarchy,
// MGTransferPrebuilt, an SOR relaxation smoother and a dense direct coarse
// solve. One switch at a time is bent by each probe.
//
// usage: multigrid_family <probe> [sub]
//   mg_no_level_dofs <n_dofs|transfer|constrained> | mg_boundary_indices
//   | mg_smoother_on_indefinite | mg_coarse_solver | mf_cg_needs_a_preconditioner
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.
//
// mg_no_level_dofs ABORTS (Debug) or SEGFAULTS (Release) by design and is run in
// its own process by the fixture's cmd.sh, which pins both exit codes.

#include <deal.II/base/function.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
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
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <cstdlib>
#include <functional>
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
// The shared problem. shift > 0 turns the SPD Laplace into the INDEFINITE
// operator K - shift * M, which is what multigrid#2 is about.
// ===========================================================================
struct MG
{
  Triangulation<dim>        tria;
  FE_Q<dim>                 fe;
  MappingQ1<dim>            mapping;
  DoFHandler<dim>           dof;
  AffineConstraints<double> constraints;
  SparsityPattern           sp;
  SparseMatrix<double>      A;
  Vector<double>            rhs;
  MGLevelObject<SparsityPattern>      mg_sp;
  MGLevelObject<SparseMatrix<double>> mg_matrices;
  MGConstrainedDoFs                   mg_constrained_dofs;
  double                              shift = 0.0;

  MG()
    : fe(2)
    , dof(tria)
  {}

  void make_grid(unsigned int refine, bool adaptive = false)
  {
    GridGenerator::hyper_cube(tria, 0.0, 1.0);
    tria.refine_global(refine);
    if (adaptive)
      {
        // One extra refinement in a corner, so the hierarchy really has a
        // refinement edge to find.
        for (const auto &cell : tria.active_cell_iterators())
          if (cell->center()[0] < 0.25 && cell->center()[1] < 0.25)
            cell->set_refine_flag();
        tria.execute_coarsening_and_refinement();
      }
  }

  void distribute(bool with_level_dofs)
  {
    dof.distribute_dofs(fe);
    if (with_level_dofs)
      dof.distribute_mg_dofs();
  }

  void setup_constraints(bool zero_boundary_for_mg)
  {
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof, constraints);
    VectorTools::interpolate_boundary_values(
      dof, 0, Functions::ZeroFunction<dim>(), constraints);
    constraints.close();
    mg_constrained_dofs.clear();
    mg_constrained_dofs.initialize(dof);
    if (zero_boundary_for_mg)
      mg_constrained_dofs.make_zero_boundary_constraints(dof, {0});
  }

  void cell_matrix(FEValues<dim> &fev, FullMatrix<double> &cm) const
  {
    const unsigned int n = fe.n_dofs_per_cell();
    cm = 0.0;
    for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
      for (unsigned int i = 0; i < n; ++i)
        for (unsigned int j = 0; j < n; ++j)
          cm(i, j) += (fev.shape_grad(i, q) * fev.shape_grad(j, q) -
                       shift * fev.shape_value(i, q) * fev.shape_value(j, q)) *
                      fev.JxW(q);
  }

  void assemble_global()
  {
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
    sp.copy_from(dsp);
    A.reinit(sp);
    rhs.reinit(dof.n_dofs());
    QGauss<dim>   quad(fe.degree + 1);
    FEValues<dim> fev(mapping, fe, quad,
                      update_values | update_gradients | update_JxW_values);
    const unsigned int n = fe.n_dofs_per_cell();
    FullMatrix<double> cm(n, n);
    Vector<double>     cr(n);
    std::vector<types::global_dof_index> local(n);
    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cell_matrix(fev, cm);
        cr = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < n; ++i)
            cr(i) += 1.0 * fev.shape_value(i, q) * fev.JxW(q);
        cell->get_dof_indices(local);
        constraints.distribute_local_to_global(cm, cr, local, A, rhs);
      }
  }

  void assemble_levels()
  {
    const unsigned int nlevels = tria.n_levels();
    mg_sp.resize(0, nlevels - 1);
    mg_matrices.resize(0, nlevels - 1);
    for (unsigned int l = 0; l < nlevels; ++l)
      {
        DynamicSparsityPattern dsp(dof.n_dofs(l), dof.n_dofs(l));
        MGTools::make_sparsity_pattern(dof, dsp, l);
        mg_sp[l].copy_from(dsp);
        mg_matrices[l].reinit(mg_sp[l]);
      }
    std::vector<AffineConstraints<double>> level_constraints(nlevels);
    for (unsigned int l = 0; l < nlevels; ++l)
      {
        level_constraints[l].clear();
        if (mg_constrained_dofs.have_boundary_indices())
          for (const auto i : mg_constrained_dofs.get_boundary_indices(l))
            level_constraints[l].constrain_dof_to_zero(i);
        level_constraints[l].close();
      }
    QGauss<dim>   quad(fe.degree + 1);
    FEValues<dim> fev(mapping, fe, quad,
                      update_values | update_gradients | update_JxW_values);
    const unsigned int n = fe.n_dofs_per_cell();
    FullMatrix<double> cm(n, n);
    std::vector<types::global_dof_index> local(n);
    for (const auto &cell : dof.cell_iterators())
      {
        fev.reinit(cell);
        cell_matrix(fev, cm);
        cell->get_mg_dof_indices(local);
        level_constraints[cell->level()].distribute_local_to_global(
          cm, local, mg_matrices[cell->level()]);
      }
  }
};

// A CG solve with a geometric-multigrid preconditioner. coarse_exact selects a
// dense direct coarse solve (true) or the same relaxation smoother (false);
// min_level says how far down the hierarchy the V-cycle goes.
struct MGRun
{
  unsigned int steps = 0;
  bool         converged = false;
  double       last_value = 0.0;
  std::vector<double> vcycle_residuals;
};

static MGRun mg_solve(MG &p, bool coarse_exact, unsigned int min_level,
                      bool record_vcycles)
{
  using Smoother = PreconditionSOR<SparseMatrix<double>>;
  MGTransferPrebuilt<Vector<double>> transfer(p.mg_constrained_dofs);
  transfer.build(p.dof);

  mg::SmootherRelaxation<Smoother, Vector<double>> smoother;
  smoother.initialize(p.mg_matrices);
  smoother.set_steps(2);
  smoother.set_symmetric(true);

  FullMatrix<double> coarse_matrix;
  coarse_matrix.copy_from(p.mg_matrices[min_level]);
  MGCoarseGridHouseholder<double, Vector<double>> coarse_direct;
  coarse_direct.initialize(coarse_matrix);
  MGCoarseGridApplySmoother<Vector<double>> coarse_smoothed(smoother);
  MGCoarseGridBase<Vector<double>> *coarse =
    coarse_exact ? static_cast<MGCoarseGridBase<Vector<double>> *>(&coarse_direct)
                 : static_cast<MGCoarseGridBase<Vector<double>> *>(
                     &coarse_smoothed);

  mg::Matrix<Vector<double>> mg_matrix(p.mg_matrices);
  Multigrid<Vector<double>>  mg(mg_matrix, *coarse, transfer, smoother,
                                smoother, min_level);
  PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double>>>
    prec(p.dof, mg, transfer);

  MGRun r;
  if (record_vcycles)
    {
      // The V-cycle as a stationary iteration, so the residual of each cycle is
      // visible rather than hidden inside a Krylov method.
      Vector<double> x(p.dof.n_dofs()), res(p.dof.n_dofs()),
        dx(p.dof.n_dofs());
      res = p.rhs;
      r.vcycle_residuals.push_back(res.l2_norm());
      for (unsigned int c = 0; c < 8; ++c)
        {
          prec.vmult(dx, res);
          x += dx;
          p.A.vmult(res, x);
          res.sadd(-1.0, 1.0, p.rhs);
          r.vcycle_residuals.push_back(res.l2_norm());
          if (!std::isfinite(res.l2_norm()) || res.l2_norm() > 1e30)
            break;
        }
    }
  SolverControl control(500, 1e-9 * std::max(1e-300, p.rhs.l2_norm()));
  SolverCG<Vector<double>> cg(control);
  Vector<double>           x(p.dof.n_dofs());
  try
    {
      cg.solve(p.A, x, p.rhs, prec);
      r.converged = true;
    }
  catch (const std::exception &)
    {}
  r.steps = control.last_step();
  r.last_value = control.last_value();
  return r;
}

// ===========================================================================
// multigrid#0 -- forgetting distribute_mg_dofs().
// ===========================================================================
static int mg_no_level_dofs(const std::string &which)
{
  MG p;
  p.make_grid(3);
  p.distribute(mutate()); // the mistake: no distribute_mg_dofs()
  // NOTHING else touches an mg_ API before the one call under test, so the
  // crash that follows is attributable to that call and no other.
  std::cout << "called_distribute_mg_dofs=" << yesno(mutate())
            << " has_level_dofs=" << yesno(p.dof.has_level_dofs())
            << " n_levels=" << p.tria.n_levels() << std::endl;
  std::cout << "cheap_guard_has_level_dofs_is_true="
            << yesno(p.dof.has_level_dofs()) << std::endl;
  std::cout << "call_under_test=" << which << std::endl;
  std::cout << "before_the_call" << std::endl;
  if (which == "n_dofs")
    std::cout << "n_dofs_on_the_finest_level="
              << p.dof.n_dofs(p.tria.n_levels() - 1) << std::endl;
  else if (which == "transfer")
    {
      MGConstrainedDoFs mgc;
      MGTransferPrebuilt<Vector<double>> transfer;
      transfer.build(p.dof);
      std::cout << "transfer_built=true" << std::endl;
    }
  else
    {
      MGConstrainedDoFs mgc;
      mgc.initialize(p.dof);
      std::cout << "mg_constrained_dofs_initialized=true" << std::endl;
    }
  std::cout << "after_the_call" << std::endl;
  return 0;
}

// ===========================================================================
// multigrid#1 -- MGConstrainedDoFs must be told about the Dirichlet boundary.
// ===========================================================================
static int mg_boundary_indices()
{
  const bool tell_it = mutate(); // the mistake: skip make_zero_boundary_constraints
  MG p;
  p.make_grid(4);
  p.distribute(true);
  p.setup_constraints(tell_it);
  p.assemble_global();
  p.assemble_levels();
  std::cout << "make_zero_boundary_constraints_called=" << yesno(tell_it)
            << " n_levels=" << p.tria.n_levels() << std::endl;
  const bool have = p.mg_constrained_dofs.have_boundary_indices();
  std::cout << "have_boundary_indices=" << yesno(have) << std::endl;
  bool every_level = have;
  for (unsigned int l = 0; l < p.tria.n_levels(); ++l)
    {
      const std::size_t n =
        have ? p.mg_constrained_dofs.get_boundary_indices(l).n_elements() : 0;
      std::cout << "level=" << l << " level_dofs=" << p.dof.n_dofs(l)
                << " boundary_indices=" << n << std::endl;
      if (n == 0)
        every_level = false;
    }
  std::cout << "every_level_carries_its_own_boundary_index_set="
            << yesno(every_level) << std::endl;

  const MGRun r = mg_solve(p, true, 0, false);
  std::cout << "cg_with_the_v_cycle converged=" << yesno(r.converged)
            << " steps=" << r.steps << " last_value=" << r.last_value
            << std::endl;
  std::cout << "v_cycle_is_a_usable_preconditioner="
            << yesno(r.converged && r.steps < 50) << std::endl;

  // The adaptive half of the entry: refinement-edge index sets.
  {
    MG a;
    a.make_grid(3, true);
    a.distribute(true);
    a.setup_constraints(true);
    bool         nonempty_somewhere = false, empty_somewhere = false;
    for (unsigned int l = 0; l < a.tria.n_levels(); ++l)
      {
        const std::size_t n =
          a.mg_constrained_dofs.get_refinement_edge_indices(l).n_elements();
        std::cout << "adaptive_level=" << l
                  << " refinement_edge_indices=" << n << std::endl;
        if (n > 0)
          nonempty_somewhere = true;
        else
          empty_somewhere = true;
      }
    std::cout << "refinement_edge_indices_are_nonempty_on_some_level="
              << yesno(nonempty_somewhere) << std::endl;
    std::cout << "refinement_edge_indices_are_empty_on_some_level="
              << yesno(empty_somewhere) << std::endl;
  }
  std::cout << "VERDICT="
            << (have ? "mg_constrained_dofs_knows_the_dirichlet_boundary"
                     : "mg_constrained_dofs_was_never_told_about_the_boundary")
            << std::endl;
  return 0;
}

// ===========================================================================
// multigrid#2 -- an SPD smoother on an indefinite operator.
// ===========================================================================
static int mg_smoother_on_indefinite()
{
  const double shift = mutate() ? 0.0 : 300.0;
  MG p;
  p.shift = shift;
  p.make_grid(4);
  p.distribute(true);
  p.setup_constraints(true);
  p.assemble_global();
  p.assemble_levels();
  std::cout << "operator_under_test="
            << (shift > 0.0 ? "laplace_minus_300_times_mass_indefinite"
                            : "laplace_spd")
            << " smoother=SOR_relaxation" << std::endl;

  const MGRun r = mg_solve(p, true, 0, true);
  bool grows = false;
  for (unsigned int c = 1; c < r.vcycle_residuals.size(); ++c)
    {
      std::cout << "v_cycle=" << c
                << " residual=" << r.vcycle_residuals[c]
                << " ratio_to_previous="
                << r.vcycle_residuals[c] /
                     std::max(1e-300, r.vcycle_residuals[c - 1])
                << std::endl;
      if (r.vcycle_residuals[c] > r.vcycle_residuals[c - 1])
        grows = true;
    }
  const double overall = r.vcycle_residuals.back() /
                         std::max(1e-300, r.vcycle_residuals.front());
  std::cout << "residual_after_eight_cycles_over_initial=" << overall
            << std::endl;
  std::cout << "v_cycle_residual_grows_from_cycle_to_cycle=" << yesno(grows)
            << std::endl;
  std::cout << "v_cycle_reduced_the_residual=" << yesno(overall < 1.0)
            << std::endl;
  std::cout << "cg_with_this_v_cycle converged=" << yesno(r.converged)
            << " steps=" << r.steps << std::endl;
  std::cout << "VERDICT="
            << (grows ? "spd_smoother_on_the_operator_under_test_diverges"
                      : "v_cycle_is_a_contraction_on_the_operator_under_test")
            << std::endl;
  return 0;
}

// ===========================================================================
// multigrid#3 -- the coarse-grid solver.
// The fine level is FIXED and the COARSEST level of the hierarchy is varied, so
// what moves is exactly the coarse-level dof count the entry names.
// ===========================================================================
static int mg_coarse_solver()
{
  const bool exact = mutate(); // the mistake: a relaxation sweep as the coarse solve
  MG p;
  p.make_grid(6);
  p.distribute(true);
  p.setup_constraints(true);
  p.assemble_global();
  p.assemble_levels();
  std::cout << "coarse_solver_under_test="
            << (exact ? "dense_direct_householder" : "relaxation_smoother")
            << " fine_level_dofs=" << p.dof.n_dofs() << std::endl;
  std::vector<unsigned int> test_steps, exact_steps;
  for (unsigned int min_level = 0; min_level <= 3; ++min_level)
    {
      const MGRun t = mg_solve(p, exact, min_level, false);
      const MGRun e = mg_solve(p, true, min_level, false);
      test_steps.push_back(t.steps);
      exact_steps.push_back(e.steps);
      std::cout << "coarsest_level=" << min_level
                << " coarse_level_dofs=" << p.dof.n_dofs(min_level)
                << " cg_steps_under_test=" << t.steps
                << " converged_under_test=" << yesno(t.converged)
                << " cg_steps_with_a_direct_coarse_solve=" << e.steps
                << std::endl;
    }
  const bool test_grows = test_steps.back() > test_steps.front() + 2;
  const bool exact_flat =
    exact_steps.back() <= exact_steps.front() + 2 &&
    exact_steps.front() <= exact_steps.back() + 2;
  std::cout << "iteration_count_under_test_grows_with_the_coarse_level_size="
            << yesno(test_grows) << std::endl;
  std::cout << "direct_coarse_solve_keeps_the_count_flat=" << yesno(exact_flat)
            << std::endl;
  std::cout << "VERDICT="
            << (test_grows
                  ? "coarse_solver_under_test_costs_iterations_as_the_coarse_level_grows"
                  : "coarse_solver_under_test_keeps_the_count_flat")
            << std::endl;
  return 0;
}

// ===========================================================================
// matrix_free#4 -- CG on a matrix-free Laplace needs a preconditioner.
// ===========================================================================
static int mf_cg_needs_a_preconditioner()
{
  const bool use_gmg = mutate();
  std::cout << "preconditioner_under_test="
            << (use_gmg ? "geometric_multigrid" : "none") << std::endl;
  std::vector<unsigned int> test_steps, gmg_steps;
  for (unsigned int refine = 3; refine <= 6; ++refine)
    {
      MG p;
      p.make_grid(refine);
      p.distribute(true);
      p.setup_constraints(true);
      p.assemble_global();
      p.assemble_levels();

      // The same operator as a MatrixFree cell_loop, checked against the
      // assembled one so the two iteration counts are comparable.
      MatrixFree<dim, double> mf;
      mf.reinit(p.mapping, p.dof, p.constraints, QGauss<dim>(p.fe.degree + 1));
      const std::function<void(const MatrixFree<dim, double> &,
                               Vector<double> &, const Vector<double> &,
                               const std::pair<unsigned int, unsigned int> &)>
        op = [](const MatrixFree<dim, double> &data, Vector<double> &d,
                const Vector<double> &s,
                const std::pair<unsigned int, unsigned int> &r) {
          FEEvaluation<dim, -1, 0, 1, double> phi(data);
          for (unsigned int cell = r.first; cell < r.second; ++cell)
            {
              phi.reinit(cell);
              phi.read_dof_values(s);
              phi.evaluate(EvaluationFlags::gradients);
              for (unsigned int q = 0; q < phi.n_q_points; ++q)
                phi.submit_gradient(phi.get_gradient(q), q);
              phi.integrate(EvaluationFlags::gradients);
              phi.distribute_local_to_global(d);
            }
        };
      Vector<double> src(p.dof.n_dofs()), a(p.dof.n_dofs()), b(p.dof.n_dofs());
      for (unsigned int i = 0; i < src.size(); ++i)
        src(i) = std::sin(0.3 * i + 1.0);
      p.constraints.set_zero(src);
      p.A.vmult(a, src);
      b = 0.0;
      mf.cell_loop(op, b, src);
      for (const auto i : mf.get_constrained_dofs())
        b(i) = a(i);
      Vector<double> d(b);
      d -= a;
      const double agree = d.l2_norm() / std::max(1e-300, a.l2_norm());

      SolverControl control(3000, 1e-9 * p.rhs.l2_norm());
      SolverCG<Vector<double>> cg(control);
      Vector<double>           x(p.dof.n_dofs());
      unsigned int             plain = 0;
      try
        {
          cg.solve(p.A, x, p.rhs, PreconditionIdentity());
        }
      catch (const std::exception &)
        {}
      plain = control.last_step();
      const MGRun g = mg_solve(p, true, 0, false);
      test_steps.push_back(use_gmg ? g.steps : plain);
      gmg_steps.push_back(g.steps);
      std::cout << "refine=" << refine << " n_dofs=" << p.dof.n_dofs()
                << " unpreconditioned_cg_steps=" << plain
                << " gmg_preconditioned_cg_steps=" << g.steps
                << " matrix_free_vs_assembled_relative_difference=" << agree
                << std::endl;
    }
  const bool test_grows = test_steps.back() > 2 * test_steps.front();
  const unsigned int gmax = *std::max_element(gmg_steps.begin(), gmg_steps.end());
  const unsigned int gmin = *std::min_element(gmg_steps.begin(), gmg_steps.end());
  std::cout << "iteration_count_under_test_more_than_doubles_over_three_refinements="
            << yesno(test_grows) << std::endl;
  std::cout << "gmg_iteration_count_is_flat_across_the_same_refinements="
            << yesno(gmax <= gmin + 3) << std::endl;
  std::cout << "gmg_iteration_count_stays_under_thirty=" << yesno(gmax < 30)
            << std::endl;
  std::cout << "VERDICT="
            << (test_grows
                  ? "preconditioner_under_test_does_not_hold_the_count_flat"
                  : "preconditioner_under_test_holds_the_count_flat")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  MultithreadInfo::set_thread_limit(1);
  const std::string probe = (argc > 1) ? argv[1] : "";
  const std::string sub = (argc > 2) ? argv[2] : "n_dofs";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  std::cout << std::setprecision(8);
  if (probe == "mg_no_level_dofs")
    return mg_no_level_dofs(sub);
  if (probe == "mg_boundary_indices")
    return mg_boundary_indices();
  if (probe == "mg_smoother_on_indefinite")
    return mg_smoother_on_indefinite();
  if (probe == "mg_coarse_solver")
    return mg_coarse_solver();
  if (probe == "mf_cg_needs_a_preconditioner")
    return mf_cg_needs_a_preconditioner();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
