// Shared translation unit for the hp-adaptive Signal family.
//
// An hp Poisson solver on the unit square, hp::FECollection of FE_Q(1..4) with
// a matched hp::QCollection, exactly the shape the catalog template has. Each
// probe below bends one part of it.
//
// SEVERAL OF THESE ASSERTS LIVE IN THE COMPILED LIBRARY
// (source/dofs/dof_handler.cc), so -DDEBUG on this file cannot revive them —
// only a deal.II built with CMAKE_BUILD_TYPE=Debug can. Assert ABORTS
// (SIGABRT, rc=134); it does not throw, so the exit code is the observable and
// a try/catch would see nothing. The fixtures that care run this program
// against BOTH libraries and pin the pair.
//
// usage: hp_family <probe>
//   refine_after_final_solve | fe_index_past_collection | single_quadrature_rule
//   | no_smoothness_estimator | p_interface_without_constraints
//   | matrixfree_hp_support | raw_copy_across_p_change
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_series.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/refinement.h>
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
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/smoothness_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

// Manufactured solution u = sin(pi x) sin(pi y), zero on the boundary of the
// unit square, with f = 2 pi^2 u. Smooth, so p-refinement is the right move.
class ExactSolution : public Function<dim>
{
public:
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  }
};
class SourceTerm : public Function<dim>
{
public:
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return 2.0 * numbers::PI * numbers::PI * std::sin(numbers::PI * p[0]) *
           std::sin(numbers::PI * p[1]);
  }
};
// A coefficient that varies inside a cell, so a coarse rule under-integrates.
static double coefficient(const Point<dim> &p)
{
  return 2.0 + std::sin(8.0 * numbers::PI * p[0]) *
                 std::sin(8.0 * numbers::PI * p[1]);
}

struct SolveInfo
{
  bool         converged        = false;
  unsigned int steps            = 0;
  double       initial_residual = 0.0;
  double       last_residual    = 0.0;
  double       energy           = 0.0;
};

struct HP
{
  Triangulation<dim>        tria;
  hp::FECollection<dim>     fes;
  hp::QCollection<dim>      qs;
  hp::QCollection<dim - 1>  qfs;
  DoFHandler<dim>           dof;
  AffineConstraints<double> constraints;
  SparsityPattern           sp;
  SparseMatrix<double>      A;
  Vector<double>            sol, rhs;
  unsigned int              max_degree;

  HP(unsigned int maxdeg = 4)
    : dof(tria)
    , max_degree(maxdeg)
  {
    for (unsigned int d = 1; d <= max_degree; ++d)
      {
        fes.push_back(FE_Q<dim>(d));
        qs.push_back(QGauss<dim>(d + 1));
        qfs.push_back(QGauss<dim - 1>(d + 1));
      }
  }

  void grid(unsigned int refine)
  {
    GridGenerator::hyper_cube(tria, 0.0, 1.0);
    tria.refine_global(refine);
  }

  // with_interface_constraints=false is the hp#4 mistake: only the Dirichlet
  // rows are constrained, the p-transition faces are left free.
  void setup(bool with_interface_constraints = true)
  {
    dof.distribute_dofs(fes);
    constraints.clear();
    if (with_interface_constraints)
      DoFTools::make_hanging_node_constraints(dof, constraints);
    VectorTools::interpolate_boundary_values(
      dof, 0, Functions::ZeroFunction<dim>(), constraints);
    constraints.close();
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
    sp.copy_from(dsp);
    A.reinit(sp);
    sol.reinit(dof.n_dofs());
    rhs.reinit(dof.n_dofs());
  }

  unsigned int n_interface_constraints()
  {
    AffineConstraints<double> c;
    DoFTools::make_hanging_node_constraints(dof, c);
    c.close();
    return c.n_constraints();
  }

  // rules of size 1 are BROADCAST to every element, which is what hp::FEValues
  // does internally and is the usual shape of the hp#2 bug.
  void assemble(const hp::QCollection<dim> &rules, bool variable_coeff)
  {
    A   = 0.0;
    rhs = 0.0;
    SourceTerm source;
    for (const auto &cell : dof.active_cell_iterators())
      {
        const unsigned int q_index =
          (rules.size() == 1) ? 0 : cell->active_fe_index();
        const unsigned int n = cell->get_fe().n_dofs_per_cell();
        FEValues<dim> fev(cell->get_fe(), rules[q_index],
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
        fev.reinit(cell);
        FullMatrix<double> cm(n, n);
        Vector<double>     cr(n);
        std::vector<types::global_dof_index> local(n);
        for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
          {
            const double a =
              variable_coeff ? coefficient(fev.quadrature_point(q)) : 1.0;
            const double f = source.value(fev.quadrature_point(q));
            for (unsigned int i = 0; i < n; ++i)
              {
                for (unsigned int j = 0; j < n; ++j)
                  cm(i, j) += a * fev.shape_grad(i, q) * fev.shape_grad(j, q) *
                              fev.JxW(q);
                cr(i) += f * fev.shape_value(i, q) * fev.JxW(q);
              }
          }
        cell->get_dof_indices(local);
        constraints.distribute_local_to_global(cm, cr, local, A, rhs);
      }
  }

  double max_asymmetry() const
  {
    double worst = 0.0;
    for (unsigned int r = 0; r < A.m(); ++r)
      for (auto it = A.begin(r); it != A.end(r); ++it)
        worst = std::max(worst, std::abs(it->value() - A.el(it->column(), r)));
    return worst;
  }

  SolveInfo solve(unsigned int max_it = 5000)
  {
    SolveInfo info;
    info.initial_residual = rhs.l2_norm();
    SolverControl control(max_it, 1e-10 * std::max(1.0, rhs.l2_norm()));
    SolverCG<Vector<double>>               cg(control);
    PreconditionSSOR<SparseMatrix<double>> prec;
    prec.initialize(A, 1.0);
    try
      {
        cg.solve(A, sol, rhs, prec);
        info.converged     = true;
        info.steps         = control.last_step();
        info.last_residual = control.last_value();
      }
    catch (const SolverControl::NoConvergence &e)
      {
        info.converged     = false;
        info.steps         = e.last_step;
        info.last_residual = e.last_residual;
      }
    constraints.distribute(sol);
    Vector<double> Au(sol.size());
    A.vmult(Au, sol);
    info.energy = 0.5 * (sol * Au) - (sol * rhs);
    return info;
  }

  double l2_error()
  {
    Vector<float> cellwise(tria.n_active_cells());
    VectorTools::integrate_difference(dof, sol, ExactSolution(), cellwise, qs,
                                      VectorTools::L2_norm);
    return VectorTools::compute_global_error(tria, cellwise,
                                             VectorTools::L2_norm);
  }

  // A functional that does not depend on the assembly rule: the integral of the
  // computed solution, always evaluated with the MATCHED collection.
  double integral_of_solution()
  {
    double total = 0.0;
    for (const auto &cell : dof.active_cell_iterators())
      {
        FEValues<dim> fev(cell->get_fe(), qs[cell->active_fe_index()],
                          update_values | update_JxW_values);
        fev.reinit(cell);
        std::vector<double> vals(fev.n_quadrature_points);
        fev.get_function_values(sol, vals);
        for (unsigned int q = 0; q < fev.n_quadrature_points; ++q)
          total += vals[q] * fev.JxW(q);
      }
    return total;
  }

  unsigned int highest_degree_in_use() const
  {
    unsigned int best = 0;
    for (const auto &cell : dof.active_cell_iterators())
      best = std::max(best, cell->get_fe().degree);
    return best;
  }
};

// ----------------------------------------------------------------- hp#0
// execute_coarsening_and_refinement() AFTER the final solve, with the closing
// DataOut still reading (dof_handler, solution).
static int refine_after_final_solve()
{
  const unsigned int n_cycles = 3;
  HP                 p;
  p.grid(2);
  Vector<double> solution;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      p.setup(true);
      p.assemble(p.qs, false);
      p.solve();
      solution = p.sol;
      std::cout << "Cycle " << cycle << ": " << p.dof.n_dofs() << " DOFs, "
                << p.tria.n_active_cells() << " cells" << std::endl;

      // The fix every tutorial uses: stop before marking on the last cycle.
      if (mutate() && cycle == n_cycles - 1)
        break;

      Vector<float> estimate(p.tria.n_active_cells());
      KellyErrorEstimator<dim>::estimate(p.dof, p.qfs, {}, p.sol, estimate);
      GridRefinement::refine_and_coarsen_fixed_number(p.tria, estimate, 0.3,
                                                     0.03);
      p.tria.execute_coarsening_and_refinement();
    }

  std::cout << "refined_after_the_last_solve=" << (mutate() ? "false" : "true")
            << std::endl;
  std::cout << "solution_vector_size=" << solution.size()
            << " dof_handler_n_dofs=" << p.dof.n_dofs()
            << " triangulation_cells=" << p.tria.n_active_cells() << std::endl;
  std::cout << "before_data_out" << std::endl;
  std::cout.flush();

  DataOut<dim> data_out;
  data_out.attach_dof_handler(p.dof);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  const std::filesystem::path out =
    std::filesystem::temp_directory_path() /
    ("t2_hp_final_" + std::to_string(::getpid()) + ".vtu");
  {
    std::ofstream o(out);
    data_out.write_vtu(o);
  }
  std::filesystem::remove(out);
  std::cout << "after_data_out" << std::endl;
  std::cout << "VERDICT=closing_data_out_completed" << std::endl;
  return 0;
}

// ----------------------------------------------------------------- hp#1
// An active_fe_index past the end of the collection, in the entry's own shape:
// a 2-entry collection with one cell set to index 5.
static int fe_index_past_collection()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0);
  tria.refine_global(1);
  hp::FECollection<dim> fes;
  fes.push_back(FE_Q<dim>(1));
  fes.push_back(FE_Q<dim>(2));
  DoFHandler<dim> dof(tria);
  const unsigned int idx = mutate() ? 1u : 5u;
  std::cout << "fe_collection_size=" << fes.size() << std::endl;
  std::cout << "n_active_cells=" << tria.n_active_cells() << std::endl;
  std::cout << "active_fe_index_set_on_one_cell=" << idx << std::endl;
  std::cout << "index_is_inside_the_collection="
            << ((idx < fes.size()) ? "true" : "false") << std::endl;
  bool first = true;
  for (const auto &cell : dof.active_cell_iterators())
    {
      cell->set_active_fe_index(first ? idx : 0u);
      first = false;
    }
  std::cout << "before_distribute_dofs" << std::endl;
  std::cout.flush();
  dof.distribute_dofs(fes);
  std::cout << "after_distribute_dofs" << std::endl;
  std::cout << "n_dofs=" << dof.n_dofs() << std::endl;
  std::cout << "VERDICT=distribute_dofs_returned" << std::endl;
  return 0;
}

// ----------------------------------------------------------------- hp#2
// One low-order rule broadcast over a collection of FE_Q(1..4).
static void assign_cyclic_degrees(HP &p)
{
  unsigned int i = 0;
  for (const auto &cell : p.dof.active_cell_iterators())
    cell->set_active_fe_index(i++ % p.fes.size());
}

static int single_quadrature_rule()
{
  auto run = [](const std::string &label, unsigned int single_rule,
                SolveInfo &info, double &integral, double &asym,
                unsigned int &qsize) {
    HP p;
    p.grid(2);
    p.dof.distribute_dofs(p.fes);
    assign_cyclic_degrees(p);
    p.setup(true);
    hp::QCollection<dim> rules;
    if (single_rule > 0)
      rules.push_back(QGauss<dim>(single_rule));
    else
      rules = p.qs;
    qsize = rules.size();
    p.assemble(rules, true);
    asym     = p.max_asymmetry();
    info     = p.solve();
    integral = p.integral_of_solution();
    std::cout << label << "_q_collection_size=" << qsize
              << " fe_collection_size=" << p.fes.size()
              << " n_dofs=" << p.dof.n_dofs() << std::endl;
    std::cout << label << "_cg_converged=" << (info.converged ? "true"
                                                              : "false")
              << " cg_steps=" << info.steps
              << " initial_residual=" << info.initial_residual
              << " final_residual=" << info.last_residual << std::endl;
    std::cout << label << "_max_matrix_asymmetry=" << asym
              << " integral_of_solution=" << integral << std::endl;
  };

  SolveInfo    i_matched, i_moderate, i_test;
  double       s_matched, s_moderate, s_test;
  double       a_matched, a_moderate, a_test;
  unsigned int q_matched, q_moderate, q_test;
  run("matched", 0, i_matched, s_matched, a_matched, q_matched);
  run("moderate_single_qgauss4", 4, i_moderate, s_moderate, a_moderate,
      q_moderate);
  // The run under test: one QGauss(2) for the whole collection.
  run(mutate() ? "run_under_test_matched" : "run_under_test_single_qgauss2",
      mutate() ? 0 : 2, i_test, s_test, a_test, q_test);

  std::cout << "run_under_test_rule="
            << (mutate() ? "matched_collection" : "single_qgauss2")
            << std::endl;
  std::cout << "run_under_test_q_collection_matches_fe_collection="
            << ((q_test == 4) ? "true" : "false") << std::endl;
  const bool noconv = !i_test.converged;
  const bool grew   = i_test.last_residual > i_test.initial_residual;
  const double shift =
    std::abs(s_moderate - s_matched) / std::max(1e-30, std::abs(s_matched));
  std::cout << "moderate_rule_relative_functional_shift=" << shift << std::endl;
  std::cout << "moderate_rule_converged="
            << (i_moderate.converged ? "true" : "false") << std::endl;
  std::cout << "moderate_rule_shifts_the_functional="
            << ((shift > 0.001) ? "true" : "false") << std::endl;
  std::cout << "max_asymmetry_matched=" << a_matched
            << " max_asymmetry_coarse=" << a_test << std::endl;
  std::cout << "coarse_rule_broke_matrix_symmetry="
            << ((a_test > 1e-10) ? "true" : "false") << std::endl;
  std::cout << "run_under_test_threw_noconvergence=" << (noconv ? "true"
                                                                : "false")
            << std::endl;
  std::cout << "run_under_test_residual_grew_above_the_initial_residual="
            << (grew ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (noconv
                  ? "a_single_coarse_rule_leaves_the_hp_operator_unsolvable"
                  : "solver_converged")
            << std::endl;
  return 0;
}

// ----------------------------------------------------------------- hp#3
// With no FESeries/smoothness object there is no p-adaptivity at all: the run
// stays h-only. (There is no p_adaptivity_from_smoothness() in this library to
// "fall back" — see the symbol grep in the fixture's cmd.sh.)
struct CycleHistory
{
  std::vector<unsigned int> dofs;
  std::vector<double>       error;
  unsigned int              top_degree = 1;
};

static CycleHistory run_cycles(bool use_smoothness, unsigned int n_cycles)
{
  CycleHistory h;
  HP           p;
  p.grid(2);
  FESeries::Legendre<dim> legendre =
    SmoothnessEstimator::Legendre::default_fe_series(p.fes);
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
    {
      p.setup(true);
      p.assemble(p.qs, false);
      p.solve();
      h.dofs.push_back(p.dof.n_dofs());
      h.error.push_back(p.l2_error());
      h.top_degree = std::max(h.top_degree, p.highest_degree_in_use());
      if (cycle == n_cycles - 1)
        break;
      Vector<float> estimate(p.tria.n_active_cells());
      KellyErrorEstimator<dim>::estimate(p.dof, p.qfs, {}, p.sol, estimate);
      GridRefinement::refine_and_coarsen_fixed_number(p.tria, estimate, 0.3,
                                                      0.0);
      if (use_smoothness)
        {
          Vector<float> smoothness(p.tria.n_active_cells());
          SmoothnessEstimator::Legendre::coefficient_decay(legendre, p.dof,
                                                           p.sol, smoothness);
          hp::Refinement::p_adaptivity_from_relative_threshold(p.dof,
                                                               smoothness, 0.5,
                                                               0.0);
          hp::Refinement::choose_p_over_h(p.dof);
          hp::Refinement::limit_p_level_difference(p.dof);
        }
      p.tria.execute_coarsening_and_refinement();
    }
  return h;
}

static int no_smoothness_estimator()
{
  const unsigned int n_cycles = 8;
  const bool         under_test_uses_smoothness = mutate();
  CycleHistory u = run_cycles(under_test_uses_smoothness, n_cycles);
  CycleHistory o = run_cycles(!under_test_uses_smoothness, n_cycles);
  for (unsigned int c = 0; c < u.dofs.size(); ++c)
    std::cout << "  under_test_cycle=" << c << " n_dofs=" << u.dofs[c]
              << " l2_error=" << u.error[c] << std::endl;
  for (unsigned int c = 0; c < o.dofs.size(); ++c)
    std::cout << "  other_run_cycle=" << c << " n_dofs=" << o.dofs[c]
              << " l2_error=" << o.error[c] << std::endl;
  std::cout << "run_under_test_used_a_smoothness_estimator="
            << (under_test_uses_smoothness ? "true" : "false") << std::endl;
  std::cout << "run_under_test_highest_degree=" << u.top_degree
            << " other_run_highest_degree=" << o.top_degree << std::endl;
  const double slope_u =
    std::log(u.error.back() / u.error.front()) /
    std::log(static_cast<double>(u.dofs.back()) / u.dofs.front());
  const double slope_o =
    std::log(o.error.back() / o.error.front()) /
    std::log(static_cast<double>(o.dofs.back()) / o.dofs.front());
  std::cout << "run_under_test_error_vs_dofs_slope=" << slope_u
            << " other_run_error_vs_dofs_slope=" << slope_o << std::endl;
  std::cout << "run_under_test_final_dofs=" << u.dofs.back()
            << " final_l2_error=" << u.error.back() << std::endl;
  std::cout << "other_run_final_dofs=" << o.dofs.back()
            << " final_l2_error=" << o.error.back() << std::endl;
  const bool stayed_p1 = (u.top_degree == 1);
  const bool worse     = u.error.back() > 10.0 * o.error.back();
  std::cout << "run_under_test_kept_every_cell_at_the_lowest_degree="
            << (stayed_p1 ? "true" : "false") << std::endl;
  std::cout << "run_under_test_is_at_least_ten_times_less_accurate="
            << (worse ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (stayed_p1 ? "without_a_smoothness_estimator_the_run_is_h_only"
                          : "p_adaptivity_raised_the_polynomial_degree")
            << std::endl;
  return 0;
}

// ----------------------------------------------------------------- hp#4
// A p-transition with no interface constraints. The mesh is uniform, so every
// constraint that make_hanging_node_constraints produces here IS the
// p-projection the entry is about.
static int p_interface_without_constraints()
{
  HP p;
  p.grid(2);
  p.dof.distribute_dofs(p.fes);
  for (const auto &cell : p.dof.active_cell_iterators())
    cell->set_active_fe_index(cell->center()[0] < 0.5 ? 0u : 2u);   // p1 | p3
  p.setup(mutate());
  std::cout << "interface_constraints_applied=" << (mutate() ? "true" : "false")
            << std::endl;
  std::cout << "n_dofs=" << p.dof.n_dofs()
            << " constraints_available_from_make_hanging_node_constraints="
            << p.n_interface_constraints() << std::endl;
  p.assemble(p.qs, false);
  p.solve();

  // Walk every interior face and compare the two traces at matched physical
  // points. Faces between equal-degree cells are the control.
  QGauss<dim - 1> fq(5);
  double worst_p_jump = 0.0, worst_same_jump = 0.0;
  unsigned int n_p_faces = 0, n_same_faces = 0;
  for (const auto &cell : p.dof.active_cell_iterators())
    for (const auto f : cell->face_indices())
      {
        if (cell->at_boundary(f))
          continue;
        const auto neigh = cell->neighbor(f);
        if (!neigh->is_active() || neigh < cell)
          continue;
        const unsigned int nf = cell->neighbor_of_neighbor(f);
        FEFaceValues<dim>  here(cell->get_fe(), fq,
                                update_values | update_quadrature_points);
        FEFaceValues<dim>  there(neigh->get_fe(), fq,
                                 update_values | update_quadrature_points);
        here.reinit(cell, f);
        there.reinit(neigh, nf);
        std::vector<double> vh(fq.size()), vt(fq.size());
        here.get_function_values(p.sol, vh);
        there.get_function_values(p.sol, vt);
        double worst = 0.0;
        for (unsigned int q = 0; q < fq.size(); ++q)
          {
            unsigned int best = 0;
            double       bd   = 1e300;
            for (unsigned int r = 0; r < fq.size(); ++r)
              {
                const double d = here.quadrature_point(q).distance(
                  there.quadrature_point(r));
                if (d < bd)
                  {
                    bd   = d;
                    best = r;
                  }
              }
            worst = std::max(worst, std::abs(vh[q] - vt[best]));
          }
        if (cell->active_fe_index() != neigh->active_fe_index())
          {
            ++n_p_faces;
            worst_p_jump = std::max(worst_p_jump, worst);
          }
        else
          {
            ++n_same_faces;
            worst_same_jump = std::max(worst_same_jump, worst);
          }
      }
  const double scale = std::max(1e-30, p.sol.linfty_norm());
  std::cout << "p_transition_faces=" << n_p_faces
            << " equal_degree_interior_faces=" << n_same_faces << std::endl;
  std::cout << "solution_linfty=" << scale << std::endl;
  std::cout << "max_jump_across_p_transition_faces=" << worst_p_jump
            << " relative=" << worst_p_jump / scale << std::endl;
  std::cout << "max_jump_across_equal_degree_faces=" << worst_same_jump
            << std::endl;
  const bool found      = n_p_faces > 0;
  const bool discont    = worst_p_jump / scale > 1e-3;
  const bool control_ok = worst_same_jump / scale < 1e-9;
  std::cout << "found_p_transition_faces=" << (found ? "true" : "false")
            << std::endl;
  std::cout << "solution_jumps_across_p_transitions="
            << (discont ? "true" : "false") << std::endl;
  std::cout << "equal_degree_faces_stayed_continuous="
            << (control_ok ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (discont ? "unconstrained_p_transition_leaves_a_solution_jump"
                        : "solution_is_continuous_across_p_transitions")
            << std::endl;
  return 0;
}

// ----------------------------------------------------------------- hp#5
// MatrixFree on a DoFHandler with a MIXED set of active_fe_indices. This entry
// is a POSITIVE claim, so the mutation is a negative control: it removes the
// mixed-p property (the entry's own Signal — indices that never reached the
// DoFHandler) and the read-back drops to one.
struct MFInfo
{
  bool         reinit_returned     = false;
  unsigned int n_active_fe_indices = 0;
  unsigned int n_cell_batches      = 0;
  unsigned int distinct_batch_fe_indices = 0;
  unsigned int cells_at_degree_two = 0;
  unsigned int n_dofs              = 0;
};

static MFInfo run_matrixfree(const std::string &label, bool set_indices)
{
  MFInfo             info;
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0);
  tria.refine_global(2);
  hp::FECollection<dim> fes;
  fes.push_back(FE_Q<dim>(1));
  fes.push_back(FE_Q<dim>(2));
  DoFHandler<dim> dof(tria);
  if (set_indices)
    {
      unsigned int i = 0;
      for (const auto &cell : dof.active_cell_iterators())
        cell->set_active_fe_index((i++) % 2);
    }
  dof.distribute_dofs(fes);
  for (const auto &cell : dof.active_cell_iterators())
    if (cell->active_fe_index() == 1)
      ++info.cells_at_degree_two;
  info.n_dofs = dof.n_dofs();

  AffineConstraints<double> constraints;
  DoFTools::make_hanging_node_constraints(dof, constraints);
  VectorTools::interpolate_boundary_values(
    dof, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  hp::QCollection<1> q1;
  q1.push_back(QGauss<1>(2));
  q1.push_back(QGauss<1>(3));

  MatrixFree<dim, double>                          mf;
  typename MatrixFree<dim, double>::AdditionalData ad;
  ad.tasks_parallel_scheme = MatrixFree<dim, double>::AdditionalData::none;
  ad.mapping_update_flags  = update_gradients | update_JxW_values;
  std::cout << label << "_before_matrixfree_reinit" << std::endl;
  std::cout.flush();
  mf.reinit(MappingQ1<dim>(), dof, constraints, q1, ad);
  info.reinit_returned = true;
  std::cout << label << "_after_matrixfree_reinit" << std::endl;
  info.n_active_fe_indices = mf.n_active_fe_indices();
  info.n_cell_batches      = mf.n_cell_batches();
  std::vector<unsigned int> seen;
  for (unsigned int b = 0; b < mf.n_cell_batches(); ++b)
    {
      const unsigned int idx = mf.get_cell_active_fe_index({b, b + 1});
      if (std::find(seen.begin(), seen.end(), idx) == seen.end())
        seen.push_back(idx);
    }
  info.distinct_batch_fe_indices = seen.size();
  std::cout << label << "_cells_at_degree_two=" << info.cells_at_degree_two
            << " n_dofs=" << info.n_dofs << std::endl;
  std::cout << label
            << "_matrixfree_n_active_fe_indices=" << info.n_active_fe_indices
            << " n_cell_batches=" << info.n_cell_batches
            << " distinct_batch_fe_indices=" << info.distinct_batch_fe_indices
            << std::endl;
  return info;
}

static int matrixfree_hp_support()
{
  // Both configurations always run: a DoFHandler whose cells alternate between
  // FE_Q(1) and FE_Q(2), and one where set_active_fe_index was never called.
  const MFInfo mixed   = run_matrixfree("mixed_p", true);
  const MFInfo uniform = run_matrixfree("uniform_p", false);
  const MFInfo &t      = mutate() ? uniform : mixed;

  std::cout << "run_under_test=" << (mutate() ? "uniform_p" : "mixed_p")
            << std::endl;
  std::cout << "under_test_mesh_is_mixed_p="
            << ((t.cells_at_degree_two > 0) ? "true" : "false") << std::endl;
  std::cout << "under_test_matrixfree_reinit_returned="
            << (t.reinit_returned ? "true" : "false") << std::endl;
  std::cout << "under_test_cell_batch_list_is_populated="
            << ((t.n_cell_batches > 0) ? "true" : "false") << std::endl;
  std::cout << "under_test_cell_batches_carry_two_fe_indices="
            << ((t.distinct_batch_fe_indices == 2) ? "true" : "false")
            << std::endl;
  // The entry's Signal — "read back n_active_fe_indices(); if it is 1 on a
  // mesh you believe is mixed-p, the indices never reached the DoFHandler" —
  // is checked against both configurations rather than assumed.
  std::cout << "n_active_fe_indices_mixed_p=" << mixed.n_active_fe_indices
            << " n_active_fe_indices_uniform_p=" << uniform.n_active_fe_indices
            << std::endl;
  std::cout << "n_active_fe_indices_cannot_tell_mixed_p_from_uniform_p="
            << ((mixed.n_active_fe_indices == uniform.n_active_fe_indices)
                  ? "true"
                  : "false")
            << std::endl;
  std::cout << "get_cell_active_fe_index_can_tell_them_apart="
            << ((mixed.distinct_batch_fe_indices !=
                 uniform.distinct_batch_fe_indices)
                  ? "true"
                  : "false")
            << std::endl;
  std::cout << "VERDICT="
            << ((t.reinit_returned && t.distinct_batch_fe_indices == 2)
                  ? "matrixfree_accepted_a_mixed_p_dofhandler"
                  : "matrixfree_saw_a_single_fe_index")
            << std::endl;
  return 0;
}

// ----------------------------------------------------------------- hp#6
// Copying dof values across a p-change by index instead of transferring them.
static int raw_copy_across_p_change()
{
  HP p;
  p.grid(3);
  p.dof.distribute_dofs(p.fes);
  for (const auto &cell : p.dof.active_cell_iterators())
    cell->set_active_fe_index(2);   // FE_Q(3) everywhere
  p.dof.distribute_dofs(p.fes);

  // A field with content the p=3 space resolves and a p=1 space would not.
  class Wiggly : public Function<dim>
  {
  public:
    double value(const Point<dim> &q, const unsigned int = 0) const override
    {
      return std::sin(3.0 * numbers::PI * q[0]) *
             std::sin(3.0 * numbers::PI * q[1]);
    }
  };
  Vector<double> u(p.dof.n_dofs());
  VectorTools::interpolate(p.dof, Wiggly(), u);
  const double before_linfty = u.linfty_norm();
  const unsigned int dofs_before = p.dof.n_dofs();

  auto error_against_field = [&](const Vector<double> &vec) {
    Vector<float> cellwise(p.tria.n_active_cells());
    VectorTools::integrate_difference(p.dof, vec, Wiggly(), cellwise, p.qs,
                                      VectorTools::L2_norm);
    return VectorTools::compute_global_error(p.tria, cellwise,
                                             VectorTools::L2_norm);
  };
  const double error_before = error_against_field(u);

  // p-refine every cell from FE_Q(3) to FE_Q(4).
  for (const auto &cell : p.dof.active_cell_iterators())
    cell->set_future_fe_index(3);

  Vector<double> carried;
  if (mutate())
    {
      SolutionTransfer<dim, Vector<double>> transfer(p.dof);
      p.tria.prepare_coarsening_and_refinement();
      transfer.prepare_for_coarsening_and_refinement(u);
      p.tria.execute_coarsening_and_refinement();
      p.dof.distribute_dofs(p.fes);
      carried.reinit(p.dof.n_dofs());
      transfer.interpolate(carried);
    }
  else
    {
      p.tria.execute_coarsening_and_refinement();
      p.dof.distribute_dofs(p.fes);
      carried.reinit(p.dof.n_dofs());
      // The mistake: dof values written straight into the new vector.
      for (unsigned int i = 0; i < std::min(carried.size(), u.size()); ++i)
        carried(i) = u(i);
    }
  const double after_linfty = carried.linfty_norm();
  const double error_after  = error_against_field(carried);

  std::cout << "used_solution_transfer=" << (mutate() ? "true" : "false")
            << std::endl;
  std::cout << "degree_before=3 degree_after=" << p.highest_degree_in_use()
            << std::endl;
  std::cout << "n_dofs_before=" << dofs_before
            << " n_dofs_after=" << p.dof.n_dofs() << std::endl;
  std::cout << "linfty_before=" << before_linfty
            << " linfty_after=" << after_linfty << std::endl;
  std::cout << "l2_error_before=" << error_before
            << " l2_error_after=" << error_after << std::endl;
  const double drop = (before_linfty - after_linfty) / before_linfty;
  std::cout << "linfty_relative_drop=" << drop << std::endl;
  std::cout << "linfty_drop_in_the_claimed_10_to_50_percent_band="
            << ((drop >= 0.10 && drop <= 0.50) ? "true" : "false") << std::endl;
  const bool wrecked = error_after > 100.0 * error_before;
  std::cout << "transferred_field_lost_the_original_content="
            << (wrecked ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (wrecked ? "writing_dof_values_across_a_p_change_destroys_them"
                        : "field_survived_the_p_change")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
#ifdef DEBUG
  std::cout << "consumer_DEBUG=1" << std::endl;
#else
  std::cout << "consumer_DEBUG=0" << std::endl;
#endif
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  if (probe == "refine_after_final_solve")
    return refine_after_final_solve();
  if (probe == "fe_index_past_collection")
    return fe_index_past_collection();
  if (probe == "single_quadrature_rule")
    return single_quadrature_rule();
  if (probe == "no_smoothness_estimator")
    return no_smoothness_estimator();
  if (probe == "p_interface_without_constraints")
    return p_interface_without_constraints();
  if (probe == "matrixfree_hp_support")
    return matrixfree_hp_support();
  if (probe == "raw_copy_across_p_change")
    return raw_copy_across_p_change();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
