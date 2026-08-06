// Shared translation unit for the goal-oriented / variational-inequality Signal
// family: an obstacle problem in the shape of step-41, and a dual-weighted
// residual loop on the L-shaped domain in the shape of step-14.
//
// usage: goal_family <probe>
//   obstacle_active_set | dwr_primal_only | dwr_same_space
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.

#include <deal.II/base/function.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_tools.h>
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
#include <set>
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
// obstacle_problem#0 -- the active-set loop.
// Membrane on (-1,1)^2, -laplace(u) = f with f = -10 pushing it down onto the
// step-41 staircase obstacle, u = 0 on the boundary and u >= psi everywhere.
// ===========================================================================
static double obstacle_psi(const Point<dim> &p)
{
  if (p[0] < -0.5)
    return -0.2;
  if (p[0] < 0.0)
    return -0.4;
  if (p[0] < 0.5)
    return -0.6;
  return -0.8;
}

static int obstacle_active_set()
{
  const bool use_multiplier = mutate(); // the mistake: gap-only active set
  std::cout << "active_set_criterion="
            << (use_multiplier ? "lambda_plus_c_times_gap" : "gap_only")
            << std::endl;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, -1.0, 1.0);
  tria.refine_global(4);
  FE_Q<dim>       fe(1);
  MappingQ1<dim>  mapping;
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);

  // The unconstrained Laplace matrix, its rhs and a lumped mass diagonal, all
  // assembled once: only the CONSTRAINTS move from iteration to iteration.
  AffineConstraints<double> bc;
  VectorTools::interpolate_boundary_values(dof, 0,
                                           Functions::ZeroFunction<dim>(), bc);
  bc.close();
  DynamicSparsityPattern dsp(dof.n_dofs());
  DoFTools::make_sparsity_pattern(dof, dsp);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> K;
  K.reinit(sp);
  Vector<double> F(dof.n_dofs()), lumped(dof.n_dofs());
  {
    QGauss<dim>   quad(2);
    FEValues<dim> fev(mapping, fe, quad,
                      update_values | update_gradients | update_JxW_values);
    const unsigned int n = fe.n_dofs_per_cell();
    FullMatrix<double> cm(n, n);
    Vector<double>     cr(n), cl(n);
    std::vector<types::global_dof_index> local(n);
    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        cr = 0.0;
        cl = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < n; ++i)
            {
              for (unsigned int j = 0; j < n; ++j)
                cm(i, j) +=
                  fev.shape_grad(i, q) * fev.shape_grad(j, q) * fev.JxW(q);
              cr(i) += -10.0 * fev.shape_value(i, q) * fev.JxW(q);
              cl(i) += fev.shape_value(i, q) * fev.JxW(q);
            }
        cell->get_dof_indices(local);
        for (unsigned int i = 0; i < n; ++i)
          {
            for (unsigned int j = 0; j < n; ++j)
              K.add(local[i], local[j], cm(i, j));
            F(local[i]) += cr(i);
            lumped(local[i]) += cl(i);
          }
      }
  }
  std::vector<Point<dim>> pts(dof.n_dofs());
  DoFTools::map_dofs_to_support_points(mapping, dof, pts);
  Vector<double> psi(dof.n_dofs());
  for (unsigned int i = 0; i < dof.n_dofs(); ++i)
    psi(i) = obstacle_psi(pts[i]);

  Vector<double>         u(dof.n_dofs()), lambda(dof.n_dofs());
  std::set<unsigned int> active;
  std::vector<std::set<unsigned int>> history;
  const unsigned int max_outer = 40;
  unsigned int       converged_at = 0;
  unsigned int       cycle_period = 0;
  double             worst_violation_at_end = 0.0;

  for (unsigned int it = 0; it < max_outer; ++it)
    {
      AffineConstraints<double> con;
      con.merge(bc);
      for (const unsigned int i : active)
        if (!con.is_constrained(i))
          {
            con.add_line(i);
            con.set_inhomogeneity(i, psi(i));
          }
      con.close();

      SparseMatrix<double> A;
      A.reinit(sp);
      A.copy_from(K);
      Vector<double> b(F);
      for (unsigned int i = 0; i < dof.n_dofs(); ++i)
        if (con.is_constrained(i))
          b(i) = con.get_inhomogeneity(i);
      for (unsigned int r = 0; r < dof.n_dofs(); ++r)
        {
          if (con.is_constrained(r))
            {
              for (auto itr = A.begin(r); itr != A.end(r); ++itr)
                itr->value() = (itr->column() == r) ? 1.0 : 0.0;
              continue;
            }
          for (auto itr = A.begin(r); itr != A.end(r); ++itr)
            if (con.is_constrained(itr->column()))
              {
                b(r) -= itr->value() * con.get_inhomogeneity(itr->column());
                itr->value() = 0.0;
              }
        }
      SparseDirectUMFPACK inv;
      inv.initialize(A);
      u = b;
      inv.solve(u);

      // The multiplier: lambda = K u - F, which is zero away from contact.
      K.vmult(lambda, u);
      lambda -= F;

      std::set<unsigned int> next;
      const double           c = 100.0;
      for (unsigned int i = 0; i < dof.n_dofs(); ++i)
        {
          if (bc.is_constrained(i))
            continue;
          const double gap = psi(i) - u(i); // > 0 means the obstacle is violated
          const bool   take = use_multiplier
                                ? (lambda(i) + c * lumped(i) * gap > 0.0)
                                : (gap > 0.0);
          if (take)
            next.insert(i);
        }
      double worst = 0.0;
      for (unsigned int i = 0; i < dof.n_dofs(); ++i)
        worst = std::max(worst, psi(i) - u(i));
      worst_violation_at_end = worst;
      std::cout << "outer=" << it << " active_set_size=" << active.size()
                << " next_active_set_size=" << next.size()
                << " min_u=" << *std::min_element(u.begin(), u.end())
                << " worst_obstacle_violation=" << worst << std::endl;

      if (next == active && it > 0)
        {
          converged_at = it;
          break;
        }
      for (unsigned int k = 0; k < history.size(); ++k)
        if (history[k] == next)
          {
            cycle_period = static_cast<unsigned int>(history.size() - k);
            break;
          }
      history.push_back(next);
      active = next;
      if (cycle_period > 0)
        break;
    }

  std::cout << "outer_iterations_run=" << history.size() << std::endl;
  std::cout << "worst_obstacle_violation_at_the_end=" << worst_violation_at_end
            << std::endl;
  std::cout << "two_consecutive_active_sets_became_identical="
            << yesno(converged_at > 0) << std::endl;
  std::cout << "active_set_repeats_an_earlier_state=" << yesno(cycle_period > 0)
            << std::endl;
  std::cout << "cycle_period=" << cycle_period << std::endl;
  std::cout << "active_set_cycles_with_period_two=" << yesno(cycle_period == 2)
            << std::endl;
  std::cout << "converged_within_ten_outer_iterations="
            << yesno(converged_at > 0 && converged_at <= 10) << std::endl;
  std::cout << "VERDICT="
            << (converged_at > 0 ? "active_set_loop_reached_two_identical_sets"
                                 : "active_set_loop_never_settled")
            << std::endl;
  return 0;
}

// ===========================================================================
// error_estimation#0 and #1 -- the dual-weighted residual on the L-shape.
// -laplace(u) = 1 on the L-shaped domain, u = 0 on the boundary, so the
// reentrant corner at the origin carries an r^(2/3) singularity. The goal
// functional is the mean value of u over the square [-0.75,-0.25] x [0.25,0.75],
// which is nowhere near that corner.
// ===========================================================================
static bool in_goal_region(const Point<dim> &p)
{
  return p[0] > -0.75 && p[0] < -0.25 && p[1] > 0.25 && p[1] < 0.75;
}
static const double goal_area = 0.25;

// The dual right-hand side density: J(v) = (1/|D|) int_D v.
class GoalDensity : public Function<dim>
{
public:
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return in_goal_region(p) ? 1.0 / goal_area : 0.0;
  }
};

struct Poisson
{
  Triangulation<dim>       &tria;
  FE_Q<dim>                 fe;
  MappingQ1<dim>            mapping;
  DoFHandler<dim>           dof;
  AffineConstraints<double> constraints;
  SparsityPattern           sp;
  SparseMatrix<double>      A;
  Vector<double>            sol, rhs;

  Poisson(Triangulation<dim> &t, unsigned int degree)
    : tria(t)
    , fe(degree)
    , dof(t)
  {}

  void setup()
  {
    dof.distribute_dofs(fe);
    constraints.clear();
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

  // source == nullptr means f = 1 (the primal); otherwise the given density.
  void assemble_and_solve(const Function<dim> *source)
  {
    A = 0.0;
    rhs = 0.0;
    QGauss<dim>   quad(fe.degree + 2);
    FEValues<dim> fev(mapping, fe, quad,
                      update_values | update_gradients |
                        update_quadrature_points | update_JxW_values);
    const unsigned int n = fe.n_dofs_per_cell();
    FullMatrix<double> cm(n, n);
    Vector<double>     cr(n);
    std::vector<types::global_dof_index> local(n);
    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        cr = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          {
            const double f =
              source ? source->value(fev.quadrature_point(q)) : 1.0;
            for (unsigned int i = 0; i < n; ++i)
              {
                for (unsigned int j = 0; j < n; ++j)
                  cm(i, j) +=
                    fev.shape_grad(i, q) * fev.shape_grad(j, q) * fev.JxW(q);
                cr(i) += f * fev.shape_value(i, q) * fev.JxW(q);
              }
          }
        cell->get_dof_indices(local);
        constraints.distribute_local_to_global(cm, cr, local, A, rhs);
      }
    SparseDirectUMFPACK inv;
    inv.initialize(A);
    sol = rhs;
    inv.solve(sol);
    constraints.distribute(sol);
  }

  double goal_value() const
  {
    QGauss<dim>         quad(fe.degree + 3);
    FEValues<dim>       fev(mapping, fe, quad,
                            update_values | update_quadrature_points |
                              update_JxW_values);
    double              integral = 0.0;
    std::vector<double> vals(quad.size());
    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        fev.get_function_values(sol, vals);
        for (unsigned int q = 0; q < quad.size(); ++q)
          if (in_goal_region(fev.quadrature_point(q)))
            integral += vals[q] * fev.JxW(q);
      }
    return integral / goal_area;
  }
};

// The DWR indicator, cell by cell:
//   eta_K = int_K f w - int_K grad(u_h).grad(w),  w = z - I_h z
// which sums to the error representation for this problem and needs no face
// terms, because a Q1 function has zero Laplacian on a Cartesian cell.
static void dwr_indicators(const Poisson &primal, const Poisson &dual,
                           Vector<double> &eta, double &sum)
{
  Vector<double> Ihz(primal.dof.n_dofs());
  FETools::interpolate(dual.dof, dual.sol, primal.dof, primal.constraints, Ihz);
  eta.reinit(primal.tria.n_active_cells());
  sum = 0.0;
  QGauss<dim>   quad(dual.fe.degree + 3);
  FEValues<dim> fp(primal.mapping, primal.fe, quad,
                   update_values | update_gradients |
                     update_quadrature_points | update_JxW_values);
  FEValues<dim> fd(dual.mapping, dual.fe, quad,
                   update_values | update_gradients);
  std::vector<double>         zv(quad.size()), iv(quad.size());
  std::vector<Tensor<1, dim>> zg(quad.size()), ig(quad.size()), ug(quad.size());
  auto         pc = primal.dof.begin_active();
  auto         dc = dual.dof.begin_active();
  unsigned int c = 0;
  for (; pc != primal.dof.end(); ++pc, ++dc, ++c)
    {
      fp.reinit(pc);
      fd.reinit(dc);
      fd.get_function_values(dual.sol, zv);
      fd.get_function_gradients(dual.sol, zg);
      fp.get_function_values(Ihz, iv);
      fp.get_function_gradients(Ihz, ig);
      fp.get_function_gradients(primal.sol, ug);
      double e = 0.0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        {
          const double         w = zv[q] - iv[q];
          const Tensor<1, dim> gw = zg[q] - ig[q];
          e += (1.0 * w - ug[q] * gw) * fp.JxW(q);
        }
      eta(c) = e;
      sum += e;
    }
}

struct Cycle
{
  unsigned int n_dofs = 0;
  double       goal = 0.0;
  double       estimator = 0.0;
  double       kelly = 0.0;
  double       corner_fraction = 0.0;
  double       goal_fraction = 0.0;
};

// One adaptive loop. criterion 0 = Kelly (primal residual only), 1 = DWR.
static std::vector<Cycle> adaptive_loop(int criterion, unsigned int cycles,
                                        unsigned int dual_degree,
                                        double      &last_estimator)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_L(tria, -1.0, 1.0);
  tria.refine_global(3);
  std::vector<Cycle> out;
  for (unsigned int cy = 0; cy < cycles; ++cy)
    {
      Poisson primal(tria, 1), dual(tria, dual_degree);
      primal.setup();
      primal.assemble_and_solve(nullptr);
      dual.setup();
      const GoalDensity j;
      dual.assemble_and_solve(&j);

      Vector<double> eta;
      double         sum = 0.0;
      dwr_indicators(primal, dual, eta, sum);
      Vector<float> kelly(tria.n_active_cells());
      KellyErrorEstimator<dim>::estimate(
        primal.mapping, primal.dof, QGauss<dim - 1>(3),
        std::map<types::boundary_id, const Function<dim> *>(), primal.sol,
        kelly);

      Cycle rec;
      rec.n_dofs = primal.dof.n_dofs();
      rec.goal = primal.goal_value();
      rec.estimator = sum;
      rec.kelly = kelly.l2_norm();
      unsigned int near_corner = 0, in_goal = 0, total = 0;
      for (const auto &cell : tria.active_cell_iterators())
        {
          ++total;
          if (cell->center().norm() < 0.2)
            ++near_corner;
          if (in_goal_region(cell->center()))
            ++in_goal;
        }
      rec.corner_fraction = double(near_corner) / total;
      rec.goal_fraction = double(in_goal) / total;
      out.push_back(rec);
      last_estimator = sum;

      if (cy + 1 == cycles)
        break;
      Vector<float> crit(tria.n_active_cells());
      if (criterion == 0)
        crit = kelly;
      else
        for (unsigned int i = 0; i < eta.size(); ++i)
          crit(i) = static_cast<float>(std::abs(eta(i)));
      GridRefinement::refine_and_coarsen_fixed_number(tria, crit, 0.3, 0.0);
      tria.execute_coarsening_and_refinement();
    }
  return out;
}

// A uniformly refined Q2 reference value of the functional.
static double reference_goal(unsigned int refine)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_L(tria, -1.0, 1.0);
  tria.refine_global(refine);
  Poisson p(tria, 2);
  p.setup();
  p.assemble_and_solve(nullptr);
  return p.goal_value();
}

static int dwr_primal_only()
{
  const int under_test = mutate() ? 1 : 0; // 0 = Kelly only, 1 = DWR
  std::cout << "refinement_criterion_under_test="
            << (under_test == 0 ? "kelly_primal_residual_only"
                                : "dual_weighted_residual")
            << std::endl;
  const double j5 = reference_goal(5), j6 = reference_goal(6);
  std::cout << "reference_goal_refine5=" << j5
            << " reference_goal_refine6=" << j6
            << " reference_uncertainty=" << std::abs(j6 - j5) << std::endl;
  const double jref = j6;

  double                   dummy = 0.0;
  const std::vector<Cycle> t = adaptive_loop(under_test, 5, 2, dummy);
  const std::vector<Cycle> d = adaptive_loop(1, 5, 2, dummy);
  const std::vector<Cycle> k = adaptive_loop(0, 5, 2, dummy);
  for (unsigned int i = 0; i < t.size(); ++i)
    std::cout << "cycle=" << i << " n_dofs_under_test=" << t[i].n_dofs
              << " goal_error_under_test=" << std::abs(jref - t[i].goal)
              << " corner_cell_fraction_under_test=" << t[i].corner_fraction
              << " goal_region_cell_fraction_under_test=" << t[i].goal_fraction
              << std::endl;
  for (unsigned int i = 0; i < d.size(); ++i)
    std::cout << "cycle=" << i << " n_dofs_dwr=" << d[i].n_dofs
              << " goal_error_dwr=" << std::abs(jref - d[i].goal)
              << " dwr_estimator=" << d[i].estimator
              << " corner_cell_fraction_dwr=" << d[i].corner_fraction
              << " goal_region_cell_fraction_dwr=" << d[i].goal_fraction
              << std::endl;
  for (unsigned int i = 0; i < k.size(); ++i)
    std::cout << "cycle=" << i << " n_dofs_kelly=" << k[i].n_dofs
              << " goal_error_kelly=" << std::abs(jref - k[i].goal)
              << " kelly_total=" << k[i].kelly
              << " corner_cell_fraction_kelly=" << k[i].corner_fraction
              << " goal_region_cell_fraction_kelly=" << k[i].goal_fraction
              << std::endl;

  const double err_test = std::abs(jref - t.back().goal);
  const double err_dwr = std::abs(jref - d.back().goal);
  const double eff_kelly =
    k.back().kelly / std::max(1e-300, std::abs(jref - k.back().goal));
  const double eff_dwr = std::abs(d.back().estimator) /
                         std::max(1e-300, std::abs(jref - d.back().goal));
  // The estimator the criterion under test actually has in hand.
  const double eff_test = (under_test == 0) ? eff_kelly : eff_dwr;
  const bool   eff_test_ok = (eff_test > 0.1 && eff_test < 10.0);
  std::cout << "effectivity_of_the_kelly_total_against_the_goal_error="
            << eff_kelly << std::endl;
  std::cout << "effectivity_of_the_dwr_estimator_against_the_goal_error="
            << eff_dwr << std::endl;
  std::cout << "effectivity_under_test=" << eff_test << std::endl;
  std::cout << "effectivity_under_test_is_within_a_factor_ten_of_one="
            << yesno(eff_test_ok) << std::endl;
  std::cout << "primal_only_effectivity_is_within_a_factor_ten_of_one="
            << yesno(eff_kelly > 0.1 && eff_kelly < 10.0) << std::endl;
  std::cout << "dual_weighted_effectivity_is_within_a_factor_ten_of_one="
            << yesno(eff_dwr > 0.1 && eff_dwr < 10.0) << std::endl;
  std::cout << "criterion_under_test_puts_more_than_fifteen_percent_of_cells_in_the_goal_region="
            << yesno(t.back().goal_fraction > 0.15) << std::endl;
  std::cout << "kelly_puts_more_cells_at_the_reentrant_corner_than_dwr="
            << yesno(k.back().corner_fraction > d.back().corner_fraction)
            << std::endl;
  std::cout << "dwr_puts_more_cells_in_the_goal_region_than_kelly="
            << yesno(d.back().goal_fraction > k.back().goal_fraction)
            << std::endl;
  std::cout << "goal_error_under_test_is_worse_than_the_dual_weighted_one="
            << yesno(err_test > 1.5 * err_dwr) << std::endl;
  std::cout << "VERDICT="
            << (eff_test_ok
                  ? "estimator_under_test_tracks_the_goal_error"
                  : "estimator_under_test_does_not_estimate_the_goal_error")
            << std::endl;
  return 0;
}

static int dwr_same_space()
{
  const unsigned int dual_degree = mutate() ? 2 : 1; // 1 = same space as primal
  std::cout << "primal_degree=1 dual_degree=" << dual_degree << std::endl;
  const double             jref = reference_goal(6);
  double                   est = 0.0;
  const std::vector<Cycle> r = adaptive_loop(1, 3, dual_degree, est);
  for (unsigned int i = 0; i < r.size(); ++i)
    std::cout << "cycle=" << i << " n_dofs=" << r[i].n_dofs
              << " goal_error=" << std::abs(jref - r[i].goal)
              << " estimator=" << r[i].estimator << std::endl;
  const double goal_err = std::abs(jref - r.back().goal);
  const double eff = std::abs(est) / std::max(1e-300, goal_err);
  std::cout << "final_estimator=" << est << " final_goal_error=" << goal_err
            << " effectivity_index=" << eff << std::endl;
  const bool zero_est = std::abs(est) < 1e-12;
  std::cout << "estimator_under_test_is_zero_to_machine_precision="
            << yesno(zero_est) << std::endl;
  std::cout << "effectivity_index_under_test_is_zero=" << yesno(eff < 1e-9)
            << std::endl;
  std::cout << "effectivity_index_under_test_is_one="
            << yesno(eff > 0.5 && eff < 2.0) << std::endl;
  std::cout << "VERDICT="
            << (zero_est ? "same_space_dual_gives_an_identically_zero_estimator"
                         : "estimator_under_test_carries_information")
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
  if (probe == "obstacle_active_set")
    return obstacle_active_set();
  if (probe == "dwr_primal_only")
    return dwr_primal_only();
  if (probe == "dwr_same_space")
    return dwr_same_space();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
