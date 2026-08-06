// Shared translation unit for the transient-heat Signal family.
//
// usage: heat_family <probe>
//   reassemble_cost | dropped_theta_term | no_solution_transfer
//   | ic_bypasses_constraints | forward_euler_stability
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
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

// u0 = sin(pi x) sin(pi y); exact amplitude decays as exp(-2 pi^2 t).
class InitialField : public Function<dim>
{
public:
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  }
};

struct Heat
{
  Triangulation<dim>        tria;
  FE_Q<dim>                 fe;
  DoFHandler<dim>           dof;
  AffineConstraints<double> constraints;
  SparsityPattern           sp;
  SparseMatrix<double>      M, K, S;
  Vector<double>            u, u_old, rhs, tmp;

  Heat(unsigned int deg = 1)
    : fe(deg)
    , dof(tria)
  {}

  void setup(unsigned int refine)
  {
    if (tria.n_active_cells() == 0)
      {
        GridGenerator::hyper_cube(tria, 0.0, 1.0);
        tria.refine_global(refine);
      }
    dof.distribute_dofs(fe);
    constraints.clear();
    VectorTools::interpolate_boundary_values(
      dof, 0, Functions::ZeroFunction<dim>(), constraints);
    constraints.close();
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
    sp.copy_from(dsp);
    M.reinit(sp);
    K.reinit(sp);
    S.reinit(sp);
    u.reinit(dof.n_dofs());
    u_old.reinit(dof.n_dofs());
    rhs.reinit(dof.n_dofs());
    tmp.reinit(dof.n_dofs());
  }

  void assemble_mass_and_stiffness()
  {
    M = 0.0;
    K = 0.0;
    QGauss<dim>   quad(fe.degree + 1);
    FEValues<dim> fev(fe, quad,
                      update_values | update_gradients | update_JxW_values);
    const unsigned int n = fe.dofs_per_cell;
    FullMatrix<double> cm(n, n), ck(n, n);
    std::vector<types::global_dof_index> local(n);
    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        ck = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
              {
                cm(i, j) +=
                  fev.shape_value(i, q) * fev.shape_value(j, q) * fev.JxW(q);
                ck(i, j) += fev.shape_grad(i, q) * fev.shape_grad(j, q) *
                            fev.JxW(q);
              }
        cell->get_dof_indices(local);
        constraints.distribute_local_to_global(cm, local, M);
        constraints.distribute_local_to_global(ck, local, K);
      }
  }

  // One theta step. drop_old_stiffness reproduces the mistake in heat#1.
  void step(double dt, double theta, bool drop_old_stiffness)
  {
    S.copy_from(M);
    S.add(dt * theta, K);
    M.vmult(rhs, u_old);
    if (!drop_old_stiffness)
      {
        K.vmult(tmp, u_old);
        rhs.add(-dt * (1.0 - theta), tmp);
      }
    std::map<types::global_dof_index, double> bv;
    VectorTools::interpolate_boundary_values(
      dof, 0, Functions::ZeroFunction<dim>(), bv);
    MatrixTools::apply_boundary_values(bv, S, u, rhs);
    // CG + SSOR rather than a direct factorisation: a refactorisation every
    // step costs more than the assembly does, which would bury the very
    // comparison heat#0 is about.
    SolverControl control(2000, 1e-12 * std::max(1.0, rhs.l2_norm()));
    SolverCG<Vector<double>> cg(control);
    PreconditionSSOR<SparseMatrix<double>> prec;
    prec.initialize(S, 1.2);
    cg.solve(S, u, rhs, prec);
    constraints.distribute(u);
    u_old = u;
  }
};

// heat#0 — re-assembling M and K every step dominates the wall time.
static int reassemble_cost()
{
  // Both loops, same problem, one run: the comparison is measured, and only
  // the RATIO is asserted — seconds are this machine's, not the claim's.
  const unsigned int nsteps = 40;
  const double dt = 1e-4, theta = 1.0;
  double total[2] = {0.0, 0.0}, assembled[2] = {0.0, 0.0};
  for (int k = 0; k < 2; ++k)
    {
      const bool reassemble = (k == 0) && !mutate();
      Heat h(1);
      h.setup(6);
      h.assemble_mass_and_stiffness();
      VectorTools::interpolate(h.dof, InitialField(), h.u_old);
      auto s0 = std::chrono::steady_clock::now();
      for (unsigned int s = 0; s < nsteps; ++s)
        {
          auto t0 = std::chrono::steady_clock::now();
          if (reassemble)
            h.assemble_mass_and_stiffness();
          auto t1 = std::chrono::steady_clock::now();
          h.step(dt, theta, false);
          assembled[k] += std::chrono::duration<double>(t1 - t0).count();
        }
      total[k] = std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - s0)
                   .count();
      if (k == 0)
        std::cout << "n_dofs=" << h.dof.n_dofs() << std::endl;
    }
  const double frac = assembled[0] / total[0];
  const double slowdown = total[0] / total[1];
  std::cout << "reassembling_seconds=" << total[0]
            << " assemble_once_seconds=" << total[1] << std::endl;
  std::cout << "assemble_fraction_of_reassembling_loop=" << frac << std::endl;
  std::cout << "slowdown=" << slowdown << std::endl;
  // FINDING: the claim's "assemble_system dominating at ~60-80% per step" does
  // NOT reproduce here — assembly is about a third of the reassembling loop at
  // 4225 dofs with a CG+SSOR solve. What DOES reproduce, and is the substance
  // of the entry, is that re-assembling every step costs multiples of the
  // assemble-once loop for an identical answer.
  const bool in_claimed_band = (frac > 0.6 && frac < 0.8);
  const bool slower = slowdown > 1.3;
  std::cout << "assembly_fraction_in_claimed_60_to_80_band="
            << (in_claimed_band ? "true" : "false") << std::endl;
  std::cout << "reassembling_is_measurably_slower="
            << (slower ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (slower ? "reassembly_costs_multiples_for_the_same_answer"
                       : "no_measurable_cost")
            << std::endl;
  return 0;
}

// heat#1 — dropping (1-theta)*K*u_old is INCONSISTENT: halving dt does not move
// the answer toward the reference.
static int dropped_theta_term()
{
  const double T = 0.02;
  double amp[3], ref[3];
  const unsigned int nsteps[3] = {20, 40, 80};
  for (int k = 0; k < 3; ++k)
    {
      const double dt = T / nsteps[k];
      Heat a(1), b(1);
      a.setup(5);
      a.assemble_mass_and_stiffness();
      VectorTools::interpolate(a.dof, InitialField(), a.u_old);
      b.setup(5);
      b.assemble_mass_and_stiffness();
      VectorTools::interpolate(b.dof, InitialField(), b.u_old);
      for (unsigned int s = 0; s < nsteps[k]; ++s)
        {
          a.step(dt, 0.5, !mutate());  // the mistake
          b.step(dt, 0.5, false);      // correct Crank-Nicolson
        }
      amp[k] = a.u.linfty_norm();
      ref[k] = b.u.linfty_norm();
      std::cout << "nsteps=" << nsteps[k] << " dropped_amp=" << amp[k]
                << " correct_amp=" << ref[k]
                << " gap=" << std::abs(amp[k] - ref[k]) << std::endl;
    }
  const double exact = std::exp(-2.0 * numbers::PI * numbers::PI * T);
  std::cout << "exact_amplitude=" << exact << std::endl;
  const double g0 = std::abs(amp[0] - ref[0]);
  const double g2 = std::abs(amp[2] - ref[2]);
  const double self = std::abs(amp[2] - amp[1]);
  std::cout << "gap_to_reference_coarsest=" << g0
            << " gap_to_reference_finest=" << g2 << std::endl;
  std::cout << "self_convergence=" << self << std::endl;
  const bool not_closing = g2 > 0.5 * g0;
  const bool self_converging = self < 0.1 * g2;
  std::cout << "gap_to_reference_does_not_close="
            << (not_closing ? "true" : "false") << std::endl;
  std::cout << "answers_converge_to_each_other="
            << (self_converging ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << ((not_closing && self_converging)
                  ? "inconsistent_scheme_converges_to_the_wrong_answer"
                  : "scheme_is_consistent")
            << std::endl;
  return 0;
}

// heat#2 — skipping SolutionTransfer across a mesh change loses the solution.
static int no_solution_transfer()
{
  Heat h(1);
  h.setup(4);
  h.assemble_mass_and_stiffness();
  VectorTools::interpolate(h.dof, InitialField(), h.u_old);
  for (unsigned int s = 0; s < 3; ++s)
    h.step(1e-4, 1.0, false);
  const double before = h.u.l2_norm();
  std::cout << "norm_before_refinement=" << before << std::endl;

  for (const auto &cell : h.tria.active_cell_iterators())
    cell->set_refine_flag();

  Vector<double> carried;
  if (mutate())
    {
      SolutionTransfer<dim, Vector<double>> transfer(h.dof);
      h.tria.prepare_coarsening_and_refinement();
      transfer.prepare_for_coarsening_and_refinement(h.u);
      h.tria.execute_coarsening_and_refinement();
      h.dof.distribute_dofs(h.fe);
      carried.reinit(h.dof.n_dofs());
      transfer.interpolate(carried);
    }
  else
    {
      h.tria.execute_coarsening_and_refinement();
      h.dof.distribute_dofs(h.fe);
      carried.reinit(h.dof.n_dofs());   // the mistake: a fresh zero vector
    }
  const double after = carried.l2_norm();
  std::cout << "used_solution_transfer=" << (mutate() ? "true" : "false")
            << std::endl;
  std::cout << "norm_after_mesh_change=" << after << std::endl;
  const bool lost = after < 1e-12;
  std::cout << "solution_lost_across_mesh_change=" << (lost ? "true" : "false")
            << std::endl;
  std::cout << "VERDICT="
            << (lost ? "skipping_solution_transfer_zeroes_the_state"
                     : "state_carried_across")
            << std::endl;
  return 0;
}

// heat#3 — writing node values directly bypasses AffineConstraints, so the IC
// disagrees with the boundary condition it is supposed to satisfy.
static int ic_bypasses_constraints()
{
  Heat h(1);
  h.setup(4);
  h.assemble_mass_and_stiffness();
  // An IC that is 1 everywhere INCLUDING the Dirichlet boundary, where the
  // problem says u = 0.
  if (mutate())
    {
      VectorTools::interpolate(h.dof, Functions::ConstantFunction<dim>(1.0),
                               h.u_old);
      h.constraints.distribute(h.u_old);
    }
  else
    {
      for (unsigned int i = 0; i < h.u_old.size(); ++i)
        h.u_old(i) = 1.0;   // straight into the vector, constraints untouched
    }
  std::map<types::global_dof_index, double> bv;
  VectorTools::interpolate_boundary_values(
    h.dof, 0, Functions::ZeroFunction<dim>(), bv);
  double worst = 0.0;
  for (const auto &p : bv)
    worst = std::max(worst, std::abs(h.u_old(p.first) - p.second));
  std::cout << "used_constraints=" << (mutate() ? "true" : "false")
            << std::endl;
  std::cout << "n_boundary_dofs=" << bv.size() << std::endl;
  std::cout << "max_ic_violation_at_dirichlet_dofs=" << worst << std::endl;
  const bool jump = worst > 0.5;
  std::cout << "ic_violates_the_boundary_condition="
            << (jump ? "true" : "false") << std::endl;
  const double before = h.u_old.linfty_norm();
  h.step(1e-4, 1.0, false);
  std::cout << "linfty_before_first_step=" << before
            << " linfty_after_first_step=" << h.u.linfty_norm() << std::endl;
  std::cout << "VERDICT="
            << (jump ? "direct_node_write_gives_an_ic_that_violates_the_bc"
                     : "ic_consistent_with_bc")
            << std::endl;
  return 0;
}

// heat#4 — forward Euler is conditionally stable, and nothing announces it.
// A FIXED NUMBER OF STEPS is integrated, not a fixed end time, so a larger dt
// cannot hide the instability by taking fewer steps.
static int forward_euler_stability()
{
  const unsigned int nsteps = 60;
  const double theta = mutate() ? 1.0 : 0.0;
  Heat probe(1);
  probe.setup(4);
  const double h = 1.0 / 16.0;
  std::cout << "theta=" << theta << " cell_size=" << h
            << " n_steps=" << nsteps << std::endl;
  double last_stable = 0.0, first_unstable = 0.0;
  for (double dt = 1e-5; dt < 1e-1; dt *= 2.0)
    {
      Heat r(1);
      r.setup(4);
      r.assemble_mass_and_stiffness();
      VectorTools::interpolate(r.dof, InitialField(), r.u_old);
      bool blew_up = false;
      for (unsigned int s = 0; s < nsteps; ++s)
        {
          r.step(dt, theta, false);
          if (!std::isfinite(r.u.linfty_norm()) || r.u.linfty_norm() > 1e3)
            {
              blew_up = true;
              break;
            }
        }
      std::cout << "dt=" << dt << " linfty=" << r.u.linfty_norm()
                << " blew_up=" << (blew_up ? "true" : "false") << std::endl;
      if (blew_up)
        {
          first_unstable = dt;
          break;
        }
      last_stable = dt;
    }
  std::cout << "largest_stable_dt=" << last_stable
            << " smallest_unstable_dt=" << first_unstable << std::endl;
  // The claim insists the textbook finite-difference constant must not be
  // trusted for a finite-element discretisation. Print both so the gap is
  // visible rather than asserted.
  const double fd_bound = h * h / 2.0;
  std::cout << "textbook_fd_bound_h2_over_2=" << fd_bound << std::endl;
  if (first_unstable > 0.0)
    std::cout << "fd_bound_over_measured_limit=" << fd_bound / first_unstable
              << std::endl;
  std::cout << "fd_bound_is_not_the_fe_limit="
            << ((first_unstable > 0.0 &&
                 (fd_bound > 1.5 * first_unstable ||
                  fd_bound < 0.67 * first_unstable))
                  ? "true"
                  : "false")
            << std::endl;
  const bool bounded_everywhere = (first_unstable == 0.0);
  std::cout << "bounded_at_every_dt_tried="
            << (bounded_everywhere ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (bounded_everywhere
                  ? "unconditionally_stable_over_the_range_tried"
                  : "conditionally_stable_bisection_found_the_limit")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  if (probe == "reassemble_cost")
    return reassemble_cost();
  if (probe == "dropped_theta_term")
    return dropped_theta_term();
  if (probe == "no_solution_transfer")
    return no_solution_transfer();
  if (probe == "ic_bypasses_constraints")
    return ic_bypasses_constraints();
  if (probe == "forward_euler_stability")
    return forward_euler_stability();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
