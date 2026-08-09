// Shared translation unit for the wave-equation Signal family.
//
// d^2u/dt^2 = c^2 laplacian(u) on the unit square, integrated with Newmark in
// its acceleration form so that the SAME code covers implicit (beta=0.25),
// explicit (beta=0) and a boundary damping term (absorbing BC):
//
//   u~ = u_n + dt v_n + (0.5-beta) dt^2 a_n
//   v~ = v_n + (1-gamma) dt a_n
//   (M + gamma dt C + beta dt^2 c^2 K) a_{n+1} = f - C v~ - c^2 K u~
//   u_{n+1} = u~ + beta dt^2 a_{n+1},  v_{n+1} = v~ + gamma dt a_{n+1}
//
// The effective matrix is assembled ONCE and the boundary values are applied to
// it once, exactly as the catalog template does; only the right-hand side moves.
//
// usage: wave_family <probe>
//   newmark_beta_stability | reassemble_each_step | explicit_newmark_cfl
//   | zero_initial_acceleration | reflecting_boundary | vtu_every_step_cost
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
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

// The catalog template's own initial displacement: a Gaussian pulse.
class InitialDisplacement : public Function<dim>
{
public:
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    const double r2 = (p - Point<dim>(0.5, 0.5)).norm_square();
    return std::exp(-100.0 * r2);
  }
};

struct Wave
{
  Triangulation<dim>   tria;
  FE_Q<dim>            fe;
  DoFHandler<dim>      dof;
  SparsityPattern      sp;
  SparseMatrix<double> M, K, C, S;
  Vector<double>       u, v, a, up, vp, rhs, tmp, anew;
  std::map<types::global_dof_index, double> bvals;

  double c2        = 1.0;
  bool   dirichlet = true;   // false -> natural (reflecting) boundary
  double damping   = 0.0;    // coefficient of the boundary damping term
  double h         = 0.0;
  // When set, the step uses a one-shot factorisation of the effective matrix
  // instead of CG+SSOR — the pattern the catalog recommends for a constant dt
  // and mesh, and the cheapest per-step solve available here.
  SparseDirectUMFPACK *factorisation = nullptr;

  Wave(unsigned int deg = 1)
    : fe(deg)
    , dof(tria)
  {}

  void setup(unsigned int refine, bool dirichlet_bc = true,
             double absorb = 0.0)
  {
    dirichlet = dirichlet_bc;
    damping   = absorb;
    GridGenerator::hyper_cube(tria, 0.0, 1.0);
    tria.refine_global(refine);
    h = 1.0 / std::pow(2.0, static_cast<double>(refine));
    dof.distribute_dofs(fe);
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_sparsity_pattern(dof, dsp);
    sp.copy_from(dsp);
    M.reinit(sp);
    K.reinit(sp);
    C.reinit(sp);
    S.reinit(sp);
    const unsigned int n = dof.n_dofs();
    u.reinit(n);
    v.reinit(n);
    a.reinit(n);
    up.reinit(n);
    vp.reinit(n);
    rhs.reinit(n);
    tmp.reinit(n);
    anew.reinit(n);
    if (dirichlet)
      VectorTools::interpolate_boundary_values(
        dof, 0, Functions::ZeroFunction<dim>(), bvals);
  }

  // Mass, stiffness, and (when absorbing) the boundary mass matrix that
  // carries the c*du/dt damping term.
  void assemble()
  {
    M = 0.0;
    K = 0.0;
    C = 0.0;
    QGauss<dim>     quad(fe.degree + 1);
    QGauss<dim - 1> fquad(fe.degree + 1);
    FEValues<dim>   fev(fe, quad,
                        update_values | update_gradients | update_JxW_values);
    FEFaceValues<dim> ffv(fe, fquad, update_values | update_JxW_values);
    const unsigned int n = fe.dofs_per_cell;
    FullMatrix<double> cm(n, n), ck(n, n), cc(n, n);
    std::vector<types::global_dof_index> local(n);
    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        ck = 0.0;
        cc = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
              {
                cm(i, j) +=
                  fev.shape_value(i, q) * fev.shape_value(j, q) * fev.JxW(q);
                ck(i, j) +=
                  fev.shape_grad(i, q) * fev.shape_grad(j, q) * fev.JxW(q);
              }
        if (damping != 0.0)
          for (const auto f : cell->face_indices())
            if (cell->face(f)->at_boundary())
              {
                ffv.reinit(cell, f);
                for (unsigned int q = 0; q < fquad.size(); ++q)
                  for (unsigned int i = 0; i < n; ++i)
                    for (unsigned int j = 0; j < n; ++j)
                      cc(i, j) += damping * ffv.shape_value(i, q) *
                                  ffv.shape_value(j, q) * ffv.JxW(q);
              }
        cell->get_dof_indices(local);
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            {
              M.add(local[i], local[j], cm(i, j));
              K.add(local[i], local[j], ck(i, j));
              if (damping != 0.0)
                C.add(local[i], local[j], cc(i, j));
            }
      }
  }

  static void solve_spd(const SparseMatrix<double> &A, Vector<double> &x,
                        const Vector<double> &b, unsigned int &iterations)
  {
    SolverControl control(3000, 1e-10 * std::max(1.0, b.l2_norm()));
    SolverCG<Vector<double>>               cg(control);
    PreconditionSSOR<SparseMatrix<double>> prec;
    prec.initialize(A, 1.2);
    cg.solve(A, x, b, prec);
    iterations = control.last_step();
  }

  // u0 = Gaussian, v0 = 0, a0 from M a0 = -c^2 K u0 (or zeroed: wave#3).
  void set_initial_state(bool solve_a0)
  {
    VectorTools::interpolate(dof, InitialDisplacement(), u);
    if (dirichlet)
      for (const auto &p : bvals)
        u(p.first) = p.second;
    v = 0.0;
    a = 0.0;
    if (!solve_a0)
      return;
    K.vmult(rhs, u);
    rhs *= -c2;
    SparseMatrix<double> Mbc(sp);
    Mbc.copy_from(M);
    if (dirichlet)
      {
        std::map<types::global_dof_index, double> zero = bvals;
        MatrixTools::apply_boundary_values(zero, Mbc, a, rhs);
      }
    unsigned int it = 0;
    solve_spd(Mbc, a, rhs, it);
  }

  // Effective matrix, assembled once and boundary-corrected once.
  void prepare(double dt, double beta, double gamma)
  {
    S.copy_from(M);
    if (damping != 0.0)
      S.add(gamma * dt, C);
    S.add(beta * dt * dt * c2, K);
    if (dirichlet)
      {
        Vector<double> dummy_x(dof.n_dofs()), dummy_b(dof.n_dofs());
        std::map<types::global_dof_index, double> zero = bvals;
        MatrixTools::apply_boundary_values(zero, S, dummy_x, dummy_b);
      }
  }

  // One Newmark step. Returns the CG iteration count.
  unsigned int step(double dt, double beta, double gamma)
  {
    up = u;
    up.add(dt, v);
    up.add((0.5 - beta) * dt * dt, a);
    vp = v;
    vp.add((1.0 - gamma) * dt, a);

    K.vmult(rhs, up);
    rhs *= -c2;
    if (damping != 0.0)
      {
        C.vmult(tmp, vp);
        rhs -= tmp;
      }
    if (dirichlet)
      for (const auto &p : bvals)
        rhs(p.first) = 0.0;

    anew = 0.0;
    unsigned int it = 0;
    if (factorisation != nullptr)
      factorisation->vmult(anew, rhs);
    else
      solve_spd(S, anew, rhs, it);

    u = up;
    u.add(beta * dt * dt, anew);
    v = vp;
    v.add(gamma * dt, anew);
    a = anew;
    return it;
  }

  double energy()
  {
    M.vmult(tmp, v);
    const double kin = 0.5 * (v * tmp);
    K.vmult(tmp, u);
    const double pot = 0.5 * c2 * (u * tmp);
    return kin + pot;
  }
};

static bool finite_and_bounded(const Vector<double> &x, double cap = 1e6)
{
  const double n = x.linfty_norm();
  return std::isfinite(n) && n < cap;
}

// Run a fixed number of steps, stop early on blow-up. Returns the step at
// which the amplitude left the bounded range (0 if it never did).
struct RunResult
{
  unsigned int blew_up_at = 0;
  double       final_linfty = 0.0;
  double       max_linfty   = 0.0;
  double       growth_per_step = 1.0;
};

static RunResult integrate(Wave &w, double dt, double beta, double gamma,
                           unsigned int nsteps, bool trace = false)
{
  RunResult r;
  double    first_amp = std::max(1e-30, w.u.linfty_norm());
  unsigned int done   = 0;
  for (unsigned int s = 0; s < nsteps; ++s)
    {
      try
        {
          w.step(dt, beta, gamma);
        }
      catch (const std::exception &)
        {
          r.blew_up_at = s + 1;
          break;
        }
      done = s + 1;
      const double amp = w.u.linfty_norm();
      r.max_linfty = std::max(r.max_linfty, std::isfinite(amp) ? amp : 1e300);
      if (trace && ((s + 1) % 5 == 0 || s == 0))
        std::cout << "  step=" << s + 1 << " linfty=" << amp << std::endl;
      if (!finite_and_bounded(w.u))
        {
          r.blew_up_at = s + 1;
          break;
        }
    }
  r.final_linfty = w.u.linfty_norm();
  if (done > 0 && std::isfinite(r.final_linfty) && r.final_linfty > 0.0)
    r.growth_per_step =
      std::pow(r.final_linfty / first_amp, 1.0 / static_cast<double>(done));
  return r;
}

// ---------------------------------------------------------------- wave#0
// Newmark with beta < gamma/2 is only CONDITIONALLY stable; the canonical
// (0.25, 0.5) pair is not. Both runs use the SAME dt, so beta is the only
// thing that differs.
static int newmark_beta_stability()
{
  const unsigned int refine = 4;
  const double       gamma  = 0.5;
  const double       beta   = mutate() ? 0.25 : 0.10;   // 0.10 < gamma/2
  const double       dt     = 0.05;
  const unsigned int nsteps = 60;

  Wave w;
  w.setup(refine, true);
  w.assemble();
  w.set_initial_state(true);
  w.prepare(dt, beta, gamma);
  std::cout << "n_dofs=" << w.dof.n_dofs() << " cell_size=" << w.h
            << " dt=" << dt << " gamma=" << gamma << " beta=" << beta
            << std::endl;
  std::cout << "beta_below_gamma_over_two="
            << ((beta < 0.5 * gamma) ? "true" : "false") << std::endl;
  const double amp0 = w.u.linfty_norm();
  RunResult    r    = integrate(w, dt, beta, gamma, nsteps, true);
  std::cout << "initial_linfty=" << amp0 << " final_linfty=" << r.final_linfty
            << std::endl;
  std::cout << "blew_up_at_step=" << r.blew_up_at << " of " << nsteps
            << std::endl;
  std::cout << "growth_factor_per_step=" << r.growth_per_step << std::endl;

  // The same problem with the canonical pair, always run, as the contrast.
  Wave ref;
  ref.setup(refine, true);
  ref.assemble();
  ref.set_initial_state(true);
  ref.prepare(dt, 0.25, gamma);
  RunResult rr = integrate(ref, dt, 0.25, gamma, nsteps);
  std::cout << "contrast_beta_0.25_final_linfty=" << rr.final_linfty
            << " contrast_blew_up_at_step=" << rr.blew_up_at << std::endl;

  const bool unstable = (r.blew_up_at > 0);
  const bool ref_bounded = (rr.blew_up_at == 0) && (rr.final_linfty < 2.0);
  // The claim's own numbers: doubling every 5-10 steps, NaN within ~50. How
  // fast it doubles is a function of how far dt sits above the stability
  // limit, so the rate is scanned over several dt rather than quoted from one.
  const double steps_to_double =
    (r.growth_per_step > 1.0) ? std::log(2.0) / std::log(r.growth_per_step)
                              : 0.0;
  std::cout << "steps_per_doubling=" << steps_to_double << std::endl;
  bool any_in_band = false;
  for (const double dts :
       {0.020, 0.025, 0.030, 0.033, 0.0335, 0.0337, 0.034, 0.036, 0.040,
        0.050})
    {
      Wave s;
      s.setup(refine, true);
      s.assemble();
      s.set_initial_state(true);
      s.prepare(dts, 0.10, gamma);
      RunResult    rs = integrate(s, dts, 0.10, gamma, 400);
      const double sd = (rs.growth_per_step > 1.0)
                          ? std::log(2.0) / std::log(rs.growth_per_step)
                          : 0.0;
      std::cout << "  scan_dt=" << dts << " blew_up_at_step=" << rs.blew_up_at
                << " steps_per_doubling=" << sd << std::endl;
      if (rs.blew_up_at > 0 && sd >= 5.0 && sd <= 10.0)
        any_in_band = true;
    }
  std::cout << "doubling_in_claimed_5_to_10_step_band_at_run_dt="
            << ((steps_to_double >= 5.0 && steps_to_double <= 10.0) ? "true"
                                                                   : "false")
            << std::endl;
  std::cout << "doubling_in_claimed_5_to_10_step_band_at_some_dt="
            << (any_in_band ? "true" : "false") << std::endl;
  std::cout << "left_bounded_range_within_50_steps="
            << ((r.blew_up_at > 0 && r.blew_up_at <= 50) ? "true" : "false")
            << std::endl;
  std::cout << "run_amplitude_diverged=" << (unstable ? "true" : "false")
            << std::endl;
  std::cout << "canonical_pair_stayed_bounded="
            << (ref_bounded ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (unstable ? "beta_below_gamma_over_two_diverges_at_this_dt"
                         : "amplitude_stayed_bounded")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------- wave#1
// Re-assembling M and K inside the time loop.
static int reassemble_each_step()
{
  const unsigned int nsteps = 60;
  const double       dt = 0.002, beta = 0.25, gamma = 0.5;
  double total[2] = {0, 0}, asm_time[2] = {0, 0}, solve_time[2] = {0, 0};
  unsigned int cg_iters[2] = {0, 0};
  unsigned int ndofs = 0;

  // Warm-up, untimed: the first loop of a process pays page faults and a cold
  // cache, and that must not be charged to whichever variant happens to run
  // first. The reference loop is then timed BEFORE the loop under test, so the
  // ordering works against the result being reported.
  {
    Wave warm;
    warm.setup(6, true);
    warm.assemble();
    warm.set_initial_state(true);
    warm.prepare(dt, beta, gamma);
    for (unsigned int s = 0; s < 5; ++s)
      warm.step(dt, beta, gamma);
  }

  // Each loop is timed TWICE and the shorter run kept: interference from other
  // processes only ever adds time, so the minimum is the honest measurement and
  // it tightens both sides of the comparison.
  for (int k = 1; k >= 0; --k)
    {
      const bool reassemble = (k == 0) && !mutate();
      total[k]              = 1e300;
      for (int rep = 0; rep < 2; ++rep)
        {
          Wave w;
          w.setup(6, true);
          w.assemble();
          w.set_initial_state(true);
          w.prepare(dt, beta, gamma);
          ndofs = w.dof.n_dofs();
          TimerOutput timer(std::cout, TimerOutput::never,
                            TimerOutput::wall_times);
          unsigned int iters   = 0;
          auto         t_start = std::chrono::steady_clock::now();
          for (unsigned int s = 0; s < nsteps; ++s)
            {
              {
                TimerOutput::Scope sc(timer, "assemble_system");
                if (reassemble)
                  {
                    w.assemble();
                    w.prepare(dt, beta, gamma);
                  }
              }
              {
                TimerOutput::Scope sc(timer, "solve");
                iters += w.step(dt, beta, gamma);
              }
            }
          const double elapsed = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t_start)
                                   .count();
          if (elapsed < total[k])
            {
              total[k] = elapsed;
              const auto data = timer.get_summary_data(
                TimerOutput::OutputData::total_wall_time);
              asm_time[k]   = data.at("assemble_system");
              solve_time[k] = data.at("solve");
              cg_iters[k]   = iters;
              if (k == 0)
                {
                  std::cout
                    << "=== TimerOutput::print_summary of the loop under test"
                    << std::endl;
                  timer.print_summary();
                }
            }
        }
    }

  const double frac     = asm_time[0] / total[0];
  const double slowdown = total[0] / total[1];
  std::cout << "n_dofs=" << ndofs << " n_steps=" << nsteps << std::endl;
  std::cout << "loop_under_test_seconds=" << total[0]
            << " assemble_once_loop_seconds=" << total[1] << std::endl;
  std::cout << "assemble_section_seconds=" << asm_time[0]
            << " solve_section_seconds=" << solve_time[0] << std::endl;
  std::cout << "assemble_fraction_of_wall_time=" << frac << std::endl;
  std::cout << "slowdown_vs_assemble_once=" << slowdown << std::endl;
  std::cout << "cg_iterations_loop_under_test=" << cg_iters[0]
            << " cg_iterations_assemble_once=" << cg_iters[1] << std::endl;

  // The claim also asserts the cost scales as O(ndof^2). Measure the exponent
  // instead of believing it: assemble the same system on three meshes.
  double t_asm[3], n_asm[3];
  for (int r = 0; r < 3; ++r)
    {
      Wave w;
      w.setup(5 + r, true);
      n_asm[r] = w.dof.n_dofs();
      auto t0  = std::chrono::steady_clock::now();
      for (int rep = 0; rep < 5; ++rep)
        w.assemble();
      t_asm[r] = std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0)
                   .count() /
                 5.0;
      std::cout << "  assembly_scaling_point ndofs=" << n_asm[r]
                << " seconds=" << t_asm[r] << std::endl;
    }
  const double exponent =
    std::log(t_asm[2] / t_asm[0]) / std::log(n_asm[2] / n_asm[0]);
  std::cout << "assembly_ndofs_small=" << n_asm[0]
            << " assembly_ndofs_large=" << n_asm[2] << std::endl;
  std::cout << "assembly_seconds_small=" << t_asm[0]
            << " assembly_seconds_large=" << t_asm[2] << std::endl;
  std::cout << "measured_assembly_scaling_exponent=" << exponent << std::endl;
  std::cout << "assembly_scales_quadratically_in_ndofs="
            << ((exponent > 1.6) ? "true" : "false") << std::endl;
  std::cout << "assembly_fraction_in_claimed_60_to_80_band="
            << ((frac > 0.6 && frac < 0.8) ? "true" : "false") << std::endl;
  const bool slower = slowdown > 1.5;
  const bool iters_unchanged =
    (cg_iters[0] == cg_iters[1]);
  std::cout << "reassembling_is_measurably_slower=" << (slower ? "true"
                                                               : "false")
            << std::endl;
  std::cout << "cg_iteration_count_is_unchanged="
            << (iters_unchanged ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (slower
                  ? "reassembling_every_step_costs_multiples_for_the_same_answer"
                  : "no_measurable_cost")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------- wave#2
// Explicit Newmark (beta=0) is CFL-bound. The claim's rule of thumb is
// dt < h/c; the measured limit for this consistent-mass discretisation is
// printed next to it rather than assumed.
static int explicit_newmark_cfl()
{
  const unsigned int refine = 4;
  const double gamma = 0.5, beta = 0.0;
  const unsigned int nsteps = 60;
  Wave probe;
  probe.setup(refine, true);
  const double h_over_c = probe.h / std::sqrt(probe.c2);
  std::cout << "cell_size=" << probe.h << " wave_speed=1"
            << " claimed_cfl_bound_h_over_c=" << h_over_c << std::endl;

  // Bisection-free scan for the true limit at beta=0.
  double last_stable = 0.0, first_unstable = 0.0;
  for (double dt = h_over_c / 64.0; dt < 8.0 * h_over_c; dt *= 1.15)
    {
      Wave w;
      w.setup(refine, true);
      w.assemble();
      w.set_initial_state(true);
      w.prepare(dt, beta, gamma);
      RunResult r = integrate(w, dt, beta, gamma, nsteps);
      if (r.blew_up_at > 0)
        {
          first_unstable = dt;
          break;
        }
      last_stable = dt;
    }
  std::cout << "measured_largest_stable_dt=" << last_stable
            << " measured_smallest_unstable_dt=" << first_unstable
            << std::endl;
  if (first_unstable > 0.0)
    std::cout << "claimed_bound_over_measured_limit="
              << h_over_c / first_unstable << std::endl;
  std::cout << "dt_equal_to_h_over_c_is_actually_safe="
            << ((first_unstable > 0.0 && h_over_c < first_unstable) ? "true"
                                                                    : "false")
            << std::endl;

  // One explicit run at a dt that SATISFIES the claim's rule of thumb, to say
  // plainly whether dt < h/c is sufficient for this discretisation.
  {
    const double dt_below = 0.8 * h_over_c;
    Wave         b;
    b.setup(refine, true);
    b.assemble();
    b.set_initial_state(true);
    b.prepare(dt_below, beta, gamma);
    RunResult rb = integrate(b, dt_below, beta, gamma, nsteps);
    std::cout << "dt_below_claimed_bound=" << dt_below
              << " blew_up_at_step=" << rb.blew_up_at
              << " final_linfty=" << rb.final_linfty << std::endl;
    std::cout << "a_dt_that_satisfies_dt_less_than_h_over_c_still_diverged="
              << ((rb.blew_up_at > 0) ? "true" : "false") << std::endl;
  }

  // The run under test: dt four times the claimed bound, explicit.
  const double dt = mutate() ? h_over_c / 8.0 : 4.0 * h_over_c;
  Wave w;
  w.setup(refine, true);
  w.assemble();
  w.set_initial_state(true);
  w.prepare(dt, beta, gamma);
  const double e0 = w.energy();
  std::cout << "run_dt=" << dt << " run_dt_over_h_over_c=" << dt / h_over_c
            << " run_beta=" << beta << std::endl;
  std::cout << "initial_energy=" << e0 << std::endl;
  RunResult r = integrate(w, dt, beta, gamma, nsteps, true);
  const double e1 = w.energy();
  std::cout << "final_energy=" << e1
            << " energy_ratio=" << (e1 / std::max(1e-300, e0)) << std::endl;
  std::cout << "blew_up_at_step=" << r.blew_up_at << " of " << nsteps
            << std::endl;
  std::cout << "growth_factor_per_step=" << r.growth_per_step << std::endl;
  std::cout << "energy_growth_per_step_factor_above_ten="
            << ((r.blew_up_at > 0 &&
                 std::pow(e1 / std::max(1e-300, e0),
                          1.0 / std::max(1u, r.blew_up_at)) > 10.0)
                  ? "true"
                  : "false")
            << std::endl;

  // Same too-large dt, implicit: the claim's stated remedy.
  Wave imp;
  imp.setup(refine, true);
  imp.assemble();
  imp.set_initial_state(true);
  const double dt_big = 4.0 * h_over_c;
  imp.prepare(dt_big, 0.25, gamma);
  RunResult ri = integrate(imp, dt_big, 0.25, gamma, nsteps);
  std::cout << "contrast_implicit_same_dt_final_linfty=" << ri.final_linfty
            << " contrast_implicit_blew_up_at_step=" << ri.blew_up_at
            << std::endl;

  const bool unstable = (r.blew_up_at > 0);
  const bool implicit_bounded = (ri.blew_up_at == 0 && ri.final_linfty < 2.0);
  std::cout << "explicit_run_diverged=" << (unstable ? "true" : "false")
            << std::endl;
  std::cout << "implicit_at_the_same_dt_stayed_bounded="
            << (implicit_bounded ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (unstable ? "explicit_newmark_above_the_cfl_limit_diverges"
                         : "explicit_run_stayed_bounded")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------- wave#3
// a_0 = 0 instead of solving M a_0 = -c^2 K u_0. The reference is the SAME
// program with the same dt and the same mesh, so a_0 is the only difference.
static int zero_initial_acceleration()
{
  const unsigned int refine = 5, nsteps = 60;
  const double dt = 0.005, beta = 0.25, gamma = 0.5;
  Wave run, ref;
  run.setup(refine, true);
  run.assemble();
  run.set_initial_state(mutate());   // mutate -> solve for a_0
  run.prepare(dt, beta, gamma);
  ref.setup(refine, true);
  ref.assemble();
  ref.set_initial_state(true);
  ref.prepare(dt, beta, gamma);

  std::cout << "n_dofs=" << run.dof.n_dofs() << " dt=" << dt << std::endl;
  std::cout << "initial_acceleration_solved=" << (mutate() ? "true" : "false")
            << std::endl;
  std::cout << "reference_initial_acceleration_linfty=" << ref.a.linfty_norm()
            << std::endl;
  std::cout << "run_initial_acceleration_linfty=" << run.a.linfty_norm()
            << std::endl;

  double early = 0.0, late = 0.0, worst_abs = 0.0;
  Vector<double> diff(run.dof.n_dofs());
  for (unsigned int s = 0; s < nsteps; ++s)
    {
      run.step(dt, beta, gamma);
      ref.step(dt, beta, gamma);
      diff = run.u;
      diff -= ref.u;
      const double rel =
        diff.linfty_norm() / std::max(1e-30, ref.u.linfty_norm());
      worst_abs = std::max(worst_abs, diff.linfty_norm());
      if (s < 10)
        early = std::max(early, rel);
      if (s >= nsteps - 10)
        late = std::max(late, rel);
      if ((s + 1) % 10 == 0)
        std::cout << "  step=" << s + 1 << " relative_deviation=" << rel
                  << " reference_linfty=" << ref.u.linfty_norm() << std::endl;
    }
  std::cout << "max_relative_deviation_first_10_steps=" << early << std::endl;
  std::cout << "max_relative_deviation_last_10_steps=" << late << std::endl;
  std::cout << "absolute_deviation_linfty=" << diff.linfty_norm() << std::endl;
  std::cout << "largest_absolute_deviation_over_the_run=" << worst_abs
            << std::endl;
  const bool measurable = early > 0.01;
  const bool decays     = (early > 0.0) && (late < 0.5 * early);
  std::cout << "spurious_mode_reaches_the_claimed_0.1_to_0.3_magnitude="
            << ((worst_abs >= 0.1) ? "true" : "false") << std::endl;
  std::cout << "early_deviation_is_of_order_one="
            << ((early > 0.5) ? "true" : "false") << std::endl;
  std::cout << "early_deviation_is_measurable="
            << (measurable ? "true" : "false") << std::endl;
  std::cout << "deviation_decays_at_later_times=" << (decays ? "true" : "false")
            << std::endl;
  std::cout << "VERDICT="
            << (measurable
                  ? "zeroing_the_initial_acceleration_moves_the_solution"
                  : "initial_acceleration_choice_left_no_trace")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------- wave#4
// No absorbing term on the outflow boundary: the pulse reflects and the energy
// never leaves. The natural (do-nothing) boundary is the reflecting one.
static int reflecting_boundary()
{
  const unsigned int refine = 5;
  const double dt = 0.004, beta = 0.25, gamma = 0.5;
  const unsigned int nsteps = 500;   // t_end = 2.0, i.e. two domain crossings
  const double absorb = mutate() ? 1.0 : 0.0;

  Wave w;
  w.setup(refine, /*dirichlet=*/false, absorb);
  w.assemble();
  w.set_initial_state(true);
  w.prepare(dt, beta, gamma);
  std::cout << "n_dofs=" << w.dof.n_dofs() << " dt=" << dt
            << " t_end=" << dt * nsteps << std::endl;
  std::cout << "absorbing_boundary_term=" << (absorb != 0.0 ? "true" : "false")
            << std::endl;
  const double e0 = w.energy();
  std::cout << "initial_energy=" << e0 << std::endl;

  double min_after_first_decay = 1e300, max_after_min = 0.0;
  bool   seen_min = false;
  for (unsigned int s = 0; s < nsteps; ++s)
    {
      w.step(dt, beta, gamma);
      const double amp = w.u.linfty_norm();
      // Once the pulse has left the middle, look for a later RISE: that is the
      // reflection coming back.
      if (s > nsteps / 5)
        {
          if (amp < min_after_first_decay)
            {
              min_after_first_decay = amp;
              seen_min              = true;
              max_after_min         = 0.0;
            }
          else if (seen_min)
            max_after_min = std::max(max_after_min, amp);
        }
      if ((s + 1) % 100 == 0)
        std::cout << "  step=" << s + 1 << " t=" << (s + 1) * dt
                  << " linfty=" << amp << " energy=" << w.energy()
                  << std::endl;
    }
  const double e1 = w.energy();
  const double ratio = e1 / e0;
  std::cout << "final_energy=" << e1 << std::endl;
  std::cout << "energy_retained_fraction=" << ratio << std::endl;
  std::cout << "amplitude_minimum_after_pulse_left=" << min_after_first_decay
            << " later_maximum=" << max_after_min << std::endl;
  const bool trapped = ratio > 0.9;
  const bool rebounds =
    seen_min && (max_after_min > 1.5 * min_after_first_decay);
  std::cout << "energy_stays_in_the_domain=" << (trapped ? "true" : "false")
            << std::endl;
  std::cout << "amplitude_rises_again_after_falling="
            << (rebounds ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (trapped ? "reflecting_boundary_keeps_all_the_energy"
                        : "absorbing_boundary_lets_the_energy_leave")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------- wave#5
// A VTU file per time step. The section LABEL is ours, not deal.II's, so the
// probe creates it and then reads the summary back.
static int vtu_every_step_cost()
{
  const unsigned int refine = 5, nsteps = 120;
  const double dt = 0.002, beta = 0.25, gamma = 0.5;
  const unsigned int every = mutate() ? 20 : 1;

  namespace fs = std::filesystem;
  const fs::path outdir =
    fs::temp_directory_path() /
    fs::path("t2_wave_vtu_" + std::to_string(::getpid()) + "_" +
             std::to_string(every));
  fs::remove_all(outdir);
  fs::create_directories(outdir);

  double       total[3] = {0, 0, 0}, out_s[3] = {0, 0, 0},
         solve_s[3] = {0, 0, 0};
  unsigned int files[3] = {0, 0, 0};
  std::vector<std::pair<double, std::string>> times_and_names;

  // k=0 the loop under test; k=1 the same loop writing every 20th step;
  // k=2 the loop under test again but with the one-shot factorisation the
  // catalog recommends for a constant dt and mesh — the claim's ">50% of
  // total" is a statement about a RATIO, so what the output is compared
  // against decides the answer, and both comparisons are reported.
  for (int k = 0; k < 3; ++k)
    {
      const unsigned int stride = (k == 1) ? 20 : every;
      Wave               w;
      w.setup(refine, true);
      w.assemble();
      w.set_initial_state(true);
      w.prepare(dt, beta, gamma);
      SparseDirectUMFPACK fact;
      if (k == 2)
        {
          fact.initialize(w.S);
          w.factorisation = &fact;
        }
      TimerOutput timer(std::cout, TimerOutput::never,
                        TimerOutput::wall_times);
      auto t0 = std::chrono::steady_clock::now();
      for (unsigned int s = 0; s < nsteps; ++s)
        {
          {
            TimerOutput::Scope sc(timer, "assemble_system");
            // assembled once, before the loop: nothing to do per step
          }
          {
            TimerOutput::Scope sc(timer, "solve");
            w.step(dt, beta, gamma);
          }
          if ((s % stride) == 0)
            {
              TimerOutput::Scope sc(timer, "output_results");
              DataOut<dim>       data_out;
              data_out.attach_dof_handler(w.dof);
              data_out.add_data_vector(w.u, "displacement");
              data_out.build_patches();
              char name[64];
              std::snprintf(name, sizeof(name), "wave_%04u.vtu", s);
              std::ofstream o(outdir / name);
              data_out.write_vtu(o);
              if (k == 0)
                times_and_names.emplace_back((s + 1) * dt, std::string(name));
              ++files[k];
            }
        }
      total[k] = std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - t0)
                   .count();
      const auto data =
        timer.get_summary_data(TimerOutput::OutputData::total_wall_time);
      out_s[k]   = data.count("output_results") ? data.at("output_results") : 0.0;
      solve_s[k] = data.at("solve");
      if (k != 1)
        {
          std::cout << "=== TimerOutput::print_summary, "
                    << (k == 0 ? "CG+SSOR per step" : "one-shot factorisation")
                    << std::endl;
          timer.print_summary();
          std::cout << "solve_section_seconds=" << data.at("solve")
                    << std::endl;
        }
    }

  // The recommended half of the entry: one .pvd instead of a heap of .vtu.
  {
    std::ofstream pvd(outdir / "wave.pvd");
    DataOutBase::write_pvd_record(pvd, times_and_names);
  }
  const bool pvd_ok = fs::exists(outdir / "wave.pvd");

  unsigned int on_disk = 0;
  for (const auto &e : fs::directory_iterator(outdir))
    if (e.path().extension() == ".vtu")
      ++on_disk;

  const double frac      = out_s[0] / total[0];
  const double frac_fact = out_s[2] / total[2];
  std::cout << "output_every_n_steps=" << every << " n_steps=" << nsteps
            << std::endl;
  std::cout << "vtu_files_written=" << files[0]
            << " vtu_files_on_disk=" << on_disk << std::endl;
  std::cout << "loop_under_test_seconds=" << total[0]
            << " output_section_seconds=" << out_s[0]
            << " solve_section_seconds_cg=" << solve_s[0] << std::endl;
  std::cout << "every_20th_step_loop_seconds=" << total[1]
            << " its_output_section_seconds=" << out_s[1] << std::endl;
  std::cout << "factorised_loop_seconds=" << total[2]
            << " its_output_section_seconds=" << out_s[2]
            << " its_solve_section_seconds=" << solve_s[2] << std::endl;
  std::cout << "output_fraction_of_wall_time_with_cg=" << frac << std::endl;
  std::cout << "output_fraction_of_wall_time_with_factorisation=" << frac_fact
            << std::endl;
  std::cout << "slowdown_vs_output_every_20=" << total[0] / total[1]
            << std::endl;
  std::cout << "write_pvd_record_produced_a_file=" << (pvd_ok ? "true"
                                                              : "false")
            << std::endl;
  fs::remove_all(outdir);

  const bool multiplies = (total[0] / total[1]) > 2.0;
  std::cout << "output_section_is_over_half_the_wall_time_with_cg="
            << ((frac > 0.5) ? "true" : "false") << std::endl;
  std::cout << "output_section_is_over_half_the_wall_time_with_factorisation="
            << ((frac_fact > 0.5) ? "true" : "false") << std::endl;
  std::cout << "output_section_is_the_top_entry_with_factorisation="
            << ((out_s[2] > solve_s[2]) ? "true" : "false") << std::endl;
  std::cout << "per_step_output_multiplies_the_wall_time="
            << (multiplies ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (multiplies ? "one_vtu_per_step_multiplies_the_wall_time"
                           : "output_stride_made_no_measurable_difference")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  if (probe == "newmark_beta_stability")
    return newmark_beta_stability();
  if (probe == "reassemble_each_step")
    return reassemble_each_step();
  if (probe == "explicit_newmark_cfl")
    return explicit_newmark_cfl();
  if (probe == "zero_initial_acceleration")
    return zero_initial_acceleration();
  if (probe == "reflecting_boundary")
    return reflecting_boundary();
  if (probe == "vtu_every_step_cost")
    return vtu_every_step_cost();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
