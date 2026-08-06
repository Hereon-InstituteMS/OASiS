// Shared translation unit for the Navier-Stokes Signal family.
// One compile serves several fixture directories; each fixture runs one probe.
//
// usage: ns_family <probe>
//   linear_solve_is_stokes | equal_order_checkerboard | reynolds_continuation
//   | time_integrator_order | pressure_level_undetermined | supg_tau_dimension
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.
//
// The steady model problem is the lid-driven cavity on the unit square with
// Taylor-Hood Q2/Q1 and a Newton linearisation of
//    nu (grad u, grad v) + ((u . grad) u, v) - (p, div v) - (q, div u) = 0,
// whose Jacobian carries BOTH linearisations of the convective term,
//    ((du . grad) u, v) + ((u . grad) du, v).
// deal.II has no Navier-Stokes solver and no Newton solver, so the outer loop,
// the continuation and every diagnostic below are the user's own.
//
// The unsteady probe uses the 2D Taylor-Green vortex, which is an EXACT
// solution of the incompressible Navier-Stokes equations:
//    u = e^{-2 pi^2 nu t} (-cos(pi x) sin(pi y), sin(pi x) cos(pi y))
//    p = -1/4 e^{-4 pi^2 nu t} (cos(2 pi x) + cos(2 pi y))
// so the time error can be measured against the analytic answer with no
// reference run.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <map>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace dealii;
constexpr int dim = 2;

static bool mutate()
{
  const char *m = std::getenv("T2_MUTATE");
  return m != nullptr && std::string(m) == "1";
}

// optional argv[2] / argv[3], for calibration runs only
static double       g_scan  = 0.0;
static unsigned int g_scan2 = 0;

static std::string flatten(const std::string &s, std::size_t n = 240)
{
  std::string f;
  for (char c : s)
    f += (c == '\n') ? ' ' : c;
  return f.substr(0, n);
}

// velocity -> block 0, pressure -> block 1 (step-22's grouping)
static std::vector<unsigned int> block_component()
{
  std::vector<unsigned int> bc(dim + 1, 0);
  bc[dim] = 1;
  return bc;
}

// ---------------------------------------------------------------------------
// The Taylor-Green vortex, used as an exact solution by the unsteady probe.
// ---------------------------------------------------------------------------
class TaylorGreen : public Function<dim>
{
public:
  TaylorGreen(double nu, double t)
    : Function<dim>(dim + 1)
    , nu(nu)
    , t(t)
  {}
  void vector_value(const Point<dim> &p, Vector<double> &v) const override
  {
    const double pi = numbers::PI;
    const double a  = std::exp(-2.0 * pi * pi * nu * t);
    v[0]            = -a * std::cos(pi * p[0]) * std::sin(pi * p[1]);
    v[1]            = a * std::sin(pi * p[0]) * std::cos(pi * p[1]);
    v[dim]          = -0.25 * a * a *
             (std::cos(2.0 * pi * p[0]) + std::cos(2.0 * pi * p[1]));
  }
  double nu, t;
};

// ---------------------------------------------------------------------------
// A cavity / box Navier-Stokes solver.
// ---------------------------------------------------------------------------
struct Flow
{
  Triangulation<dim>        tria;
  FESystem<dim>             fe;
  DoFHandler<dim>           dof;
  AffineConstraints<double> constraints;
  SparsityPattern           sp;
  SparseMatrix<double>      K;
  Vector<double>            sol, du, rhs, old_sol;
  types::global_dof_index   n_u = 0, n_p = 0;

  double nu           = 1.0;
  double lid          = 1.0;
  bool   pin_pressure = true;
  bool   convection   = true;    // assemble the convective term at all
  double dt           = 0.0;     // 0 = steady
  double theta        = 1.0;     // 1 = backward Euler, 0.5 = Crank-Nicolson
  bool   taylor_green = false;   // Dirichlet data from the exact solution
  double time         = 0.0;

  // measured by the last solve
  unsigned int last_inner_steps = 0;
  std::string  last_inner_message;

  Flow(unsigned int vel_degree, unsigned int pre_degree)
    : fe(FE_Q<dim>(vel_degree) ^ dim, FE_Q<dim>(pre_degree))
    , dof(tria)
  {}

  void make_grid(unsigned int refine, double distort = 0.0)
  {
    GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
    tria.refine_global(refine);
    if (distort > 0.0)
      GridTools::distort_random(distort, tria, false);
    dof.distribute_dofs(fe);
    DoFRenumbering::component_wise(dof, block_component());
    const auto counts =
      DoFTools::count_dofs_per_fe_block(dof, block_component());
    n_u = counts[0];
    n_p = counts[1];
    sol.reinit(dof.n_dofs());
    du.reinit(dof.n_dofs());
    rhs.reinit(dof.n_dofs());
    old_sol.reinit(dof.n_dofs());
  }

  // `inhomogeneous` carries the driving boundary data (first Newton iteration);
  // later iterations solve for an increment that must not move the boundary.
  // Newton solves for an INCREMENT, so an inhomogeneous constraint makes du
  // equal the prescribed value and `sol += du` would ADD the boundary data
  // again at every load stage or time step. The constraint therefore carries
  // the GAP between the prescribed value and what the current iterate holds --
  // inhomogeneous on the first iteration of a stage, zero afterwards.
  void set_bc(bool inhomogeneous)
  {
    constraints.clear();
    std::map<types::global_dof_index, double> bv;
    const FEValuesExtractors::Vector          vel(0);
    const ComponentMask                       vmask = fe.component_mask(vel);
    if (taylor_green)
      {
        TaylorGreen exact(nu, time);
        for (types::boundary_id b : {0, 1, 2, 3})
          VectorTools::interpolate_boundary_values(dof, b, exact, bv, vmask);
      }
    else
      {
        for (types::boundary_id b : {0, 1, 2})
          VectorTools::interpolate_boundary_values(
            dof, b, Functions::ZeroFunction<dim>(dim + 1), bv, vmask);
        std::vector<double> top(dim + 1, 0.0);
        top[0] = lid;
        VectorTools::interpolate_boundary_values(
          dof, 3, Functions::ConstantFunction<dim>(top), bv, vmask);
      }
    if (pin_pressure)
      bv[n_u] = 0.0;   // pressure dof 0 := 0
    for (const auto &e : bv)
      {
        constraints.add_line(e.first);
        constraints.set_inhomogeneity(
          e.first, inhomogeneous ? (e.second - sol(e.first)) : 0.0);
      }
    constraints.close();
  }

  void allocate()
  {
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_sparsity_pattern(dof, dsp, constraints, true);
    sp.copy_from(dsp);
    K.reinit(sp);
  }

  // Assemble the Newton system at `sol`; returns the residual norm over the
  // free rows.
  double assemble(bool want_matrix = true)
  {
    K   = 0.0;
    rhs = 0.0;
    QGauss<dim>   quad(fe.degree + 2);
    FEValues<dim> fev(fe, quad,
                      update_values | update_gradients |
                        update_quadrature_points | update_JxW_values);
    const unsigned int                   n = fe.dofs_per_cell;
    FullMatrix<double>                   cm(n, n);
    Vector<double>                       cv(n);
    std::vector<types::global_dof_index> local(n);
    const FEValuesExtractors::Vector     v(0);
    const FEValuesExtractors::Scalar     pr(dim);

    const unsigned int          nq = quad.size();
    std::vector<Tensor<1, dim>> uq(nq), uold(nq);
    std::vector<Tensor<2, dim>> gq(nq), gold(nq);
    std::vector<double>         pq(nq), divq(nq), divold(nq);

    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        cv = 0.0;
        fev[v].get_function_values(sol, uq);
        fev[v].get_function_gradients(sol, gq);
        fev[v].get_function_divergences(sol, divq);
        fev[pr].get_function_values(sol, pq);
        if (dt > 0.0)
          {
            fev[v].get_function_values(old_sol, uold);
            fev[v].get_function_gradients(old_sol, gold);
          }
        for (unsigned int q = 0; q < nq; ++q)
          {
            // the stationary part of the residual at the new state
            const Tensor<1, dim> conv_new = convection ? gq[q] * uq[q]
                                                       : Tensor<1, dim>();
            const Tensor<1, dim> conv_old = convection ? gold[q] * uold[q]
                                                       : Tensor<1, dim>();
            for (unsigned int i = 0; i < n; ++i)
              {
                const Tensor<1, dim> phi_i  = fev[v].value(i, q);
                const Tensor<2, dim> gphi_i = fev[v].gradient(i, q);
                const double         div_i  = fev[v].divergence(i, q);
                const double         q_i    = fev[pr].value(i, q);

                double res = 0.0;
                if (dt > 0.0)
                  {
                    res += ((uq[q] - uold[q]) * phi_i) / dt;
                    res += theta * (nu * scalar_product(gq[q], gphi_i) +
                                    conv_new * phi_i);
                    res += (1.0 - theta) *
                           (nu * scalar_product(gold[q], gphi_i) +
                            conv_old * phi_i);
                  }
                else
                  res += nu * scalar_product(gq[q], gphi_i) + conv_new * phi_i;
                res += -pq[q] * div_i - q_i * divq[q];
                cv(i) -= res * fev.JxW(q);

                if (!want_matrix)
                  continue;
                for (unsigned int j = 0; j < n; ++j)
                  {
                    const Tensor<1, dim> phi_j  = fev[v].value(j, q);
                    const Tensor<2, dim> gphi_j = fev[v].gradient(j, q);
                    const double         div_j  = fev[v].divergence(j, q);
                    const double         p_j    = fev[pr].value(j, q);
                    double               k      = 0.0;
                    const double         w      = (dt > 0.0) ? theta : 1.0;
                    if (dt > 0.0)
                      k += (phi_j * phi_i) / dt;
                    k += w * nu * scalar_product(gphi_j, gphi_i);
                    if (convection)
                      k += w * (((gq[q] * phi_j) + (gphi_j * uq[q])) * phi_i);
                    k += -p_j * div_i - q_i * div_j;
                    cm(i, j) += k * fev.JxW(q);
                  }
              }
          }
        cell->get_dof_indices(local);
        constraints.distribute_local_to_global(cm, cv, local, K, rhs);
      }
    double s = 0.0;
    for (types::global_dof_index i = 0; i < dof.n_dofs(); ++i)
      if (!constraints.is_constrained(i))
        s += rhs(i) * rhs(i);
    return std::sqrt(s);
  }

  // direct solve unless `gmres` is asked for
  bool solve_increment(bool gmres, unsigned int max_it = 20000)
  {
    du                 = 0.0;
    last_inner_steps   = 0;
    last_inner_message.clear();
    try
      {
        if (!gmres)
          {
            SparseDirectUMFPACK d;
            d.initialize(K);
            d.vmult(du, rhs);
          }
        else
          {
            SolverControl ctrl(max_it, 1e-10 * std::max(1.0, rhs.l2_norm()));
            SolverGMRES<Vector<double>> g(
              ctrl, SolverGMRES<Vector<double>>::AdditionalData(200));
            g.solve(K, du, rhs, PreconditionIdentity());
            last_inner_steps = ctrl.last_step();
          }
      }
    catch (const SolverControl::NoConvergence &e)
      {
        last_inner_steps   = e.last_step;
        last_inner_message = flatten(e.what());
        return false;
      }
    catch (const std::exception &e)
      {
        last_inner_message = flatten(e.what());
        return false;
      }
    return true;
  }

  // Newton loop. Returns the residual history; `ok` says whether it converged.
  std::vector<double> newton(unsigned int max_it, double tol, bool &ok,
                             bool gmres = false,
                             std::vector<unsigned int> *inner = nullptr)
  {
    std::vector<double> hist;
    ok         = false;
    double  r0 = 0.0;
    for (unsigned int it = 0; it < max_it; ++it)
      {
        set_bc(it == 0);
        const double r = assemble(true);
        hist.push_back(r);
        if (!std::isfinite(r))
          return hist;
        if (it == 0)
          r0 = r;
        if (it > 0 && r < tol * std::max(1.0, r0))
          {
            ok = true;
            return hist;
          }
        if (!solve_increment(gmres))
          return hist;
        if (inner)
          inner->push_back(last_inner_steps);
        constraints.distribute(du);
        sol += du;
        if (r > 1e12 * std::max(1.0, r0))
          return hist;
      }
    return hist;
  }

  double velocity_l2() const
  {
    double s = 0.0;
    for (types::global_dof_index i = 0; i < n_u; ++i)
      s += sol(i) * sol(i);
    return std::sqrt(s);
  }
  double pressure_linfty() const
  {
    double s = 0.0;
    for (types::global_dof_index i = n_u; i < dof.n_dofs(); ++i)
      s = std::max(s, std::abs(sol(i)));
    return s;
  }
  double pressure_mean() const
  {
    double s = 0.0;
    for (types::global_dof_index i = n_u; i < dof.n_dofs(); ++i)
      s += sol(i);
    return s / double(n_p);
  }

  // L2 error of the velocity against a given Function (velocity part only)
  double velocity_error(const Function<dim> &exact) const
  {
    Vector<double>              cell_err(tria.n_active_cells());
    const ComponentSelectFunction<dim> mask(std::make_pair(0, dim), dim + 1);
    VectorTools::integrate_difference(dof, sol, exact, cell_err,
                                      QGauss<dim>(fe.degree + 3),
                                      VectorTools::L2_norm, &mask);
    return VectorTools::compute_global_error(tria, cell_err,
                                             VectorTools::L2_norm);
  }

  // How far the velocity field is from being symmetric under x -> 1-x with
  // u_x even and u_y odd -- the symmetry the Stokes cavity has and the
  // Navier-Stokes cavity does not.
  double reflection_defect() const
  {
    std::map<types::global_dof_index, Point<dim>> support;
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof, support);
    // index the velocity dofs by (component, mirrored position)
    std::map<std::pair<unsigned int, long long>, double> byloc;
    std::vector<types::global_dof_index> local(fe.dofs_per_cell);
    auto key = [](unsigned int c, const Point<dim> &p) {
      return std::make_pair(c, (long long)std::llround(p[0] * 1e6) * 4000003LL +
                                 (long long)std::llround(p[1] * 1e6));
    };
    for (const auto &cell : dof.active_cell_iterators())
      {
        cell->get_dof_indices(local);
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          {
            const unsigned int c = fe.system_to_component_index(i).first;
            if (c < dim)
              byloc[key(c, support[local[i]])] = sol(local[i]);
          }
      }
    double worst = 0.0, scale = 0.0;
    for (const auto &cell : dof.active_cell_iterators())
      {
        cell->get_dof_indices(local);
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          {
            const unsigned int c = fe.system_to_component_index(i).first;
            if (c >= dim)
              continue;
            const Point<dim> &p = support[local[i]];
            Point<dim>        m(1.0 - p[0], p[1]);
            const auto        it = byloc.find(key(c, m));
            if (it == byloc.end())
              continue;
            const double mine = sol(local[i]);
            const double mirr = (c == 0) ? it->second : -it->second;
            worst             = std::max(worst, std::abs(mine - mirr));
            scale             = std::max(scale, std::abs(mine));
          }
      }
    return worst / std::max(1e-300, scale);
  }
};

// ===========================================================================
// navier_stokes#0 -- NS is nonlinear. A single linear solve returns the Stokes
// answer whatever Reynolds number the user had in mind.
// ===========================================================================
static int linear_solve_is_stokes()
{
  const double re = (g_scan > 0.0) ? g_scan : 200.0;
  const double nu = 1.0 / re;
  std::cout << "intended_reynolds_number=" << re << " kinematic_viscosity=" << nu
            << std::endl;

  // (a) the Stokes reference: one linear solve, convective term never assembled
  Flow stokes(2, 1);
  stokes.make_grid(4);
  stokes.nu         = nu;
  stokes.convection = false;
  stokes.set_bc(true);
  stokes.allocate();
  stokes.assemble(true);
  stokes.solve_increment(false);
  stokes.constraints.distribute(stokes.du);
  stokes.sol = stokes.du;
  std::cout << "stokes_reference_n_dofs=" << stokes.dof.n_dofs()
            << " velocity_l2=" << stokes.velocity_l2() << std::endl;

  // (b) the converged Navier-Stokes answer at the SAME nu, by continuation
  Flow ns(2, 1);
  ns.make_grid(4);
  ns.nu = nu;
  ns.set_bc(true);
  ns.allocate();
  bool ok = false;
  for (double r : {10.0, 50.0, 100.0, re})
    {
      ns.nu = 1.0 / r;
      ns.newton(30, 1e-10, ok);
      if (!ok)
        break;
    }
  std::cout << "navier_stokes_reference_converged=" << (ok ? "true" : "false")
            << " velocity_l2=" << ns.velocity_l2() << std::endl;

  // the field the user actually gets
  const Flow &under = mutate() ? ns : stokes;
  std::cout << "formulation_under_test="
            << (mutate() ? "newton_iteration_with_the_convective_term"
                         : "one_linear_solve_without_the_convective_term")
            << std::endl;

  Vector<double> ds(under.sol);
  ds -= stokes.sol;
  Vector<double> dn(under.sol);
  dn -= ns.sol;
  const double rel_stokes = ds.l2_norm() / std::max(1e-300, stokes.sol.l2_norm());
  const double rel_ns     = dn.l2_norm() / std::max(1e-300, ns.sol.l2_norm());
  std::cout << "relative_difference_from_the_stokes_reference=" << rel_stokes
            << std::endl;
  std::cout << "relative_difference_from_the_navier_stokes_reference=" << rel_ns
            << std::endl;
  std::cout << "matches_the_stokes_answer="
            << ((rel_stokes < 1e-10) ? "true" : "false") << std::endl;
  std::cout << "differs_from_the_navier_stokes_answer="
            << ((rel_ns > 0.05) ? "true" : "false") << std::endl;

  // the physical fingerprint: Stokes flow in a cavity is reflection symmetric
  const double sd_stokes = stokes.reflection_defect();
  const double sd_ns     = ns.reflection_defect();
  const double sd_under  = under.reflection_defect();
  std::cout << "stokes_reflection_defect=" << sd_stokes
            << " navier_stokes_reflection_defect=" << sd_ns << std::endl;
  std::cout << "answer_under_test_reflection_defect=" << sd_under << std::endl;
  std::cout << "answer_under_test_is_reflection_symmetric="
            << ((sd_under < 1e-8) ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << ((rel_stokes < 1e-10)
                  ? "a_single_linear_solve_returns_the_zero_reynolds_answer"
                  : "the_convective_term_is_actually_solved_for")
            << std::endl;
  return 0;
}

// ===========================================================================
// navier_stokes#1 -- Q1/Q1 has no inf-sup stability and the pressure
// checkerboards. Measured on the pressure at cell centres, exactly the signal
// the claim names.
// ===========================================================================
static int equal_order_checkerboard()
{
  const unsigned int refine = 4;
  const unsigned int vdeg   = mutate() ? 2 : 1;
  std::cout << "velocity_degree=" << vdeg << " pressure_degree=1" << std::endl;
  std::cout << "pair_under_test="
            << (mutate() ? "taylor_hood_Q2_Q1" : "equal_order_Q1_Q1")
            << std::endl;
  Flow f(vdeg, 1);
  // A slightly distorted mesh: on a perfectly uniform grid the checkerboard is
  // EXACTLY in the kernel and the direct solve has nothing to return. Distorted,
  // the mode is only nearly singular -- which is what a real mesh gives you.
  f.make_grid(refine, (g_scan > 0.0) ? g_scan : 0.15);
  f.nu = 0.02;
  f.set_bc(true);
  f.allocate();
  bool ok = false;
  const auto hist = f.newton(30, 1e-10, ok);
  std::cout << "newton_converged=" << (ok ? "true" : "false")
            << " newton_steps=" << hist.size() << std::endl;
  std::cout << "n_dofs=" << f.dof.n_dofs() << " n_pressure_dofs=" << f.n_p
            << std::endl;

  // pressure at every cell centre
  QMidpoint<dim> mid;
  FEValues<dim>  fev(f.fe, mid, update_values | update_quadrature_points);
  const FEValuesExtractors::Scalar pr(dim);
  std::vector<double>              pv(1);
  const double                     h = 1.0 / double(1u << refine);
  std::map<std::pair<int, int>, double> grid;
  double                                pmin = 1e300, pmax = -1e300;
  for (const auto &cell : f.dof.active_cell_iterators())
    {
      fev.reinit(cell);
      fev[pr].get_function_values(f.sol, pv);
      const Point<dim> c = fev.quadrature_point(0);
      const int ix = int(std::llround(c[0] / h - 0.5));
      const int iy = int(std::llround(c[1] / h - 0.5));
      grid[{ix, iy}] = pv[0];
      pmin           = std::min(pmin, pv[0]);
      pmax           = std::max(pmax, pv[0]);
    }
  const double range = pmax - pmin;
  std::cout << "cell_centre_pressure_min=" << pmin << " max=" << pmax
            << " range=" << range << std::endl;

  // The pressure LEVEL is arbitrary in a closed cavity, so the pattern has to
  // be measured on the mean-free part. Projection onto the checkerboard sign
  // pattern reaches 1 for a pure checkerboard and stays near 0 for a smooth
  // field; the second number is the fraction of horizontally adjacent cell
  // pairs across which the mean-free pressure changes sign.
  double mean = 0.0;
  for (const auto &e : grid)
    mean += e.second;
  mean /= double(std::max<std::size_t>(1, grid.size()));
  double sum_abs = 0.0, proj = 0.0;
  for (const auto &e : grid)
    {
      const double s = ((e.first.first + e.first.second) % 2 == 0) ? 1.0 : -1.0;
      proj += s * (e.second - mean);
      sum_abs += std::abs(e.second - mean);
    }
  const double checker = std::abs(proj) / std::max(1e-300, sum_abs);
  std::cout << "cell_centre_pressure_mean=" << mean
            << " mean_free_amplitude=" << (sum_abs / double(grid.size()))
            << std::endl;
  unsigned int alt = 0, pairs = 0;
  const int    n   = int(1u << refine);
  for (int iy = 0; iy < n; ++iy)
    for (int ix = 0; ix + 1 < n; ++ix)
      {
        const auto a = grid.find({ix, iy}), b = grid.find({ix + 1, iy});
        if (a == grid.end() || b == grid.end())
          continue;
        ++pairs;
        if ((a->second - mean) * (b->second - mean) < 0.0)
          ++alt;
      }
  const double alt_frac = double(alt) / std::max(1u, pairs);
  std::cout << "checkerboard_projection_of_the_cell_centre_pressure=" << checker
            << std::endl;
  std::cout << "fraction_of_adjacent_cell_pairs_that_alternate=" << alt_frac
            << std::endl;
  // The magnitude is the other half of the signal: with no inf-sup stability
  // the spurious mode is in the kernel of the discrete gradient, so the direct
  // solve returns an arbitrary multiple of it.
  std::cout << "max_abs_pressure_dof=" << f.pressure_linfty() << std::endl;
  std::cout << "pressure_magnitude_exceeds_1e10="
            << ((f.pressure_linfty() > 1e10) ? "true" : "false") << std::endl;
  const bool checkers = checker > 0.40 && alt_frac > 0.30;
  std::cout << "pressure_checkerboards=" << (checkers ? "true" : "false")
            << std::endl;
  std::cout << "VERDICT="
            << (checkers ? "equal_order_pressure_checkerboards"
                         : "pressure_field_is_clean")
            << std::endl;
  return 0;
}

// ===========================================================================
// navier_stokes#2 -- continuation in the Reynolds number.
// ===========================================================================
static int reynolds_continuation()
{
  const double       target = (g_scan > 0.0) ? g_scan : 700.0;
  const unsigned int refine = (g_scan2 > 0) ? g_scan2 : 5;
  std::cout << "target_reynolds_number=" << target
            << " cells_per_side=" << (1u << refine) << std::endl;
  std::cout << "start_under_test="
            << (mutate() ? "continuation_through_lower_reynolds_numbers"
                         : "cold_start_from_a_zero_initial_guess")
            << std::endl;
  Flow f(2, 1);
  f.make_grid(refine);
  f.set_bc(true);
  f.allocate();
  bool                      ok = false;
  std::vector<double>       hist;
  std::vector<unsigned int> steps_per_stage;
  if (mutate())
    {
      for (double r : {10.0, 50.0, 100.0, 200.0, 400.0, target})
        {
          f.nu       = 1.0 / r;
          const auto h = f.newton(30, 1e-10, ok);
          steps_per_stage.push_back(h.size());
          std::cout << "stage_reynolds=" << r << " newton_steps=" << h.size()
                    << " converged=" << (ok ? "true" : "false") << std::endl;
          hist = h;
          if (!ok)
            break;
        }
    }
  else
    {
      f.nu = 1.0 / target;
      hist = f.newton(30, 1e-10, ok);
      steps_per_stage.push_back(hist.size());
    }
  std::cout << "final_newton_residual_history=";
  for (unsigned int i = 0; i < hist.size() && i < 14; ++i)
    std::cout << (i ? "," : "") << hist[i];
  std::cout << std::endl;
  std::cout << "newton_converged=" << (ok ? "true" : "false") << std::endl;
  std::cout << "newton_steps_on_the_final_stage=" << hist.size() << std::endl;
  bool over = false;
  for (double v : hist)
    if (!std::isfinite(v) || v > 1e3)
      over = true;
  std::cout << "a_newton_residual_exceeded_1e3=" << (over ? "true" : "false")
            << std::endl;
  std::cout << "final_stage_took_between_four_and_six_steps="
            << ((ok && hist.size() >= 4 && hist.size() <= 6) ? "true" : "false")
            << std::endl;
  std::cout << "VERDICT="
            << (ok ? "the_starting_guess_reaches_the_target_reynolds_number"
                   : "cold_start_newton_fails_at_the_target_reynolds_number")
            << std::endl;
  return 0;
}

// ===========================================================================
// navier_stokes#3 -- backward Euler is first order in time, Crank-Nicolson is
// second. Measured against the exact Taylor-Green vortex.
// ===========================================================================
static double taylor_green_error(double theta, unsigned int nsteps, double T,
                                 double nu, unsigned int refine, bool &ok)
{
  Flow f(2, 1);
  f.make_grid(refine);
  f.nu           = nu;
  f.theta        = theta;
  f.taylor_green = true;
  f.dt           = T / double(nsteps);
  f.time         = 0.0;
  f.set_bc(true);
  f.allocate();
  // exact initial condition
  VectorTools::interpolate(f.dof, TaylorGreen(nu, 0.0), f.sol);
  ok = true;
  for (unsigned int s = 1; s <= nsteps; ++s)
    {
      f.old_sol = f.sol;
      f.time    = double(s) * f.dt;
      bool step_ok = false;
      f.newton(12, 1e-12, step_ok);
      if (!step_ok)
        ok = false;
    }
  return f.velocity_error(TaylorGreen(nu, T));
}

static int time_integrator_order()
{
  const double       nu     = 0.05;
  const double       T      = (g_scan > 0.0) ? g_scan : 0.5;
  const unsigned int refine = (g_scan2 > 0) ? g_scan2 : 4;
  const double       theta  = mutate() ? 0.5 : 1.0;
  std::cout << "scheme_under_test="
            << (mutate() ? "crank_nicolson_theta_one_half"
                         : "backward_euler_theta_one")
            << std::endl;
  std::cout << "viscosity=" << nu << " final_time=" << T
            << " cells_per_side=" << (1u << refine) << std::endl;
  bool   ok1, ok2, okb1, okb2;
  double e[2], eb[2];
  const unsigned int ns[2] = {4, 8};
  e[0]  = taylor_green_error(theta, ns[0], T, nu, refine, ok1);
  e[1]  = taylor_green_error(theta, ns[1], T, nu, refine, ok2);
  eb[0] = taylor_green_error(mutate() ? 1.0 : 0.5, ns[0], T, nu, refine, okb1);
  eb[1] = taylor_green_error(mutate() ? 1.0 : 0.5, ns[1], T, nu, refine, okb2);
  for (int k = 0; k < 2; ++k)
    std::cout << "steps=" << ns[k] << " dt=" << (T / ns[k])
              << " l2_velocity_error_under_test=" << e[k]
              << " l2_velocity_error_other_scheme=" << eb[k] << std::endl;
  const double order = std::log(e[0] / e[1]) / std::log(2.0);
  const double order_other = std::log(eb[0] / eb[1]) / std::log(2.0);
  std::cout << "observed_temporal_order_under_test=" << order << std::endl;
  std::cout << "observed_temporal_order_other_scheme=" << order_other
            << std::endl;
  std::cout << "all_time_steps_converged="
            << ((ok1 && ok2 && okb1 && okb2) ? "true" : "false") << std::endl;
  const bool first_order  = order < 1.35;
  const bool second_order = order > 1.7;
  std::cout << "observed_order_is_first=" << (first_order ? "true" : "false")
            << std::endl;
  std::cout << "observed_order_is_second=" << (second_order ? "true" : "false")
            << std::endl;
  const double ratio = e[1] / std::max(1e-300, eb[1]);
  std::cout << "error_ratio_under_test_over_other_scheme=" << ratio
            << std::endl;
  std::cout << "scheme_under_test_is_the_less_accurate_one="
            << ((ratio > 3.0) ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (first_order ? "backward_euler_is_only_first_order_in_time"
                            : "scheme_under_test_is_second_order_in_time")
            << std::endl;
  return 0;
}

// ===========================================================================
// navier_stokes#4 -- the pressure of a closed cavity is fixed only up to a
// constant.
// ===========================================================================
static int pressure_level_undetermined()
{
  std::cout << "pressure_constraint="
            << (mutate() ? "one_pressure_dof_pinned" : "none") << std::endl;
  // Two runs that differ ONLY in the pressure level of the starting vector.
  double       plevel[2], pmax[2];
  double       vel[2];
  bool         ok[2];
  Vector<double> keep[2];
  std::vector<unsigned int> inner[2];
  for (int k = 0; k < 2; ++k)
    {
      Flow f(2, 1);
      f.make_grid(3);
      f.nu           = 1.0 / 100.0;
      f.pin_pressure = mutate();
      f.set_bc(true);
      f.allocate();
      if (k == 1)
        for (types::global_dof_index i = f.n_u; i < f.dof.n_dofs(); ++i)
          f.sol(i) = 1000.0;
      const auto hist = f.newton(20, 1e-9, ok[k], true, &inner[k]);
      plevel[k]       = f.pressure_mean();
      pmax[k]         = f.pressure_linfty();
      vel[k]          = f.velocity_l2();
      keep[k]         = f.sol;
      std::cout << "run" << k << "_start_pressure_offset="
                << ((k == 1) ? 1000.0 : 0.0) << " newton_steps=" << hist.size()
                << " converged=" << (ok[k] ? "true" : "false")
                << " velocity_l2=" << vel[k] << " pressure_mean=" << plevel[k]
                << " max_abs_pressure=" << pmax[k] << std::endl;
      std::cout << "run" << k << "_gmres_steps_per_newton_step=";
      for (unsigned int i = 0; i < inner[k].size() && i < 12; ++i)
        std::cout << (i ? "," : "") << inner[k][i];
      std::cout << std::endl;
    }
  // the velocity field, and the pressure SHAPE, must agree
  double vgap = 0.0, sgap = 0.0;
  const types::global_dof_index nu_ = 0;
  (void)nu_;
  {
    Flow probe(2, 1);
    probe.make_grid(3);
    const types::global_dof_index nu2 = probe.n_u;
    for (types::global_dof_index i = 0; i < nu2; ++i)
      vgap = std::max(vgap, std::abs(keep[0](i) - keep[1](i)));
    for (types::global_dof_index i = nu2; i < probe.dof.n_dofs(); ++i)
      sgap = std::max(sgap,
                      std::abs((keep[0](i) - plevel[0]) -
                               (keep[1](i) - plevel[1])));
  }
  std::cout << "max_velocity_difference_between_the_runs=" << vgap << std::endl;
  std::cout << "max_pressure_shape_difference_between_the_runs=" << sgap
            << std::endl;
  std::cout << "pressure_mean_difference_between_the_runs="
            << std::abs(plevel[0] - plevel[1]) << std::endl;
  const bool both_ok      = ok[0] && ok[1];
  const bool same_flow    = vgap < 1e-6 && sgap < 1e-6;
  const bool level_differs = std::abs(plevel[0] - plevel[1]) > 1.0;
  std::cout << "both_runs_converged=" << (both_ok ? "true" : "false")
            << std::endl;
  std::cout << "velocity_and_pressure_shape_agree="
            << (same_flow ? "true" : "false") << std::endl;
  std::cout << "pressure_level_differs=" << (level_differs ? "true" : "false")
            << std::endl;
  // the claim's own number, measured
  std::cout << "pressure_exceeded_1e10="
            << ((pmax[0] > 1e10 || pmax[1] > 1e10) ? "true" : "false")
            << std::endl;
  std::cout << "VERDICT="
            << ((same_flow && level_differs)
                  ? "the_pressure_level_is_undetermined_without_a_constraint"
                  : "the_pressure_level_is_determined")
            << std::endl;
  return 0;
}

// ===========================================================================
// navier_stokes#5 -- the SUPG parameter tau and the element length it needs.
// Scalar transport by the frozen advection field of a momentum equation, which
// is where tau lives; the boundary layer sits at the outflow wall.
// ===========================================================================
enum class Tau
{
  none,
  isotropic_h,       // h / (2|a|): the textbook one-dimensional formula
  claim_over_sqrt2,  // h / (2 sqrt(2) |a|): what the claim prescribes for 2D
  streamline         // h_streamline / (2|a|): the standard multi-D length
};

static void supg_solve(Tau kind, unsigned int ncells, double eps,
                       double &umin, double &umax, double &tau_mean,
                       double &wall_dist)
{
  // shape gradients at the cell centre, where the streamline element length of
  // the multi-dimensional tau formula is evaluated

  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_rectangle(
    tria, {ncells, ncells}, Point<dim>(0, 0), Point<dim>(1, 1), true);
  FE_Q<dim>       fe(1);
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);
  AffineConstraints<double> c;
  for (types::boundary_id b : {0, 1, 2, 3})
    VectorTools::interpolate_boundary_values(
      dof, b, Functions::ZeroFunction<dim>(), c);
  c.close();
  DynamicSparsityPattern dsp(dof.n_dofs());
  DoFTools::make_sparsity_pattern(dof, dsp, c, true);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A(sp);
  Vector<double>       rhs(dof.n_dofs()), sol(dof.n_dofs());

  Tensor<1, dim> a;
  a[0] = 1.0 / std::sqrt(2.0);
  a[1] = 1.0 / std::sqrt(2.0);
  const double anorm = a.norm();

  QGauss<dim>    quad(2);
  QMidpoint<dim> mid;
  FEValues<dim>  fev(fe, quad,
                     update_values | update_gradients | update_JxW_values);
  FEValues<dim>  fmid(fe, mid, update_gradients);
  const unsigned int n = fe.dofs_per_cell;
  FullMatrix<double> cm(n, n);
  Vector<double>     cv(n);
  std::vector<types::global_dof_index> local(n);
  double  tau_sum = 0.0;
  unsigned int ncell = 0;
  for (const auto &cell : dof.active_cell_iterators())
    {
      fev.reinit(cell);
      fmid.reinit(cell);
      cm = 0.0;
      cv = 0.0;
      const double h = cell->diameter() / std::sqrt(2.0);   // side length
      double tau = 0.0;
      switch (kind)
        {
          case Tau::none:
            tau = 0.0;
            break;
          case Tau::isotropic_h:
            tau = h / (2.0 * anorm);
            break;
          case Tau::claim_over_sqrt2:
            tau = h / (2.0 * std::sqrt(2.0) * anorm);
            break;
          case Tau::streamline:
            {
              // h_s = 2 |a| / sum_i |a . grad N_i| AT THE CELL CENTRE. For a
              // square cell of side h and a skew at 45 degrees this is h*sqrt(2)
              // -- the length the flow actually crosses -- so the multi-D tau is
              // LARGER than the isotropic one, not smaller.
              double sg = 0.0;
              for (unsigned int i = 0; i < n; ++i)
                sg += std::abs(a * fmid.shape_grad(i, 0));
              const double hs = (sg > 0.0) ? 2.0 * anorm / sg : h;
              tau             = hs / (2.0 * anorm);
            }
            break;
        }
      tau_sum += tau;
      ++ncell;
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < n; ++i)
          {
            const double         phi_i = fev.shape_value(i, q);
            const Tensor<1, dim> g_i   = fev.shape_grad(i, q);
            const double         agi   = a * g_i;
            for (unsigned int j = 0; j < n; ++j)
              {
                const Tensor<1, dim> g_j = fev.shape_grad(j, q);
                const double         agj = a * g_j;
                cm(i, j) += (eps * (g_i * g_j) + agj * phi_i +
                             tau * agi * agj) *
                            fev.JxW(q);
              }
            cv(i) += (phi_i + tau * agi) * 1.0 * fev.JxW(q);
          }
      cell->get_dof_indices(local);
      c.distribute_local_to_global(cm, cv, local, A, rhs);
    }
  SparseDirectUMFPACK d;
  d.initialize(A);
  d.vmult(sol, rhs);
  c.distribute(sol);
  tau_mean = tau_sum / std::max(1u, ncell);

  std::map<types::global_dof_index, Point<dim>> support;
  DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof, support);
  umin = 1e300;
  umax = -1e300;
  Point<dim> peak;
  for (const auto &p : support)
    {
      const double v = sol(p.first);
      umin           = std::min(umin, v);
      if (v > umax)
        {
          umax = v;
          peak = p.second;
        }
    }
  // distance of the overshoot peak from the outflow walls x=1 / y=1
  wall_dist = std::min(1.0 - peak[0], 1.0 - peak[1]);
}

static int supg_tau_dimension()
{
  const unsigned int ncells = 24;
  const double       eps    = 1e-4;
  const double       h      = 1.0 / double(ncells);
  std::cout << "cells_per_side=" << ncells << " diffusion=" << eps
            << " advection=skew_45_degrees element_size=" << h << std::endl;
  struct R
  {
    double umin, umax, tau, wall;
  } r[4];
  const Tau        kinds[4] = {Tau::none, Tau::isotropic_h,
                               Tau::claim_over_sqrt2, Tau::streamline};
  const char *const names[4] = {"no_stabilisation", "tau_isotropic_h",
                                "tau_divided_by_sqrt2_as_the_claim_says",
                                "tau_streamline_length"};
  for (int k = 0; k < 4; ++k)
    {
      supg_solve(kinds[k], ncells, eps, r[k].umin, r[k].umax, r[k].tau,
                 r[k].wall);
      std::cout << names[k] << "_tau=" << r[k].tau << " min_u=" << r[k].umin
                << " max_u=" << r[k].umax
                << " undershoot_fraction=" << (-r[k].umin / r[k].umax)
                << " worst_point_distance_from_the_outflow_wall=" << r[k].wall
                << std::endl;
    }
  // With f = 1, |a| = 1 and zero inflow data the exact interior solution is the
  // arclength from the inflow boundary, u = sqrt(2) min(x,y), so the exact
  // interior maximum is sqrt(2) and anything above it is an overshoot. No
  // reference run is needed.
  const double exact_max = std::sqrt(2.0);
  std::cout << "exact_interior_maximum=" << exact_max << std::endl;
  for (int k = 0; k < 4; ++k)
    std::cout << names[k] << "_overshoot_fraction="
              << ((r[k].umax - exact_max) / exact_max) << std::endl;

  // the sqrt(2) the claim points at, measured between the formulas
  std::cout << "ratio_streamline_tau_over_isotropic_tau="
            << (r[3].tau / r[1].tau) << std::endl;
  std::cout << "ratio_isotropic_tau_over_the_claims_2d_tau="
            << (r[1].tau / r[2].tau) << std::endl;

  const int    idx  = mutate() ? 3 : 2;
  const double frac = (r[idx].umax - exact_max) / exact_max;
  std::cout << "tau_under_test="
            << (mutate() ? "streamline_element_length_h_s_over_2_a"
                         : "h_over_2_sqrt2_a_as_the_claim_prescribes")
            << std::endl;
  std::cout << "overshoot_fraction_under_test=" << frac << std::endl;
  std::cout << "overshoot_exceeds_half_the_exact_maximum="
            << ((frac > 0.5) ? "true" : "false") << std::endl;
  std::cout << "overshoot_peak_within_three_cells_of_the_outflow_wall="
            << ((r[idx].wall < 3.0 * h) ? "true" : "false") << std::endl;
  // Does the claim's prescription help or hurt? Both are measured against the
  // isotropic tau it says to replace.
  const double f_iso   = (r[1].umax - exact_max) / exact_max;
  const double f_claim = (r[2].umax - exact_max) / exact_max;
  const double f_sl    = (r[3].umax - exact_max) / exact_max;
  std::cout << "dividing_tau_by_sqrt2_reduces_the_overshoot="
            << ((f_claim < f_iso) ? "true" : "false") << std::endl;
  std::cout << "streamline_length_tau_reduces_the_overshoot="
            << ((f_sl < f_iso) ? "true" : "false") << std::endl;
  std::cout << "streamline_tau_is_larger_than_the_isotropic_one="
            << ((r[3].tau > r[1].tau) ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << ((frac > 0.5)
                  ? "the_claims_2d_tau_leaves_a_large_outflow_overshoot"
                  : "tau_under_test_controls_the_outflow_overshoot")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  const std::string probe = (argc > 1) ? argv[1] : "";
  if (argc > 2)
    g_scan = std::atof(argv[2]);
  if (argc > 3)
    g_scan2 = (unsigned int)std::atoi(argv[3]);
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  std::cout << std::setprecision(10);
  if (probe == "linear_solve_is_stokes")
    return linear_solve_is_stokes();
  if (probe == "equal_order_checkerboard")
    return equal_order_checkerboard();
  if (probe == "reynolds_continuation")
    return reynolds_continuation();
  if (probe == "time_integrator_order")
    return time_integrator_order();
  if (probe == "pressure_level_undetermined")
    return pressure_level_undetermined();
  if (probe == "supg_tau_dimension")
    return supg_tau_dimension();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 3;
}
