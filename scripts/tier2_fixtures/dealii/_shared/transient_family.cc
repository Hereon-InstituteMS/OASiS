// Shared translation unit for two transient Signal families:
//   * the energy behaviour of a time integrator on the wave equation, and
//   * the Boussinesq coupling between momentum and temperature, in the
//     infinite-Prandtl (Stokes) form that step-31 uses.
//
// usage: transient_family <probe>
//   implicit_euler_energy | buoyancy_coupling
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.

#include <deal.II/base/function.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
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
// time_dependent_wave#0 -- what an integrator does to the energy.
// u_tt = c^2 laplace(u) on the unit square, u = 0 on the boundary, started from
// a smooth bump at rest. The energy 0.5 v^T M v + 0.5 c^2 u^T K u is a quadratic
// form in the same matrices the scheme uses, so it is computed exactly rather
// than quadrature-approximated -- and it is ALSO computed the way the entry
// names it, from VectorTools::integrate_difference, so the two agree or the
// disagreement is printed.
// ===========================================================================
class Bump : public Function<dim>
{
public:
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  }
};

static int implicit_euler_energy()
{
  const bool newmark = mutate(); // the mistake: implicit (backward) Euler
  std::cout << "integrator_under_test="
            << (newmark ? "newmark_beta_quarter_gamma_half" : "implicit_euler")
            << std::endl;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0);
  tria.refine_global(5);
  FE_Q<dim>       fe(1);
  MappingQ1<dim>  mapping;
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(
    dof, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof.n_dofs());
  DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> M, K, S;
  M.reinit(sp);
  K.reinit(sp);
  S.reinit(sp);
  {
    QGauss<dim>   quad(3);
    FEValues<dim> fev(mapping, fe, quad,
                      update_values | update_gradients | update_JxW_values);
    const unsigned int n = fe.n_dofs_per_cell();
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
                ck(i, j) +=
                  fev.shape_grad(i, q) * fev.shape_grad(j, q) * fev.JxW(q);
              }
        cell->get_dof_indices(local);
        constraints.distribute_local_to_global(cm, local, M);
        constraints.distribute_local_to_global(ck, local, K);
      }
  }
  const double c2 = 1.0;
  const double dt = 5e-3;
  const unsigned int nsteps = 2000;

  Vector<double> u(dof.n_dofs()), v(dof.n_dofs()), a(dof.n_dofs());
  Vector<double> tmp(dof.n_dofs()), rhs(dof.n_dofs());
  VectorTools::interpolate(dof, Bump(), u);
  constraints.distribute(u);

  auto energy = [&](const Vector<double> &uu, const Vector<double> &vv) {
    Vector<double> t1(dof.n_dofs());
    M.vmult(t1, vv);
    const double kin = 0.5 * (vv * t1);
    K.vmult(t1, uu);
    const double pot = 0.5 * c2 * (uu * t1);
    return kin + pot;
  };
  // The same two pieces via the route the entry names.
  auto energy_via_integrate_difference = [&](const Vector<double> &uu,
                                             const Vector<double> &vv) {
    Vector<double> cw(tria.n_active_cells());
    VectorTools::integrate_difference(mapping, dof, vv,
                                      Functions::ZeroFunction<dim>(), cw,
                                      QGauss<dim>(3), VectorTools::L2_norm);
    const double l2v =
      VectorTools::compute_global_error(tria, cw, VectorTools::L2_norm);
    VectorTools::integrate_difference(mapping, dof, uu,
                                      Functions::ZeroFunction<dim>(), cw,
                                      QGauss<dim>(3),
                                      VectorTools::H1_seminorm);
    const double h1u =
      VectorTools::compute_global_error(tria, cw, VectorTools::H1_seminorm);
    return 0.5 * l2v * l2v + 0.5 * c2 * h1u * h1u;
  };

  const double e0 = energy(u, v);
  const double e0_id = energy_via_integrate_difference(u, v);
  std::cout << "initial_energy_from_the_matrices=" << e0
            << " initial_energy_from_integrate_difference=" << e0_id
            << std::endl;
  std::cout << "energy_from_integrate_difference_matches_the_matrix_energy="
            << yesno(std::abs(e0 - e0_id) < 1e-8 * std::abs(e0)) << std::endl;

  // Newmark beta = 1/4, gamma = 1/2 needs the initial acceleration.
  if (newmark)
    {
      K.vmult(rhs, u);
      rhs *= -c2;
      SparseMatrix<double> Mc;
      Mc.reinit(sp);
      Mc.copy_from(M);
      for (unsigned int i = 0; i < dof.n_dofs(); ++i)
        if (constraints.is_constrained(i))
          {
            for (auto it = Mc.begin(i); it != Mc.end(i); ++it)
              it->value() = (it->column() == i) ? 1.0 : 0.0;
            rhs(i) = 0.0;
          }
      SparseDirectUMFPACK inv;
      inv.initialize(Mc);
      a = rhs;
      inv.solve(a);
    }

  // The effective matrix of each scheme, assembled and factorised once.
  //   implicit Euler on (u,v): (M + dt^2 c^2 K) v^{n+1} = M v^n - dt c^2 K u^n
  //   Newmark(1/4,1/2):        (M + beta dt^2 c^2 K) a^{n+1} = -c^2 K u~
  const double beta = 0.25, gamma = 0.5;
  S.copy_from(M);
  S.add(newmark ? beta * dt * dt * c2 : dt * dt * c2, K);
  for (unsigned int i = 0; i < dof.n_dofs(); ++i)
    if (constraints.is_constrained(i))
      for (auto it = S.begin(i); it != S.end(i); ++it)
        it->value() = (it->column() == i) ? 1.0 : 0.0;
  SparseDirectUMFPACK inv;
  inv.initialize(S);

  std::vector<double> trace;
  trace.push_back(e0);
  bool monotone_decay = true;
  for (unsigned int s = 0; s < nsteps; ++s)
    {
      if (!newmark)
        {
          M.vmult(rhs, v);
          K.vmult(tmp, u);
          rhs.add(-dt * c2, tmp);
          for (unsigned int i = 0; i < dof.n_dofs(); ++i)
            if (constraints.is_constrained(i))
              rhs(i) = 0.0;
          Vector<double> vn(rhs);
          inv.solve(vn);
          v = vn;
          u.add(dt, v);
          constraints.distribute(u);
        }
      else
        {
          Vector<double> ut(u), vt(v);
          ut.add(dt, v);
          ut.add((0.5 - beta) * dt * dt, a);
          vt.add((1.0 - gamma) * dt, a);
          K.vmult(rhs, ut);
          rhs *= -c2;
          for (unsigned int i = 0; i < dof.n_dofs(); ++i)
            if (constraints.is_constrained(i))
              rhs(i) = 0.0;
          Vector<double> an(rhs);
          inv.solve(an);
          a = an;
          u = ut;
          u.add(beta * dt * dt, a);
          v = vt;
          v.add(gamma * dt, a);
          constraints.distribute(u);
        }
      const double e = energy(u, v);
      if (e > trace.back() * (1.0 + 1e-12))
        monotone_decay = false;
      trace.push_back(e);
      if ((s + 1) % 500 == 0)
        std::cout << "step=" << (s + 1) << " energy=" << e
                  << " energy_over_initial=" << e / e0 << std::endl;
    }
  const double retained = trace.back() / e0;
  const double drift = std::abs(retained - 1.0);
  std::cout << "final_energy_over_initial=" << retained
            << " relative_energy_drift=" << drift << std::endl;
  std::cout << "energy_under_test_decays_monotonically="
            << yesno(monotone_decay && retained < 1.0) << std::endl;
  std::cout << "energy_under_test_lost_more_than_a_tenth="
            << yesno(retained < 0.9) << std::endl;
  std::cout << "energy_under_test_is_conserved_to_ten_significant_figures="
            << yesno(drift < 1e-10) << std::endl;
  std::cout << "VERDICT="
            << (retained < 0.9
                  ? "integrator_under_test_dissipates_the_wave_energy"
                  : "integrator_under_test_keeps_the_energy")
            << std::endl;
  return 0;
}

// ===========================================================================
// time_dependent_ns#0 and #1 -- Boussinesq.
// Infinite-Prandtl (Stokes) Boussinesq, the step-31 form:
//   -laplace(u) + grad p = Ra * T * e_y,  div u = 0,  T_t + u.grad T = laplace T
// Taylor-Hood Q2/Q1 for (u,p) and Q2 for T on the same triangulation. Both the
// Stokes matrix and the temperature matrix are CONSTANT, so each is factorised
// once and every step is two back-substitutions.
// ===========================================================================
struct Boussinesq
{
  Triangulation<dim>        tria;
  FESystem<dim>             fe_stokes;
  FE_Q<dim>                 fe_temp;
  MappingQ1<dim>            mapping;
  DoFHandler<dim>           dof_stokes, dof_temp;
  AffineConstraints<double> con_stokes, con_temp;
  SparsityPattern           sp_stokes, sp_temp;
  SparseMatrix<double>      A_stokes, A_temp, M_temp;
  Vector<double>            stokes_sol, stokes_rhs, T, T_rhs;
  SparseDirectUMFPACK       inv_stokes, inv_temp;
  double                    Ra = 0.0, dt = 0.0, width = 1.0;
  // Subtracting a purely conductive profile from the buoyancy term changes
  // nothing physical (it is a gradient, absorbed by the pressure); the flag is
  // kept so the side-heated cavity can be driven by the full field.
  bool                      subtract_conductive_profile = false;
  // The temperature Dirichlet data is INHOMOGENEOUS and constant in time, so
  // its elimination is done once: g holds the boundary values, bc_correction
  // holds A_raw * g, and the factorised matrix has those rows and columns
  // replaced by the identity. Assembling the matrix and the rhs through two
  // separate distribute_local_to_global calls would silently DROP this term.
  Vector<double>            g_temp, bc_correction;

  Boussinesq()
    : fe_stokes(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1)
    , fe_temp(2)
    , dof_stokes(tria)
    , dof_temp(tria)
  {}

  // free_slip_sides: Rayleigh-Benard (u_x = 0 only on x walls);
  // otherwise every wall is no-slip, which is the side-heated cavity.
  void setup(unsigned int nx, unsigned int ny, double w, bool free_slip_sides,
             double hot_left)
  {
    width = w;
    GridGenerator::subdivided_hyper_rectangle(
      tria, {nx, ny}, Point<dim>(0.0, 0.0), Point<dim>(w, 1.0), true);
    dof_stokes.distribute_dofs(fe_stokes);
    dof_temp.distribute_dofs(fe_temp);

    const FEValuesExtractors::Vector U(0);
    std::vector<bool> only_x(dim + 1, false);
    only_x[0] = true;
    con_stokes.clear();
    for (types::boundary_id id : {2u, 3u})
      VectorTools::interpolate_boundary_values(
        dof_stokes, id, Functions::ZeroFunction<dim>(dim + 1), con_stokes,
        fe_stokes.component_mask(U));
    for (types::boundary_id id : {0u, 1u})
      VectorTools::interpolate_boundary_values(
        dof_stokes, id, Functions::ZeroFunction<dim>(dim + 1), con_stokes,
        free_slip_sides ? ComponentMask(only_x) : fe_stokes.component_mask(U));
    // The pressure is only defined up to a constant: pin the last dof.
    con_stokes.add_line(dof_stokes.n_dofs() - 1);
    con_stokes.close();

    con_temp.clear();
    VectorTools::interpolate_boundary_values(
      dof_temp, hot_left > 0.5 ? 0u : 2u,
      Functions::ConstantFunction<dim>(1.0), con_temp);
    VectorTools::interpolate_boundary_values(
      dof_temp, hot_left > 0.5 ? 1u : 3u, Functions::ZeroFunction<dim>(),
      con_temp);
    con_temp.close();

    {
      DynamicSparsityPattern dsp(dof_stokes.n_dofs());
      DoFTools::make_sparsity_pattern(dof_stokes, dsp, con_stokes, false);
      sp_stokes.copy_from(dsp);
      A_stokes.reinit(sp_stokes);
      stokes_sol.reinit(dof_stokes.n_dofs());
      stokes_rhs.reinit(dof_stokes.n_dofs());
    }
    {
      DynamicSparsityPattern dsp(dof_temp.n_dofs());
      DoFTools::make_sparsity_pattern(dof_temp, dsp, con_temp, false);
      sp_temp.copy_from(dsp);
      A_temp.reinit(sp_temp);
      M_temp.reinit(sp_temp);
      T.reinit(dof_temp.n_dofs());
      T_rhs.reinit(dof_temp.n_dofs());
    }
  }

  void assemble_stokes_matrix()
  {
    A_stokes = 0.0;
    QGauss<dim>   quad(3);
    FEValues<dim> fev(mapping, fe_stokes, quad,
                      update_values | update_gradients | update_JxW_values);
    const FEValuesExtractors::Vector U(0);
    const FEValuesExtractors::Scalar P(dim);
    const unsigned int n = fe_stokes.n_dofs_per_cell();
    FullMatrix<double> cm(n, n);
    std::vector<types::global_dof_index> local(n);
    for (const auto &cell : dof_stokes.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
              cm(i, j) += (scalar_product(fev[U].gradient(i, q),
                                          fev[U].gradient(j, q)) -
                           fev[P].value(i, q) * fev[U].divergence(j, q) -
                           fev[U].divergence(i, q) * fev[P].value(j, q)) *
                          fev.JxW(q);
        cell->get_dof_indices(local);
        con_stokes.distribute_local_to_global(cm, local, A_stokes);
      }
    for (unsigned int i = 0; i < dof_stokes.n_dofs(); ++i)
      if (con_stokes.is_constrained(i) && A_stokes.el(i, i) == 0.0)
        A_stokes.set(i, i, 1.0);
    inv_stokes.initialize(A_stokes);
  }

  void assemble_temperature_matrix()
  {
    A_temp = 0.0;
    M_temp = 0.0;
    QGauss<dim>   quad(4);
    FEValues<dim> fev(mapping, fe_temp, quad,
                      update_values | update_gradients | update_JxW_values);
    const unsigned int n = fe_temp.n_dofs_per_cell();
    FullMatrix<double> cm(n, n), cmm(n, n);
    std::vector<types::global_dof_index> local(n);
    for (const auto &cell : dof_temp.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        cmm = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
              {
                const double m =
                  fev.shape_value(i, q) * fev.shape_value(j, q) * fev.JxW(q);
                cmm(i, j) += m;
                cm(i, j) += m / dt + fev.shape_grad(i, q) *
                                       fev.shape_grad(j, q) * fev.JxW(q);
              }
        cell->get_dof_indices(local);
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            {
              A_temp.add(local[i], local[j], cm(i, j));
              M_temp.add(local[i], local[j], cmm(i, j));
            }
      }
    // Eliminate the inhomogeneous Dirichlet data once and for all.
    g_temp.reinit(dof_temp.n_dofs());
    bc_correction.reinit(dof_temp.n_dofs());
    for (unsigned int i = 0; i < dof_temp.n_dofs(); ++i)
      if (con_temp.is_constrained(i))
        g_temp(i) = con_temp.get_inhomogeneity(i);
    A_temp.vmult(bc_correction, g_temp);
    for (unsigned int r = 0; r < dof_temp.n_dofs(); ++r)
      {
        if (con_temp.is_constrained(r))
          {
            for (auto it = A_temp.begin(r); it != A_temp.end(r); ++it)
              it->value() = (it->column() == r) ? 1.0 : 0.0;
            continue;
          }
        for (auto it = A_temp.begin(r); it != A_temp.end(r); ++it)
          if (con_temp.is_constrained(it->column()))
            it->value() = 0.0;
      }
    inv_temp.initialize(A_temp);
  }

  // with_buoyancy false is the mistake of time_dependent_ns#1: momentum
  // solved "in isolation", with no coupling term at all.
  void solve_stokes(bool with_buoyancy)
  {
    stokes_rhs = 0.0;
    QGauss<dim>   quad(3);
    FEValues<dim> fs(mapping, fe_stokes, quad,
                     update_values | update_quadrature_points |
                       update_JxW_values);
    FEValues<dim> ft(mapping, fe_temp, quad, update_values);
    const FEValuesExtractors::Vector U(0);
    const unsigned int n = fe_stokes.n_dofs_per_cell();
    Vector<double>     cr(n);
    std::vector<types::global_dof_index> local(n);
    std::vector<double> tv(quad.size());
    auto ct = dof_temp.begin_active();
    for (auto cs = dof_stokes.begin_active(); cs != dof_stokes.end();
         ++cs, ++ct)
      {
        fs.reinit(cs);
        ft.reinit(ct);
        ft.get_function_values(T, tv);
        cr = 0.0;
        if (with_buoyancy)
          for (unsigned int q = 0; q < quad.size(); ++q)
            {
              const double drive =
                subtract_conductive_profile
                  ? (tv[q] - (1.0 - fs.quadrature_point(q)[1]))
                  : tv[q];
              for (unsigned int i = 0; i < n; ++i)
                cr(i) += Ra * drive * fs[U].value(i, q)[1] * fs.JxW(q);
            }
        cs->get_dof_indices(local);
        con_stokes.distribute_local_to_global(cr, local, stokes_rhs);
      }
    stokes_sol = stokes_rhs;
    inv_stokes.solve(stokes_sol);
    con_stokes.distribute(stokes_sol);
  }

  void step_temperature()
  {
    T_rhs = 0.0;
    QGauss<dim>   quad(4);
    FEValues<dim> ft(mapping, fe_temp, quad,
                     update_values | update_gradients | update_JxW_values);
    FEValues<dim> fs(mapping, fe_stokes, quad, update_values);
    const FEValuesExtractors::Vector U(0);
    const unsigned int n = fe_temp.n_dofs_per_cell();
    Vector<double>     cr(n);
    std::vector<types::global_dof_index> local(n);
    std::vector<double>         tv(quad.size());
    std::vector<Tensor<1, dim>> tg(quad.size()), uv(quad.size());
    auto cs = dof_stokes.begin_active();
    for (auto ct = dof_temp.begin_active(); ct != dof_temp.end(); ++ct, ++cs)
      {
        ft.reinit(ct);
        fs.reinit(cs);
        ft.get_function_values(T, tv);
        ft.get_function_gradients(T, tg);
        fs[U].get_function_values(stokes_sol, uv);
        cr = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < n; ++i)
            cr(i) += (tv[q] / dt - uv[q] * tg[q]) * ft.shape_value(i, q) *
                     ft.JxW(q);
        ct->get_dof_indices(local);
        for (unsigned int i = 0; i < n; ++i)
          T_rhs(local[i]) += cr(i);
      }
    T_rhs -= bc_correction;
    for (unsigned int i = 0; i < dof_temp.n_dofs(); ++i)
      if (con_temp.is_constrained(i))
        T_rhs(i) = g_temp(i);
    Vector<double> tn(T_rhs);
    inv_temp.solve(tn);
    T = tn;
  }

  double kinetic_energy() const
  {
    QGauss<dim>   quad(3);
    FEValues<dim> fs(mapping, fe_stokes, quad,
                     update_values | update_JxW_values);
    const FEValuesExtractors::Vector U(0);
    std::vector<Tensor<1, dim>>      uv(quad.size());
    double                           e = 0.0;
    for (const auto &cs : dof_stokes.active_cell_iterators())
      {
        fs.reinit(cs);
        fs[U].get_function_values(stokes_sol, uv);
        for (unsigned int q = 0; q < quad.size(); ++q)
          e += 0.5 * (uv[q] * uv[q]) * fs.JxW(q);
      }
    return e;
  }

  double max_speed() const
  {
    QGauss<dim>   quad(3);
    FEValues<dim> fs(mapping, fe_stokes, quad, update_values);
    const FEValuesExtractors::Vector U(0);
    std::vector<Tensor<1, dim>>      uv(quad.size());
    double                           m = 0.0;
    for (const auto &cs : dof_stokes.active_cell_iterators())
      {
        fs.reinit(cs);
        fs[U].get_function_values(stokes_sol, uv);
        for (unsigned int q = 0; q < quad.size(); ++q)
          m = std::max(m, uv[q].norm());
      }
    return m;
  }
};

// time_dependent_ns#1 -- the side-heated cavity with and without buoyancy.
static int buoyancy_coupling()
{
  const bool with_buoyancy = mutate(); // the mistake: momentum in isolation
  std::cout << "buoyancy_term_in_momentum=" << yesno(with_buoyancy)
            << std::endl;
  Boussinesq b;
  b.Ra = 1e4;
  b.dt = 1e-3;
  b.setup(16, 16, 1.0, false, 1.0); // hot LEFT wall, cold right, all no-slip
  b.assemble_stokes_matrix();
  b.assemble_temperature_matrix();
  // The conductive temperature field of a side-heated cavity: T = 1 - x.
  VectorTools::interpolate(b.dof_temp,
                           ScalarFunctionFromFunctionObject<dim>(
                             [](const Point<dim> &p) { return 1.0 - p[0]; }),
                           b.T);
  b.con_temp.distribute(b.T);
  std::cout << "n_stokes_dofs=" << b.dof_stokes.n_dofs()
            << " n_temperature_dofs=" << b.dof_temp.n_dofs()
            << " rayleigh=" << b.Ra << std::endl;
  std::cout << "temperature_field_range=" << *std::min_element(b.T.begin(),
                                                               b.T.end())
            << " to " << *std::max_element(b.T.begin(), b.T.end()) << std::endl;

  b.solve_stokes(with_buoyancy);
  const double ke0 = b.kinetic_energy(), sp0 = b.max_speed();
  std::cout << "first_solve_kinetic_energy=" << ke0
            << " first_solve_max_speed=" << sp0 << std::endl;
  // Ten coupled steps, so a flow that only appears through the coupling has
  // every chance to appear.
  for (unsigned int s = 0; s < 10; ++s)
    {
      b.step_temperature();
      b.solve_stokes(with_buoyancy);
    }
  const double ke = b.kinetic_energy(), sp = b.max_speed();
  std::cout << "after_ten_steps_kinetic_energy=" << ke
            << " after_ten_steps_max_speed=" << sp << std::endl;
  const bool zero_flow = sp < 1e-12;
  std::cout << "velocity_under_test_is_identically_zero=" << yesno(zero_flow)
            << std::endl;
  std::cout << "temperature_stayed_coupled_to_nothing=" << yesno(zero_flow)
            << std::endl;
  std::cout << "VERDICT="
            << (zero_flow ? "momentum_without_the_buoyancy_term_gives_no_flow"
                          : "buoyancy_term_drives_the_cavity")
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
  if (probe == "implicit_euler_energy")
    return implicit_euler_energy();
  if (probe == "buoyancy_coupling")
    return buoyancy_coupling();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
