// Shared translation unit for the Stokes / saddle-point Signal family.
// One compile serves several fixture directories; each fixture runs one probe.
//
// usage: stokes_family <probe>
//   block_renumbering | pressure_nullspace | jacobi_vs_vanka
//   | equal_order_infsup | hdiv_divergence | benchmark_geometry
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparse_vanka.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <complex>
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

// velocity components -> block 0, pressure -> block 1 (step-22's grouping)
static std::vector<unsigned int> block_component()
{
  std::vector<unsigned int> bc(dim + 1, 0);
  bc[dim] = 1;
  return bc;
}

// ---------------------------------------------------------------------------
// A Taylor-Hood-shaped Stokes assembler used by several probes. The bilinear
// form is the step-22 one written so the (1,1) block is EXACTLY empty:
//   a = (grad u, grad v) - (p, div v) - (q, div u)
// ---------------------------------------------------------------------------
struct StokesAssembler
{
  const DoFHandler<dim>           &dof;
  const AffineConstraints<double> &constraints;

  StokesAssembler(const DoFHandler<dim> &d, const AffineConstraints<double> &c)
    : dof(d)
    , constraints(c)
  {}

  template <typename MatrixType, typename VectorType>
  void assemble(MatrixType &A, VectorType &rhs, bool pressure_mass_instead)
  {
    const FiniteElement<dim> &fe = dof.get_fe();
    QGauss<dim>               quad(fe.degree + 2);
    FEValues<dim>             fev(fe, quad,
                                  update_values | update_gradients |
                                    update_quadrature_points |
                                    update_JxW_values);
    const unsigned int                   n = fe.dofs_per_cell;
    FullMatrix<double>                   cm(n, n);
    Vector<double>                       cv(n);
    std::vector<types::global_dof_index> local(n);
    const FEValuesExtractors::Vector     u(0);
    const FEValuesExtractors::Scalar     p(dim);
    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        cv = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < n; ++i)
            {
              const auto gu_i  = fev[u].gradient(i, q);
              const auto div_i = fev[u].divergence(i, q);
              const auto p_i   = fev[p].value(i, q);
              for (unsigned int j = 0; j < n; ++j)
                {
                  const auto gu_j  = fev[u].gradient(j, q);
                  const auto div_j = fev[u].divergence(j, q);
                  const auto p_j   = fev[p].value(j, q);
                  if (pressure_mass_instead)
                    cm(i, j) += p_i * p_j * fev.JxW(q);
                  else
                    cm(i, j) += (scalar_product(gu_i, gu_j) - p_j * div_i -
                                 p_i * div_j) *
                                fev.JxW(q);
                }
            }
        cell->get_dof_indices(local);
        constraints.distribute_local_to_global(cm, cv, local, A, rhs);
      }
  }
};

// ---------------------------------------------------------------------------
// stokes#1 -- BlockSparseMatrix indexing needs DoFRenumbering::component_wise.
// The block SIZES are NOT the diagnostic; the (1,1) block's norm is.
// ---------------------------------------------------------------------------
static int block_renumbering()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  tria.refine_global(2);
  FESystem<dim> fe(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1);

  // --- the counts, before and after renumbering, on a scratch handler.
  {
    DoFHandler<dim> scratch(tria);
    scratch.distribute_dofs(fe);
    const auto before =
      DoFTools::count_dofs_per_fe_block(scratch, block_component());
    DoFRenumbering::component_wise(scratch, block_component());
    const auto after =
      DoFTools::count_dofs_per_fe_block(scratch, block_component());
    std::cout << "counts_before_renumbering=(" << before[0] << ","
              << before[1] << ")" << std::endl;
    std::cout << "counts_after_renumbering=(" << after[0] << "," << after[1]
              << ")" << std::endl;
    std::cout << "counts_are_identical="
              << ((before[0] == after[0] && before[1] == after[1]) ? "true"
                                                                   : "false")
              << std::endl;
  }

  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);
  if (mutate())
    DoFRenumbering::component_wise(dof, block_component());
  std::cout << "renumbered=" << (mutate() ? "true" : "false") << std::endl;

  AffineConstraints<double> constraints;
  const FEValuesExtractors::Vector velocities(0);
  VectorTools::interpolate_boundary_values(
    dof, 0, Functions::ZeroFunction<dim>(dim + 1), constraints,
    fe.component_mask(velocities));
  constraints.close();

  const std::vector<types::global_dof_index> counts =
    DoFTools::count_dofs_per_fe_block(dof, block_component());
  BlockDynamicSparsityPattern dsp(counts, counts);
  DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
  BlockSparsityPattern sp;
  sp.copy_from(dsp);
  BlockSparseMatrix<double> A;
  A.reinit(sp);
  BlockVector<double> rhs(counts);

  StokesAssembler(dof, constraints).assemble(A, rhs, false);

  const double n11 = A.block(1, 1).frobenius_norm();
  const double n00 = A.block(0, 0).frobenius_norm();
  std::cout << "velocity_block_frobenius_norm=" << n00 << std::endl;
  std::cout << "pressure_block_frobenius_norm=" << n11 << std::endl;
  const bool nonzero = n11 > 1e-10 * std::max(1.0, n00);
  std::cout << "pressure_pressure_block_is_nonzero="
            << (nonzero ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (nonzero ? "block_indexing_is_wrong_without_component_wise"
                        : "block_indexing_is_correct")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// A pinned/unpinned lid-driven cavity used by stokes#2 and stokes#3.
// ---------------------------------------------------------------------------
struct Cavity
{
  Triangulation<dim>        tria;
  FESystem<dim>             fe;
  DoFHandler<dim>           dof;
  AffineConstraints<double> constraints;
  SparsityPattern           sp;
  SparseMatrix<double>      A;
  Vector<double>            rhs;
  types::global_dof_index   n_u = 0, n_p = 0;

  Cavity()
    : fe(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1)
    , dof(tria)
  {}

  void build(unsigned int refine, bool pin_pressure)
  {
    GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
    tria.refine_global(refine);
    dof.distribute_dofs(fe);
    DoFRenumbering::component_wise(dof, block_component());
    const auto counts =
      DoFTools::count_dofs_per_fe_block(dof, block_component());
    n_u               = counts[0];
    n_p               = counts[1];

    const FEValuesExtractors::Vector velocities(0);
    constraints.clear();
    for (types::boundary_id b : {0, 1, 2})
      VectorTools::interpolate_boundary_values(
        dof, b, Functions::ZeroFunction<dim>(dim + 1), constraints,
        fe.component_mask(velocities));
    // moving lid at y = 1 (boundary id 3), tangential so the discrete
    // divergence constraint stays consistent
    std::vector<double> lid(dim + 1, 0.0);
    lid[0] = 1.0;
    VectorTools::interpolate_boundary_values(
      dof, 3, Functions::ConstantFunction<dim>(lid), constraints,
      fe.component_mask(velocities));
    if (pin_pressure)
      constraints.add_line(n_u);   // pressure dof 0 := 0
    constraints.close();

    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
    sp.copy_from(dsp);
    A.reinit(sp);
    rhs.reinit(dof.n_dofs());
    StokesAssembler(dof, constraints).assemble(A, rhs, false);
  }
};

// ---------------------------------------------------------------------------
// stokes#2 -- closed cavity: the pressure is determined only up to a constant.
// ---------------------------------------------------------------------------
static int pressure_nullspace()
{
  Cavity c;
  c.build(2, mutate());
  std::cout << "pressure_pinned=" << (mutate() ? "true" : "false")
            << std::endl;
  std::cout << "n_dofs=" << c.dof.n_dofs() << " n_u=" << c.n_u
            << " n_p=" << c.n_p << std::endl;

  // Is the constant-pressure vector in the kernel of the assembled matrix?
  Vector<double> nullvec(c.dof.n_dofs()), Anull(c.dof.n_dofs());
  for (types::global_dof_index i = c.n_u; i < c.dof.n_dofs(); ++i)
    nullvec(i) = 1.0;
  c.A.vmult(Anull, nullvec);
  const double kernel_resid = Anull.l2_norm() / nullvec.l2_norm();
  std::cout << "relative_norm_of_A_times_constant_pressure=" << kernel_resid
            << std::endl;
  const bool singular = kernel_resid < 1e-12;
  std::cout << "constant_pressure_is_in_the_kernel="
            << (singular ? "true" : "false") << std::endl;

  // What a direct solver says about it.
  std::string umf = "succeeded";
  try
    {
      SparseDirectUMFPACK direct;
      direct.initialize(c.A);
      Vector<double> x(c.rhs);
      direct.vmult(x, c.rhs);
      umf = "succeeded";
    }
  catch (const std::exception &e)
    {
      umf = std::string(e.what()).substr(0, 400);
    }
  std::cout << "umfpack_first_line=" << umf.substr(0, umf.find('\n'))
            << std::endl;

  // Two GMRES runs that differ ONLY in the starting vector.
  Vector<double> x1(c.dof.n_dofs()), x2(c.dof.n_dofs());
  for (types::global_dof_index i = c.n_u; i < c.dof.n_dofs(); ++i)
    x2(i) = 1000.0;
  unsigned int steps[2] = {0, 0};
  for (int k = 0; k < 2; ++k)
    {
      Vector<double> &x = (k == 0) ? x1 : x2;
      SolverControl   ctrl(20000, 1e-9 * std::max(1.0, c.rhs.l2_norm()));
      // full (unrestarted) GMRES: the basis may grow to the problem size, so
      // the comparison between the two runs is not a restart artefact
      SolverGMRES<Vector<double>> gmres(
        ctrl, SolverGMRES<Vector<double>>::AdditionalData(
                static_cast<unsigned int>(c.dof.n_dofs() + 2)));
      try
        {
          gmres.solve(c.A, x, c.rhs, PreconditionIdentity());
        }
      catch (const std::exception &e)
        {
          std::cout << "gmres_run" << k << "_failed" << std::endl;
        }
      steps[k] = ctrl.last_step();
      c.constraints.distribute(x);
    }
  std::cout << "gmres_steps_from_zero_start=" << steps[0]
            << " gmres_steps_from_shifted_start=" << steps[1] << std::endl;

  double m1 = 0.0, m2 = 0.0;
  for (types::global_dof_index i = c.n_u; i < c.dof.n_dofs(); ++i)
    {
      m1 += x1(i);
      m2 += x2(i);
    }
  m1 /= c.n_p;
  m2 /= c.n_p;
  double vel_gap = 0.0, shape_gap = 0.0, pmax = 0.0;
  for (types::global_dof_index i = 0; i < c.n_u; ++i)
    vel_gap = std::max(vel_gap, std::abs(x1(i) - x2(i)));
  for (types::global_dof_index i = c.n_u; i < c.dof.n_dofs(); ++i)
    {
      shape_gap = std::max(shape_gap, std::abs((x1(i) - m1) - (x2(i) - m2)));
      pmax      = std::max(pmax, std::abs(x1(i)));
    }
  std::cout << "pressure_mean_from_zero_start=" << m1
            << " pressure_mean_from_shifted_start=" << m2 << std::endl;
  std::cout << "mean_shift=" << std::abs(m1 - m2) << std::endl;
  std::cout << "max_velocity_difference=" << vel_gap << std::endl;
  std::cout << "max_pressure_shape_difference=" << shape_gap << std::endl;
  std::cout << "max_abs_pressure_from_zero_start=" << pmax << std::endl;
  // The claim's own signal says the pressure block "drifts to >1e10"; that is
  // pinned here as a measured boolean rather than assumed.
  std::cout << "pressure_exceeds_1e10=" << (pmax > 1e10 ? "true" : "false")
            << std::endl;

  const bool undetermined =
    (std::abs(m1 - m2) > 1.0) && (vel_gap < 1e-4) && (shape_gap < 1e-4);
  std::cout << "same_flow_different_pressure_level="
            << (undetermined ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (undetermined ? "pressure_is_undetermined_up_to_a_constant"
                             : "pressure_is_determined")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// stokes#3 -- point Jacobi as the relaxation on a saddle point. The pressure
// block has a ZERO diagonal, so the Jacobi division is by zero. Vanka
// (SparseVanka, the deal.II relaxation step-56 uses in its multigrid smoother)
// solves a small local saddle point around each pressure dof instead.
// ---------------------------------------------------------------------------
static int jacobi_vs_vanka()
{
  Cavity c;
  c.build(3, true);   // pinned, so the matrix itself is non-singular
  std::cout << "n_dofs=" << c.dof.n_dofs() << " n_u=" << c.n_u
            << " n_p=" << c.n_p << std::endl;

  double min_p_diag = 1e300, max_u_diag = 0.0;
  unsigned int n_zero_p_diag = 0;
  for (types::global_dof_index i = c.n_u; i < c.dof.n_dofs(); ++i)
    {
      const double d = std::abs(c.A.diag_element(i));
      min_p_diag     = std::min(min_p_diag, d);
      if (d == 0.0)
        ++n_zero_p_diag;
    }
  for (types::global_dof_index i = 0; i < c.n_u; ++i)
    max_u_diag = std::max(max_u_diag, std::abs(c.A.diag_element(i)));
  std::cout << "min_abs_pressure_diagonal=" << min_p_diag
            << " n_pressure_dofs_with_zero_diagonal=" << n_zero_p_diag
            << std::endl;
  std::cout << "max_abs_velocity_diagonal=" << max_u_diag << std::endl;

  std::vector<bool> selected(c.dof.n_dofs(), false);
  for (types::global_dof_index i = c.n_u; i < c.dof.n_dofs(); ++i)
    selected[i] = true;

  Vector<double> x(c.dof.n_dofs());
  SolverControl  ctrl(200, 1e-8 * std::max(1.0, c.rhs.l2_norm()));
  SolverGMRES<Vector<double>> gmres(ctrl);
  std::string outcome = "Converged";
  std::cout << "preconditioner=" << (mutate() ? "vanka" : "jacobi")
            << std::endl;
  try
    {
      if (mutate())
        {
          SparseVanka<double> vanka(c.A, selected);
          gmres.solve(c.A, x, c.rhs, vanka);
        }
      else
        {
          PreconditionJacobi<SparseMatrix<double>> jac;
          jac.initialize(c.A, 1.0);
          gmres.solve(c.A, x, c.rhs, jac);
        }
    }
  catch (const SolverControl::NoConvergence &e)
    {
      outcome = "NoConvergence";
    }
  catch (const std::exception &e)
    {
      outcome = "OtherException";
      std::cout << "exception_first_line="
                << std::string(e.what()).substr(
                     0, std::string(e.what()).find('\n'))
                << std::endl;
    }
  std::cout << "outcome=" << outcome << std::endl;
  std::cout << "last_step=" << ctrl.last_step()
            << " last_value=" << ctrl.last_value() << std::endl;
  const bool finite = std::isfinite(ctrl.last_value()) &&
                      std::isfinite(x.l2_norm());
  std::cout << "residual_is_finite=" << (finite ? "true" : "false")
            << std::endl;
  std::cout << "VERDICT="
            << ((!finite || outcome != "Converged")
                  ? "point_relaxation_fails_on_the_zero_pressure_diagonal"
                  : "relaxation_worked")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// stokes#4 -- inf-sup. beta^2 is the smallest NONZERO eigenvalue of the
// pressure Schur complement S = B A^{-1} B^T with respect to the pressure mass
// matrix. A stable pair keeps beta bounded away from 0 under refinement; an
// equal-order pair does not, and the extra near-null pressure modes ARE the
// checkerboard.
// ---------------------------------------------------------------------------
static void infsup_one_level(unsigned int refine, unsigned int vel_degree,
                             double &beta, unsigned int &n_null,
                             unsigned int &n_p_out, double &beta_checker)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  tria.refine_global(refine);
  FESystem<dim>   fe(FE_Q<dim>(vel_degree), dim, FE_Q<dim>(1), 1);
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);
  DoFRenumbering::component_wise(dof, block_component());
  const auto counts =
    DoFTools::count_dofs_per_fe_block(dof, block_component());
  const unsigned int n_u = counts[0], n_p = counts[1];
  n_p_out                = n_p;

  AffineConstraints<double> constraints;
  const FEValuesExtractors::Vector velocities(0);
  for (types::boundary_id b : {0, 1, 2, 3})
    VectorTools::interpolate_boundary_values(
      dof, b, Functions::ZeroFunction<dim>(dim + 1), constraints,
      fe.component_mask(velocities));
  constraints.close();

  BlockDynamicSparsityPattern dsp(counts, counts);
  DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
  BlockSparsityPattern sp;
  sp.copy_from(dsp);
  BlockSparseMatrix<double> K, M;
  K.reinit(sp);
  M.reinit(sp);
  BlockVector<double> dummy(counts);
  StokesAssembler(dof, constraints).assemble(K, dummy, false);
  dummy = 0.0;
  StokesAssembler(dof, constraints).assemble(M, dummy, true);

  SparseDirectUMFPACK Ainv;
  Ainv.initialize(K.block(0, 0));

  FullMatrix<double> S(n_p, n_p);
  Vector<double>     e(n_p), t(n_u), w(n_u), col(n_p);
  for (unsigned int j = 0; j < n_p; ++j)
    {
      e    = 0.0;
      e(j) = 1.0;
      K.block(0, 1).vmult(t, e);
      Ainv.vmult(w, t);
      K.block(1, 0).vmult(col, w);
      for (unsigned int i = 0; i < n_p; ++i)
        S(i, j) = col(i);
    }
  FullMatrix<double> Mp(n_p, n_p);
  Mp.copy_from(M.block(1, 1));

  // The claim's own Signal is a CHECKERBOARD pressure. Build that vector
  // explicitly from the pressure support points and take its Rayleigh
  // quotient in the Schur complement: if it is a spurious mode, the quotient
  // collapses to zero.
  {
    std::map<types::global_dof_index, Point<dim>> support;
    DoFTools::map_dofs_to_support_points(MappingQ1<dim>(), dof, support);
    const double h = 1.0 / (1u << refine);
    Vector<double> c(n_p);
    for (unsigned int i = 0; i < n_p; ++i)
      {
        const Point<dim> &pt = support[n_u + i];
        const int ix = static_cast<int>(std::lround(pt[0] / h));
        const int iy = static_cast<int>(std::lround(pt[1] / h));
        c(i) = ((ix + iy) % 2 == 0) ? 1.0 : -1.0;
      }
    Vector<double> Sc(n_p), Mc(n_p);
    S.vmult(Sc, c);
    Mp.vmult(Mc, c);
    beta_checker = std::sqrt(std::max(0.0, (c * Sc) / (c * Mc)));
  }
  Mp.gauss_jordan();
  FullMatrix<double> G(n_p, n_p);
  Mp.mmult(G, S);

  LAPACKFullMatrix<double> L(n_p, n_p);
  L = G;
  L.compute_eigenvalues();
  std::vector<double> ev(n_p);
  for (unsigned int i = 0; i < n_p; ++i)
    ev[i] = L.eigenvalue(i).real();
  std::sort(ev.begin(), ev.end());
  const double lmax = ev.back();
  n_null            = 0;
  for (unsigned int i = 0; i < n_p; ++i)
    if (ev[i] < 1e-8 * lmax)
      ++n_null;
  // the constant pressure mode is always one of them; beta^2 is the first
  // eigenvalue above that numerical-null cluster
  beta = std::sqrt(std::max(0.0, ev[std::min<unsigned int>(n_null, n_p - 1)]));
  std::cout << "  level=" << refine << " n_p=" << n_p
            << " lambda_1=" << ev[0] << " lambda_2=" << ev[1]
            << " lambda_3=" << ev[2] << " lambda_max=" << lmax << std::endl;
}

static int equal_order_infsup()
{
  const unsigned int deg = mutate() ? 2u : 1u;
  std::cout << "velocity_degree=" << deg << " pressure_degree=1" << std::endl;
  std::cout << "pair=" << (deg == 2 ? "Taylor_Hood_Q2_Q1" : "equal_order_Q1_Q1")
            << std::endl;
  double       beta[3], bchk[3];
  unsigned int nn[3], np[3];
  for (unsigned int k = 0; k < 3; ++k)
    infsup_one_level(2 + k, deg, beta[k], nn[k], np[k], bchk[k]);
  for (unsigned int k = 0; k < 3; ++k)
    std::cout << "refine=" << (2 + k) << " n_pressure_dofs=" << np[k]
              << " n_numerically_null_pressure_modes=" << nn[k]
              << " inf_sup_beta=" << beta[k]
              << " checkerboard_rayleigh_beta=" << bchk[k] << std::endl;
  const bool checker_spurious = bchk[2] < 1e-6;
  std::cout << "checkerboard_pressure_is_a_spurious_mode="
            << (checker_spurious ? "true" : "false") << std::endl;
  const bool extra_modes = (nn[0] > 1 || nn[1] > 1 || nn[2] > 1);
  const bool beta_collapses = beta[2] < 0.5 * beta[0];
  std::cout << "more_null_modes_than_the_constant=" << (extra_modes ? "true"
                                                                    : "false")
            << std::endl;
  std::cout << "beta_halves_under_refinement="
            << (beta_collapses ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << ((extra_modes || beta_collapses)
                  ? "pair_is_not_inf_sup_stable"
                  : "pair_is_inf_sup_stable")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// stokes#5 -- H(div)-conforming velocity gives a POINTWISE divergence-free
// discrete velocity. Measured on the mixed saddle point that carries exactly
// the discrete incompressibility constraint of Stokes,
//    (u, v) + (p, div v) = (g, v)   for all v
//    (div u, q)          = 0        for all q
// with RT_k/DGQ_k (div V_h is contained in Q_h) against continuous Q2/Q1
// (it is not).
// ---------------------------------------------------------------------------
// The exact solution is a SMOOTH, EXACTLY divergence-free velocity
//   u = curl psi,  psi = sin(pi x) sin(pi y) / pi
// together with a pressure that vanishes on the boundary (the natural
// condition of this mixed form), so both discretisations approximate the same
// divergence-free field and the only question is how nearly divergence free
// their own answer is.
static Tensor<1, dim> u_exact(const Point<dim> &p)
{
  const double pi = numbers::PI;
  Tensor<1, dim> u;
  u[0] = std::sin(pi * p[0]) * std::cos(pi * p[1]);
  u[1] = -std::cos(pi * p[0]) * std::sin(pi * p[1]);
  return u;
}

class Source : public Function<dim>
{
public:
  Source()
    : Function<dim>(dim + 1)
  {}
  void vector_value(const Point<dim> &p, Vector<double> &v) const override
  {
    // g = u_exact - grad p_exact, p_exact = sin(pi x) sin(pi y)
    const double   pi = numbers::PI;
    Tensor<1, dim> u  = u_exact(p);
    v                 = 0.0;
    v[0] = u[0] - pi * std::cos(pi * p[0]) * std::sin(pi * p[1]);
    v[1] = u[1] - pi * std::sin(pi * p[0]) * std::cos(pi * p[1]);
  }
};

static int hdiv_divergence()
{
  const bool use_rt = !mutate();
  std::cout << "velocity_space=" << (use_rt ? "FE_RaviartThomas_1"
                                            : "FE_Q_2_vector")
            << std::endl;
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0, false);
  tria.refine_global(4);

  std::unique_ptr<FESystem<dim>> fe;
  if (use_rt)
    fe = std::make_unique<FESystem<dim>>(FE_RaviartThomas<dim>(1), 1,
                                         FE_DGQ<dim>(1), 1);
  else
    fe = std::make_unique<FESystem<dim>>(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1);

  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(*fe);
  std::cout << "n_dofs=" << dof.n_dofs() << std::endl;
  AffineConstraints<double> constraints;
  constraints.close();

  DynamicSparsityPattern dsp(dof.n_dofs());
  DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A(sp);
  Vector<double>       rhs(dof.n_dofs()), sol(dof.n_dofs());

  QGauss<dim>   quad(fe->degree + 3);
  FEValues<dim> fev(*fe, quad,
                    update_values | update_gradients |
                      update_quadrature_points | update_JxW_values);
  const unsigned int                   n = fe->dofs_per_cell;
  FullMatrix<double>                   cm(n, n);
  Vector<double>                       cv(n);
  std::vector<types::global_dof_index> local(n);
  const FEValuesExtractors::Vector     u(0);
  const FEValuesExtractors::Scalar     p(dim);
  Source                               g;
  Vector<double>                       gv(dim + 1);
  for (const auto &cell : dof.active_cell_iterators())
    {
      fev.reinit(cell);
      cm = 0.0;
      cv = 0.0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        {
          g.vector_value(fev.quadrature_point(q), gv);
          for (unsigned int i = 0; i < n; ++i)
            {
              const auto u_i   = fev[u].value(i, q);
              const auto div_i = fev[u].divergence(i, q);
              const auto p_i   = fev[p].value(i, q);
              for (unsigned int j = 0; j < n; ++j)
                {
                  const auto u_j   = fev[u].value(j, q);
                  const auto div_j = fev[u].divergence(j, q);
                  const auto p_j   = fev[p].value(j, q);
                  cm(i, j) +=
                    (u_i * u_j + p_j * div_i + p_i * div_j) * fev.JxW(q);
                }
              cv(i) += (u_i[0] * gv[0] + u_i[1] * gv[1]) * fev.JxW(q);
            }
        }
      cell->get_dof_indices(local);
      constraints.distribute_local_to_global(cm, cv, local, A, rhs);
    }

  SparseDirectUMFPACK direct;
  direct.initialize(A);
  direct.vmult(sol, rhs);

  // ||div u_h||_L2 and ||u_h||_L2
  double div2 = 0.0, u2 = 0.0, divmax = 0.0, err2 = 0.0;
  FEValues<dim> fev2(*fe, quad,
                     update_values | update_gradients |
                       update_quadrature_points | update_JxW_values);
  std::vector<double>       divs(quad.size());
  std::vector<Tensor<1, dim>> vals(quad.size());
  for (const auto &cell : dof.active_cell_iterators())
    {
      fev2.reinit(cell);
      fev2[u].get_function_divergences(sol, divs);
      fev2[u].get_function_values(sol, vals);
      for (unsigned int q = 0; q < quad.size(); ++q)
        {
          div2 += divs[q] * divs[q] * fev2.JxW(q);
          u2 += (vals[q] * vals[q]) * fev2.JxW(q);
          divmax = std::max(divmax, std::abs(divs[q]));
          const Tensor<1, dim> d =
            vals[q] - u_exact(fev2.quadrature_point(q));
          err2 += (d * d) * fev2.JxW(q);
        }
    }
  const double divn = std::sqrt(div2), un = std::sqrt(u2);
  std::cout << "l2_error_of_velocity_vs_exact=" << std::sqrt(err2)
            << std::endl;
  std::cout << "l2_norm_of_velocity=" << un << std::endl;
  std::cout << "l2_norm_of_divergence=" << divn << std::endl;
  std::cout << "max_pointwise_divergence=" << divmax << std::endl;
  const double rel = divn / un;
  std::cout << "relative_divergence=" << rel << std::endl;
  const bool machine_eps = rel < 1e-12;
  std::cout << "divergence_at_machine_epsilon="
            << (machine_eps ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (machine_eps ? "hdiv_pair_is_pointwise_divergence_free"
                            : "divergence_is_only_weakly_zero")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// stokes#6 -- the Schaefer-Turek geometry.
// ---------------------------------------------------------------------------
struct Inflow : public Function<dim>
{
  const double H, Um;
  Inflow(double h, double um)
    : Function<dim>(dim + 1)
    , H(h)
    , Um(um)
  {}
  void vector_value(const Point<dim> &p, Vector<double> &v) const override
  {
    v    = 0.0;
    v[0] = 4.0 * Um * p[1] * (H - p[1]) / (H * H);
  }
};

// Stokes flow past the obstacle; returns the drag coefficient in the
// Schaefer-Turek normalisation C_D = 2 F_x / (rho Ubar^2 D).
static double cylinder_drag(Triangulation<dim> &tria, double H, double &ymin,
                            double &ymax, double &cx, double &cy, double &rad,
                            double &xmin, double &xmax,
                            unsigned int &n_cells)
{
  n_cells = tria.n_active_cells();
  xmin = ymin = 1e300;
  xmax = ymax = -1e300;
  double cxmin = 1e300, cxmax = -1e300, cymin = 1e300, cymax = -1e300;
  for (const auto &cell : tria.active_cell_iterators())
    for (const auto v : cell->vertex_indices())
      {
        const Point<dim> &pt = cell->vertex(v);
        xmin                 = std::min(xmin, pt[0]);
        xmax                 = std::max(xmax, pt[0]);
        ymin                 = std::min(ymin, pt[1]);
        ymax                 = std::max(ymax, pt[1]);
      }
  for (const auto &cell : tria.active_cell_iterators())
    for (const auto f : cell->face_indices())
      if (cell->face(f)->at_boundary() &&
          cell->face(f)->boundary_id() == 2)
        for (const auto v : cell->face(f)->vertex_indices())
          {
            const Point<dim> &pt = cell->face(f)->vertex(v);
            cxmin                = std::min(cxmin, pt[0]);
            cxmax                = std::max(cxmax, pt[0]);
            cymin                = std::min(cymin, pt[1]);
            cymax                = std::max(cymax, pt[1]);
          }
  cx  = 0.5 * (cxmin + cxmax);
  cy  = 0.5 * (cymin + cymax);
  rad = 0.5 * (cxmax - cxmin);

  const double nu = 0.001, Um = 0.3, D = 2.0 * rad;
  const double Ubar = 2.0 / 3.0 * Um;

  FESystem<dim>   fe(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1);
  DoFHandler<dim> dof(tria);
  dof.distribute_dofs(fe);
  DoFRenumbering::component_wise(dof, block_component());
  AffineConstraints<double>        constraints;
  const FEValuesExtractors::Vector velocities(0);
  VectorTools::interpolate_boundary_values(dof, 0, Inflow(H, Um), constraints,
                                           fe.component_mask(velocities));
  for (types::boundary_id b : {2, 3, 4})
    VectorTools::interpolate_boundary_values(
      dof, b, Functions::ZeroFunction<dim>(dim + 1), constraints,
      fe.component_mask(velocities));
  constraints.close();

  DynamicSparsityPattern dsp(dof.n_dofs());
  DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A(sp);
  Vector<double>       rhs(dof.n_dofs()), sol(dof.n_dofs());

  // (nu grad u, grad v) - (p, div v) - (q, div u)
  QGauss<dim>   quad(4);
  FEValues<dim> fev(fe, quad,
                    update_values | update_gradients | update_JxW_values);
  const unsigned int                   n = fe.dofs_per_cell;
  FullMatrix<double>                   cm(n, n);
  Vector<double>                       cv(n);
  std::vector<types::global_dof_index> local(n);
  const FEValuesExtractors::Vector     u(0);
  const FEValuesExtractors::Scalar     pr(dim);
  for (const auto &cell : dof.active_cell_iterators())
    {
      fev.reinit(cell);
      cm = 0.0;
      cv = 0.0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < n; ++i)
          {
            const auto gu_i  = fev[u].gradient(i, q);
            const auto div_i = fev[u].divergence(i, q);
            const auto p_i   = fev[pr].value(i, q);
            for (unsigned int j = 0; j < n; ++j)
              {
                const auto gu_j  = fev[u].gradient(j, q);
                const auto div_j = fev[u].divergence(j, q);
                const auto p_j   = fev[pr].value(j, q);
                cm(i, j) += (nu * scalar_product(gu_i, gu_j) - p_j * div_i -
                             p_i * div_j) *
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

  // traction integral over the obstacle
  QGauss<dim - 1>   fquad(4);
  FEFaceValues<dim> ffv(fe, fquad,
                        update_values | update_gradients |
                          update_normal_vectors | update_JxW_values);
  std::vector<Tensor<2, dim>> grad_u(fquad.size());
  std::vector<double>         pval(fquad.size());
  double                      Fx = 0.0;
  for (const auto &cell : dof.active_cell_iterators())
    for (const auto f : cell->face_indices())
      if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == 2)
        {
          ffv.reinit(cell, f);
          ffv[u].get_function_gradients(sol, grad_u);
          ffv[pr].get_function_values(sol, pval);
          for (unsigned int q = 0; q < fquad.size(); ++q)
            {
              const Tensor<1, dim> nrm = ffv.normal_vector(q);
              // traction t = (nu grad u - p I) . n, normal points OUT of the
              // fluid domain, i.e. INTO the obstacle
              Tensor<1, dim> t;
              for (unsigned int a = 0; a < dim; ++a)
                {
                  double s = 0.0;
                  for (unsigned int b = 0; b < dim; ++b)
                    s += nu * grad_u[q][a][b] * nrm[b];
                  t[a] = s - pval[q] * nrm[a];
                }
              Fx -= t[0] * ffv.JxW(q);
            }
        }
  return 2.0 * Fx / (Ubar * Ubar * D);
}

static int benchmark_geometry()
{
  double cd[2][2];
  for (int g = 0; g < 2; ++g)
    {
      const bool official = (g == 1) || mutate();
      for (int lvl = 0; lvl < 2; ++lvl)
        {
          Triangulation<dim> tria;
          double             H;
          if (official)
            {
              GridGenerator::channel_with_cylinder(tria, 0.03, 2, 2.0, true);
              tria.refine_global(lvl);
              H = 0.41;
            }
          else
            {
              // the "generalised" generator: the cylinder diameter is fixed to
              // one and the channel extents must be INTEGER multiples of it,
              // so 2.2 x 0.41 with the cylinder at (0.2, 0.2) cannot be
              // expressed; the closest is a SYMMETRIC 2.2 x 0.40 channel.
              GridGenerator::uniform_channel_with_cylinder(
                tria, {2u, 20u, 2u, 2u}, 1.0, 1, 0.75, 2, 2.0, false, true);
              tria.refine_global(lvl);
              GridTools::scale(0.1, tria);
              GridTools::shift(Point<dim>(0.2, 0.2), tria);
              H = 0.40;
            }
          tria.reset_all_manifolds();
          double       ymin, ymax, cx, cy, rad, xmin, xmax;
          unsigned int nc;
          cd[g][lvl] =
            cylinder_drag(tria, H, ymin, ymax, cx, cy, rad, xmin, xmax, nc);
          std::cout << (official ? "official" : "offbyone") << "_level=" << lvl
                    << " n_cells=" << nc << " x=[" << xmin << "," << xmax
                    << "] y=[" << ymin << "," << ymax << "] cylinder_centre=("
                    << cx << "," << cy << ") diameter=" << 2 * rad
                    << " C_D=" << cd[g][lvl] << std::endl;
          if (lvl == 1)
            {
              std::cout << (official ? "official" : "offbyone")
                        << "_channel_height=" << (ymax - ymin) << std::endl;
              std::cout << (official ? "official" : "offbyone")
                        << "_cylinder_offset_from_centreline="
                        << std::abs(cy - 0.5 * (ymin + ymax)) << std::endl;
            }
        }
    }
  const double rel = std::abs(cd[0][1] - cd[1][1]) /
                     std::max(1e-30, std::abs(cd[1][1]));
  const double rel_coarse = std::abs(cd[0][0] - cd[1][0]) /
                            std::max(1e-30, std::abs(cd[1][0]));
  std::cout << "reference_schaefer_turek_C_D_Re20=5.58 (unsteady "
               "Navier-Stokes; the Stokes value computed here is not it)"
            << std::endl;
  std::cout << "relative_C_D_difference_coarse=" << rel_coarse
            << " relative_C_D_difference_fine=" << rel << std::endl;
  const bool systematic = rel > 0.02 && rel_coarse > 0.02;
  std::cout << "difference_exceeds_10_percent=" << (rel > 0.10 ? "true"
                                                               : "false")
            << std::endl;
  std::cout << "difference_survives_refinement="
            << (systematic ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (systematic ? "wrong_channel_geometry_shifts_the_drag"
                           : "geometries_agree")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  if (probe == "block_renumbering")
    return block_renumbering();
  if (probe == "pressure_nullspace")
    return pressure_nullspace();
  if (probe == "jacobi_vs_vanka")
    return jacobi_vs_vanka();
  if (probe == "equal_order_infsup")
    return equal_order_infsup();
  if (probe == "hdiv_divergence")
    return hdiv_divergence();
  if (probe == "benchmark_geometry")
    return benchmark_geometry();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
