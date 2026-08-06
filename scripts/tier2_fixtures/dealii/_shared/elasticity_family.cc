// Shared translation unit for the linear-elasticity Signal family.
// One compile serves several fixture directories; each fixture runs one probe.
//
// usage: elasticity_family <probe>
//   noconvergence_text | lame_swap | plane_stress | shape_value_vs_extractor
//   | symmetry_measure
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

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

struct Beam
{
  Triangulation<dim> tria;
  FESystem<dim>      fe;
  DoFHandler<dim>    dof;
  AffineConstraints<double> constraints;
  SparsityPattern    sp;
  SparseMatrix<double> A;
  Vector<double>     rhs, sol;

  Beam(unsigned int degree)
    : fe(FE_Q<dim>(degree) ^ dim)
    , dof(tria)
  {}

  void make_grid(unsigned int nx, unsigned int ny, double L, double H)
  {
    GridGenerator::subdivided_hyper_rectangle(
      tria, {nx, ny}, Point<dim>(0, 0), Point<dim>(L, H), true);
    dof.distribute_dofs(fe);
  }

  void clamp_left(bool clamp)
  {
    constraints.clear();
    if (clamp)
      VectorTools::interpolate_boundary_values(
        dof, 0, Functions::ZeroFunction<dim>(dim), constraints);
    constraints.close();
  }

  void allocate()
  {
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
    sp.copy_from(dsp);
    A.reinit(sp);
    rhs.reinit(dof.n_dofs());
    sol.reinit(dof.n_dofs());
  }

  // extractor_body_force=false reproduces the "shape_value alone" mistake.
  // asymmetric=true adds a one-sided term that breaks symmetry.
  void assemble(double mu, double lam, bool extractor_body_force,
                bool asymmetric)
  {
    A = 0.0;
    rhs = 0.0;
    QGauss<dim> quad(fe.degree + 1);
    FEValues<dim> fev(fe, quad,
                      update_values | update_gradients |
                        update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cm(dofs_per_cell, dofs_per_cell);
    Vector<double>     cv(dofs_per_cell);
    std::vector<types::global_dof_index> local(dofs_per_cell);
    const FEValuesExtractors::Vector u(0);
    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        cv = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const auto eps_i = fev[u].symmetric_gradient(i, q);
              const auto div_i = fev[u].divergence(i, q);
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const auto eps_j = fev[u].symmetric_gradient(j, q);
                  const auto div_j = fev[u].divergence(j, q);
                  cm(i, j) += (2.0 * mu * scalar_product(eps_i, eps_j) +
                               lam * div_i * div_j) *
                              fev.JxW(q);
                  if (asymmetric)
                    // A deliberately one-sided term: only the i-index carries
                    // the divergence, so A_ij != A_ji. Scaled with mu so the
                    // defect is O(max|A|) rather than a rounding-level nudge —
                    // at 0.05 it came out at 3.6e-8 relative, close enough to
                    // round-off that the diagnostic could not be trusted.
                    cm(i, j) += 2.0 * mu * div_i * fev[u].value(j, q)[0] *
                                fev.JxW(q);
                }
              // Downward body force.
              if (extractor_body_force)
                cv(i) += -1.0 * fev[u].value(i, q)[1] * fev.JxW(q);
              else
                // The mistake: one scalar shape value, no component structure.
                cv(i) += -1.0 * fev.shape_value(i, q) * fev.JxW(q);
            }
        cell->get_dof_indices(local);
        constraints.distribute_local_to_global(cm, cv, local, A, rhs);
      }
  }

  void solve_direct()
  {
    SparseDirectUMFPACK dir;
    dir.initialize(A);
    sol = rhs;
    dir.solve(sol);
    constraints.distribute(sol);
  }

  double tip_deflection() const
  {
    double worst = 0.0;
    std::vector<types::global_dof_index> local(fe.dofs_per_cell);
    for (const auto &cell : dof.active_cell_iterators())
      {
        cell->get_dof_indices(local);
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          if (fe.system_to_component_index(i).first == 1)
            worst = std::min(worst, sol(local[i]));
      }
    return worst;
  }

  double component_norm(unsigned int comp) const
  {
    double s = 0.0;
    std::vector<types::global_dof_index> local(fe.dofs_per_cell);
    for (const auto &cell : dof.active_cell_iterators())
      {
        cell->get_dof_indices(local);
        for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          if (fe.system_to_component_index(i).first == comp)
            s = std::max(s, std::abs(sol(local[i])));
      }
    return s;
  }

  double symmetry_defect() const
  {
    double worst = 0.0;
    for (unsigned int r = 0; r < A.m(); ++r)
      for (auto it = A.begin(r); it != A.end(r); ++it)
        {
          const double aij = it->value();
          const double aji = A.el(it->column(), r);
          worst = std::max(worst, std::abs(aij - aji));
        }
    return worst;
  }
};

static void lame(double E, double nu, double &mu, double &lam)
{
  mu  = E / (2.0 * (1.0 + nu));
  lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
}

// linear_elasticity#0 — there is no ExcSolverFail. A failing iterative solve
// throws SolverControl::NoConvergence, and that AssertThrow is live in Release.
static int noconvergence_text()
{
  Beam b(1);
  b.make_grid(16, 4, 4.0, 1.0);
  b.clamp_left(mutate());   // unclamped => rigid-body modes => singular
  b.allocate();
  double mu, lam;
  lame(1.0e5, 0.3, mu, lam);
  b.assemble(mu, lam, true, false);
  std::cout << "clamped=" << (mutate() ? "true" : "false") << std::endl;
  // The budget has to be generous enough that the CLAMPED problem converges,
  // otherwise the mutant fails too and the fixture proves nothing: at 50
  // iterations unpreconditioned CG had not converged even on the SPD system, so
  // removing the pathology changed nothing.
  SolverControl control(3000, 1e-10);
  SolverCG<Vector<double>> cg(control);
  PreconditionIdentity prec;
  std::string type = "none";
  std::string text = "";
  try
    {
      cg.solve(b.A, b.sol, b.rhs, prec);
      std::cout << "solver_threw=false last_step=" << control.last_step()
                << std::endl;
    }
  catch (const SolverControl::NoConvergence &e)
    {
      type = "SolverControl::NoConvergence";
      text = e.what();
      std::cout << "solver_threw=true exception_type=" << type << std::endl;
      std::cout << "last_step=" << e.last_step
                << " last_residual=" << e.last_residual << std::endl;
      std::cout << "full_message=" << text << std::endl;
    }
  catch (const std::exception &e)
    {
      type = "other";
      std::cout << "solver_threw=true exception_type=other what=" << e.what()
                << std::endl;
    }
  std::cout << "VERDICT="
            << (type == "SolverControl::NoConvergence"
                  ? "failure_is_SolverControl_NoConvergence"
                  : "no_noconvergence")
            << std::endl;
  return 0;
}

// linear_elasticity#1 — swapping mu and lambda gives a mesh-independent ratio.
static int lame_swap()
{
  double mu, lam;
  lame(1.0e5, 0.3, mu, lam);
  double ratio[2];
  const unsigned int nx[2] = {16, 32};
  for (int k = 0; k < 2; ++k)
    {
      Beam good(1), bad(1);
      good.make_grid(nx[k], nx[k] / 4, 4.0, 1.0);
      good.clamp_left(true);
      good.allocate();
      good.assemble(mu, lam, true, false);
      good.solve_direct();
      bad.make_grid(nx[k], nx[k] / 4, 4.0, 1.0);
      bad.clamp_left(true);
      bad.allocate();
      // The mistake: the two constants swapped in the bilinear form.
      bad.assemble(mutate() ? mu : lam, mutate() ? lam : mu, true, false);
      bad.solve_direct();
      ratio[k] = bad.tip_deflection() / good.tip_deflection();
      std::cout << "nx=" << nx[k] << " correct_tip=" << good.tip_deflection()
                << " swapped_tip=" << bad.tip_deflection()
                << " ratio=" << ratio[k] << std::endl;
    }
  const double drift =
    std::abs(ratio[1] - ratio[0]) / std::abs(ratio[0]);
  std::cout << "ratio_drift_between_meshes=" << drift << std::endl;
  std::cout << "ratio_is_mesh_independent=" << (drift < 0.05 ? "true" : "false")
            << std::endl;
  std::cout << "ratio_differs_from_one="
            << (std::abs(ratio[0] - 1.0) > 0.05 ? "true" : "false")
            << std::endl;
  std::cout << "VERDICT="
            << ((drift < 0.05 && std::abs(ratio[0] - 1.0) > 0.05)
                  ? "swapped_lame_is_a_mesh_independent_factor"
                  : "no_constant_factor")
            << std::endl;
  return 0;
}

// linear_elasticity#2 — plane strain is stiffer than plane stress, and the bias
// persists under refinement.
static int plane_stress()
{
  const double nu = 0.3;
  double mu, lam;
  lame(1.0e5, nu, mu, lam);
  const double lam_star = 2.0 * mu * lam / (2.0 * mu + lam);
  double ratio[2];
  const unsigned int nx[2] = {16, 32};
  for (int k = 0; k < 2; ++k)
    {
      Beam strain(2), stress(2);
      strain.make_grid(nx[k], nx[k] / 8, 4.0, 0.5);
      strain.clamp_left(true);
      strain.allocate();
      strain.assemble(mu, mutate() ? lam_star : lam, true, false);
      strain.solve_direct();
      stress.make_grid(nx[k], nx[k] / 8, 4.0, 0.5);
      stress.clamp_left(true);
      stress.allocate();
      stress.assemble(mu, lam_star, true, false);
      stress.solve_direct();
      ratio[k] = strain.tip_deflection() / stress.tip_deflection();
      std::cout << "nx=" << nx[k]
                << " plane_strain_tip=" << strain.tip_deflection()
                << " plane_stress_tip=" << stress.tip_deflection()
                << " ratio=" << ratio[k] << std::endl;
    }
  const double predicted = 1.0 - nu * nu;
  const double drift = std::abs(ratio[1] - ratio[0]);
  std::cout << "one_minus_nu_squared=" << predicted << std::endl;
  std::cout << "plane_strain_is_stiffer="
            << (ratio[0] < 0.999 ? "true" : "false") << std::endl;
  std::cout << "bias_persists_under_refinement="
            << (drift < 0.01 ? "true" : "false") << std::endl;
  std::cout << "ratio_matches_one_minus_nu_squared="
            << (std::abs(ratio[1] - predicted) < 0.03 ? "true" : "false")
            << std::endl;
  std::cout << "VERDICT="
            << ((ratio[0] < 0.999 && drift < 0.01)
                  ? "plane_strain_is_a_persistent_stiffness_bias"
                  : "no_persistent_bias")
            << std::endl;
  return 0;
}

// linear_elasticity#3 — a body force written with fe_values.shape_value alone.
static int shape_value_vs_extractor()
{
  // Both spellings, same problem, one run, so the comparison is measured and
  // not asserted from the claim's prose.
  double ux[2], uy[2];
  for (int k = 0; k < 2; ++k)
    {
      const bool use_extractor = (k == 1) || mutate();
      Beam b(1);
      b.make_grid(16, 4, 4.0, 1.0);
      b.clamp_left(true);
      b.allocate();
      double mu, lam;
      lame(1.0e5, 0.3, mu, lam);
      b.assemble(mu, lam, use_extractor, false);
      b.solve_direct();
      ux[k] = b.component_norm(0);
      uy[k] = b.component_norm(1);
    }
  std::cout << "scalar_spelling_max_abs_ux=" << ux[0]
            << " scalar_spelling_max_abs_uy=" << uy[0] << std::endl;
  std::cout << "extractor_max_abs_ux=" << ux[1]
            << " extractor_max_abs_uy=" << uy[1] << std::endl;
  const double scalar_ratio = ux[0] / uy[0];
  const double extractor_ratio = ux[1] / uy[1];
  std::cout << "scalar_ux_over_uy=" << scalar_ratio << std::endl;
  std::cout << "extractor_ux_over_uy=" << extractor_ratio << std::endl;
  // FINDING against the claim, measured here: the vertical response is
  // essentially UNCHANGED and u_y is nowhere near zero. What the scalar
  // spelling really does on a primitive FESystem is load every dof with its own
  // single non-zero component, which adds a horizontal load nobody asked for.
  // The difference is real and silent, but it lives in u_x, not u_y.
  const bool y_differs = std::abs(uy[0] - uy[1]) > 0.05 * std::abs(uy[1]);
  const bool x_differs = std::abs(ux[0] - ux[1]) > 0.05 * std::abs(ux[1]);
  std::cout << "vertical_response_differs=" << (y_differs ? "true" : "false")
            << std::endl;
  std::cout << "horizontal_response_differs=" << (x_differs ? "true" : "false")
            << std::endl;
  std::cout << "uy_identically_zero="
            << (uy[0] < 1e-14 * std::max(1.0, ux[0]) ? "true" : "false")
            << std::endl;
  std::cout << "VERDICT="
            << (x_differs
                  ? "scalar_shape_value_perturbs_the_horizontal_response_only"
                  : "spellings_agree")
            << std::endl;
  return 0;
}

// linear_elasticity#8 — measure symmetry yourself; CG never says "breakdown".
static int symmetry_measure()
{
  Beam b(1);
  b.make_grid(16, 4, 4.0, 1.0);
  b.clamp_left(true);
  b.allocate();
  double mu, lam;
  lame(1.0e5, 0.3, mu, lam);
  b.assemble(mu, lam, true, !mutate());
  const double defect = b.symmetry_defect();
  double maxabs = 0.0;
  for (unsigned int r = 0; r < b.A.m(); ++r)
    for (auto it = b.A.begin(r); it != b.A.end(r); ++it)
      maxabs = std::max(maxabs, std::abs(it->value()));
  std::cout << "asymmetric_term_present=" << (mutate() ? "false" : "true")
            << std::endl;
  std::cout << "max_abs_entry=" << maxabs << std::endl;
  std::cout << "symmetry_defect=" << defect << std::endl;
  std::cout << "relative_symmetry_defect=" << defect / maxabs << std::endl;
  const bool broken = defect / maxabs > 1e-8;
  std::cout << "symmetry_broken=" << (broken ? "true" : "false") << std::endl;

  SolverControl control(200, 1e-10);
  SolverCG<Vector<double>> cg(control);
  PreconditionIdentity prec;
  std::string outcome = "converged";
  try
    {
      cg.solve(b.A, b.sol, b.rhs, prec);
      std::cout << "cg_last_step=" << control.last_step() << std::endl;
    }
  catch (const SolverControl::NoConvergence &e)
    {
      outcome = "NoConvergence";
      std::cout << "cg_threw=SolverControl::NoConvergence last_step="
                << e.last_step << " last_residual=" << e.last_residual
                << std::endl;
    }
  std::cout << "cg_outcome=" << outcome << std::endl;
  std::cout << "VERDICT="
            << (broken ? "symmetry_defect_is_the_reliable_diagnostic"
                       : "matrix_symmetric")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  if (probe == "noconvergence_text")
    return noconvergence_text();
  if (probe == "lame_swap")
    return lame_swap();
  if (probe == "plane_stress")
    return plane_stress();
  if (probe == "shape_value_vs_extractor")
    return shape_value_vs_extractor();
  if (probe == "symmetry_measure")
    return symmetry_measure();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
