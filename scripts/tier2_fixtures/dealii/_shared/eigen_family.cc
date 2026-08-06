// Shared translation unit for the eigenvalue Signal family.
//
// The generalized Laplace eigenproblem K x = lambda M x on [0,1]^2 with zero
// Dirichlet data. Exact eigenvalues are pi^2 (m^2 + n^2): 2pi^2, 5pi^2 (double),
// 8pi^2. Solved with the catalog's own deflated inverse power iteration —
// built-in SparseMatrix + a direct solve — because this deal.II has neither
// PETSc nor SLEPc.
//
// usage: eigen_family <probe>
//   deflation_inner_product | dirichlet_spurious_modes | standard_vs_generalized
//   | constraints_on_stiffness_only | template_reference_values
//   | degenerate_pair
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.

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
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
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

struct EigenSetup
{
  Triangulation<dim>   tria;
  FE_Q<dim>            fe;
  DoFHandler<dim>      dof;
  SparsityPattern      sp;
  SparseMatrix<double> K, M;
  std::vector<types::global_dof_index> boundary;

  EigenSetup()
    : fe(1)
    , dof(tria)
  {}

  void build(unsigned int refine, double ly = 1.0)
  {
    GridGenerator::hyper_rectangle(tria, Point<dim>(0.0, 0.0),
                                   Point<dim>(1.0, ly));
    tria.refine_global(refine);
    dof.distribute_dofs(fe);
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_sparsity_pattern(dof, dsp);
    sp.copy_from(dsp);
    K.reinit(sp);
    M.reinit(sp);
    QGauss<dim>   quad(fe.degree + 2);
    FEValues<dim> fev(fe, quad,
                      update_values | update_gradients | update_JxW_values);
    const unsigned int n = fe.dofs_per_cell;
    FullMatrix<double> ck(n, n), cm(n, n);
    std::vector<types::global_dof_index> local(n);
    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        ck = 0.0;
        cm = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
              {
                ck(i, j) += fev.shape_grad(i, q) * fev.shape_grad(j, q) *
                            fev.JxW(q);
                cm(i, j) +=
                  fev.shape_value(i, q) * fev.shape_value(j, q) * fev.JxW(q);
              }
        cell->get_dof_indices(local);
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            {
              K.add(local[i], local[j], ck(i, j));
              M.add(local[i], local[j], cm(i, j));
            }
      }
    std::map<types::global_dof_index, double> bv;
    VectorTools::interpolate_boundary_values(
      dof, 0, Functions::ZeroFunction<dim>(), bv);
    for (const auto &p : bv)
      boundary.push_back(p.first);
  }

  // mass_on_boundary: what is left on M's Dirichlet rows.
  //   "tiny"     -> M_ii = 1e-12, spurious modes pushed to ~1e12 (correct)
  //   "assembly" -> M's assembled O(h^2) diagonal left in place (eigenvalue#2)
  //   "unconstrained" -> M's Dirichlet rows left entirely untouched
  //                      (eigenvalue#4: constraints applied to K only)
  void apply_dirichlet(const std::string &mass_on_boundary)
  {
    std::vector<double> m_diag(M.m(), 0.0);
    for (unsigned int r = 0; r < M.m(); ++r)
      m_diag[r] = M.el(r, r);
    for (const auto b : boundary)
      {
        for (auto it = K.begin(b); it != K.end(b); ++it)
          it->value() = 0.0;
        for (unsigned int r = 0; r < K.m(); ++r)
          if (r != b)
            for (auto it = K.begin(r); it != K.end(r); ++it)
              if (it->column() == b)
                it->value() = 0.0;
        K.set(b, b, 1.0);

        if (mass_on_boundary == "unconstrained")
          continue;
        for (auto it = M.begin(b); it != M.end(b); ++it)
          it->value() = 0.0;
        for (unsigned int r = 0; r < M.m(); ++r)
          if (r != b)
            for (auto it = M.begin(r); it != M.end(r); ++it)
              if (it->column() == b)
                it->value() = 0.0;
        if (mass_on_boundary == "tiny")
          M.set(b, b, 1e-12);
        else if (mass_on_boundary == "unit")
          M.set(b, b, 1.0);   // mirroring K_ii = 1 onto the mass matrix
        else if (mass_on_boundary == "assembly")
          M.set(b, b, m_diag[b]);   // the assembled O(h^2) diagonal, kept
      }
  }
};

// Deflated inverse power iteration for K x = lambda M x.
// m_orthogonal=false reproduces eigenvalue#1 (Euclidean deflation).
// generalized=false reproduces eigenvalue#3 (K x = lambda x, no mass matrix).
static std::vector<double> eigenvalues(EigenSetup &s, unsigned int n_modes,
                                       bool m_orthogonal, bool generalized,
                                       unsigned int iters = 300)
{
  SparseDirectUMFPACK Kinv;
  Kinv.initialize(s.K);
  std::vector<Vector<double>> found;
  std::vector<double>         lambdas;
  const unsigned int          n = s.dof.n_dofs();
  Vector<double>              x(n), y(n), Mx(n);

  auto apply_M = [&](const Vector<double> &in, Vector<double> &out) {
    if (generalized)
      s.M.vmult(out, in);
    else
      out = in;
  };

  for (unsigned int k = 0; k < n_modes; ++k)
    {
      for (unsigned int i = 0; i < n; ++i)
        x(i) = std::sin(1.0 + 0.7 * i + 3.1 * k);
      for (unsigned int it = 0; it < iters; ++it)
        {
          for (const auto &v : found)
            {
              double c;
              if (m_orthogonal)
                {
                  apply_M(v, Mx);
                  c = x * Mx;
                }
              else
                c = x * v;
              x.add(-c, v);
            }
          apply_M(x, Mx);
          Kinv.vmult(y, Mx);
          apply_M(y, Mx);
          const double nrm = std::sqrt(std::max(1e-300, y * Mx));
          y /= nrm;
          x = y;
        }
      for (const auto &v : found)
        {
          double c;
          if (m_orthogonal)
            {
              apply_M(v, Mx);
              c = x * Mx;
            }
          else
            c = x * v;
          x.add(-c, v);
        }
      apply_M(x, Mx);
      x /= std::sqrt(std::max(1e-300, x * Mx));
      Vector<double> Kx(n);
      s.K.vmult(Kx, x);
      apply_M(x, Mx);
      lambdas.push_back((x * Kx) / (x * Mx));
      found.push_back(x);
    }
  return lambdas;
}

static void print_lambdas(const std::vector<double> &l, const char *tag)
{
  for (unsigned int i = 0; i < l.size(); ++i)
    std::cout << tag << "_lambda" << i << "=" << l[i] << std::endl;
}

static EigenSetup *make(const std::string &mass_mode, unsigned int refine = 5,
                        double ly = 1.0)
{
  auto *s = new EigenSetup();
  s->build(refine, ly);
  s->apply_dirichlet(mass_mode);
  return s;
}

// eigenvalue#1 — Euclidean deflation re-converges to the previous mode.
static int deflation_inner_product()
{
  auto *s = make("tiny");
  const bool m_orth = mutate();
  auto l = eigenvalues(*s, 3, m_orth, true);
  print_lambdas(l, "run");
  const double exact0 = 2.0 * numbers::PI * numbers::PI;
  const double exact1 = 5.0 * numbers::PI * numbers::PI;
  std::cout << "m_orthogonal_deflation=" << (m_orth ? "true" : "false")
            << std::endl;
  std::cout << "exact_lambda0=" << exact0 << " exact_lambda1=" << exact1
            << std::endl;
  const bool repeated = std::abs(l[1] - l[0]) < 0.05 * exact0;
  const bool second_is_5pi2 = std::abs(l[1] - exact1) < 0.05 * exact1;
  std::cout << "second_mode_repeats_the_first=" << (repeated ? "true" : "false")
            << std::endl;
  std::cout << "second_mode_is_5pi2=" << (second_is_5pi2 ? "true" : "false")
            << std::endl;
  std::cout << "VERDICT="
            << (repeated ? "euclidean_deflation_reconverges_to_the_same_mode"
                         : "deflation_found_a_new_mode")
            << std::endl;
  return 0;
}

// eigenvalue#2 — Dirichlet dofs left with an O(1) mass diagonal put spurious
// modes right in the physical range.
static int dirichlet_spurious_modes()
{
  // The spurious pair sits at K_ii / M_ii, so WHERE it lands is arithmetic.
  // Three treatments of the Dirichlet mass diagonal, measured:
  //   1e-12  -> ~1e12, far above anything physical (the cure)
  //   1.0    -> exactly 1.0, BELOW 2pi^2 = 19.74 and right in the way
  //   O(h^2) -> ~1/h^2, which is ABOVE the first physical modes, not at O(1)
  // The claim's signal ("smallest eigenvalue ~1.0") reproduces only for the
  // mirrored M_ii = 1. Its stated cause, leaving the assembled O(h^2)
  // diagonal, does NOT put the modes at O(1) — see the printed value.
  auto *probe_asm = make("assembly");
  auto l_asm = eigenvalues(*probe_asm, 1, true, true);
  std::cout << "assembled_diagonal_smallest_lambda=" << l_asm[0] << std::endl;

  auto *s = make(mutate() ? "tiny" : "unit");
  auto l = eigenvalues(*s, 3, true, true);
  print_lambdas(l, "run");
  const double exact0 = 2.0 * numbers::PI * numbers::PI;
  std::cout << "mass_diagonal_on_dirichlet_dofs="
            << (mutate() ? "1e-12" : "1.0") << std::endl;
  std::cout << "exact_lambda0=" << exact0 << std::endl;
  const bool spurious = l[0] < 0.5 * exact0;
  const bool asm_not_at_one = l_asm[0] > 0.5 * exact0;
  std::cout << "smallest_mode_below_physical_range="
            << (spurious ? "true" : "false") << std::endl;
  std::cout << "assembled_diagonal_puts_modes_at_order_one="
            << (asm_not_at_one ? "false" : "true") << std::endl;
  std::cout << "VERDICT="
            << (spurious
                  ? "mirrored_unit_mass_diagonal_puts_a_mode_below_the_physical_range"
                  : "spectrum_starts_at_the_physical_mode")
            << std::endl;
  return 0;
}

// eigenvalue#3 — dropping the mass matrix gives mesh-dependent numbers.
static int standard_vs_generalized()
{
  double first[2][2];
  const unsigned int refines[2] = {4, 5};
  for (int r = 0; r < 2; ++r)
    {
      auto *sg = make("tiny", refines[r]);
      auto lg  = eigenvalues(*sg, 1, true, true);
      first[r][0] = lg[0];
      auto *ss = make("tiny", refines[r]);
      auto ls  = eigenvalues(*ss, 1, true, mutate());
      first[r][1] = ls[0];
      std::cout << "refine=" << refines[r]
                << " generalized_lambda0=" << first[r][0]
                << " standard_lambda0=" << first[r][1] << std::endl;
    }
  const double exact = 2.0 * numbers::PI * numbers::PI;
  const double gen_drift =
    std::abs(first[1][0] - first[0][0]) / first[0][0];
  const double std_drift =
    std::abs(first[1][1] - first[0][1]) / first[0][1];
  std::cout << "exact_lambda0=" << exact << std::endl;
  std::cout << "generalized_relative_drift=" << gen_drift << std::endl;
  std::cout << "standard_relative_drift=" << std_drift << std::endl;
  const bool gen_stable = gen_drift < 0.02;
  const bool std_moves  = std_drift > 0.5;
  std::cout << "generalized_is_mesh_independent="
            << (gen_stable ? "true" : "false") << std::endl;
  std::cout << "standard_form_scales_with_mesh="
            << (std_moves ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << ((gen_stable && std_moves)
                  ? "omitting_the_mass_matrix_gives_mesh_dependent_numbers"
                  : "both_forms_agree")
            << std::endl;
  return 0;
}

// eigenvalue#4 — constraining K but not M leaves the Dirichlet rows of M live.
static int constraints_on_stiffness_only()
{
  auto *s = make(mutate() ? "tiny" : "unconstrained");
  auto l = eigenvalues(*s, 3, true, true);
  print_lambdas(l, "run");
  const double exact0 = 2.0 * numbers::PI * numbers::PI;
  std::cout << "mass_matrix_constrained=" << (mutate() ? "true" : "false")
            << std::endl;
  std::cout << "n_boundary_dofs=" << s->boundary.size() << std::endl;
  const bool contaminated = l[0] < 0.5 * exact0;
  std::cout << "spectrum_contaminated_below_physical_range="
            << (contaminated ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << (contaminated
                  ? "constraining_only_the_stiffness_contaminates_the_spectrum"
                  : "spectrum_clean")
            << std::endl;
  return 0;
}

// eigenvalue#8 — the catalog template's own numbers, re-verified.
// eigenvalue#9 — the 5pi^2 pair must come back degenerate.
static int template_reference_values(bool degenerate_only)
{
  // Negative control for a POSITIVE claim. These two entries assert that the
  // template DOES reproduce the analytic spectrum, so the mutation has to
  // remove the property being asserted rather than inject a pitfall: on a
  // 1 x 1.05 rectangle the 5pi^2 degeneracy is a property of the SQUARE and
  // disappears, and lambda0 is no longer 2pi^2. If the fixture still passed
  // there, it would not be measuring the degeneracy at all.
  const double ly = mutate() ? 1.05 : 1.0;
  std::cout << "domain_y_extent=" << ly << std::endl;
  auto *s = make("tiny", 5, ly);
  auto l = eigenvalues(*s, 5, true, true);
  print_lambdas(l, "template");
  const double e0 = 2.0 * numbers::PI * numbers::PI;
  const double e1 = 5.0 * numbers::PI * numbers::PI;
  const double e3 = 8.0 * numbers::PI * numbers::PI;
  std::cout << "n_dofs=" << s->dof.n_dofs() << std::endl;
  std::cout << "exact_2pi2=" << e0 << " exact_5pi2=" << e1
            << " exact_8pi2=" << e3 << std::endl;
  const double err0 = std::abs(l[0] - e0) / e0;
  const double err1 = std::abs(l[1] - e1) / e1;
  const double split = std::abs(l[1] - l[2]) / e1;
  std::cout << "relative_error_mode0=" << err0
            << " relative_error_mode1=" << err1 << std::endl;
  std::cout << "degenerate_pair_relative_split=" << split << std::endl;
  const bool near0 = err0 < 0.005;
  const bool pair  = split < 1e-6;
  std::cout << "mode0_within_half_a_percent=" << (near0 ? "true" : "false")
            << std::endl;
  std::cout << "degenerate_pair_recovered=" << (pair ? "true" : "false")
            << std::endl;
  if (degenerate_only)
    std::cout << "VERDICT="
              << (pair ? "degenerate_5pi2_pair_agrees_to_machine_precision"
                       : "degenerate_pair_split")
              << std::endl;
  else
    std::cout << "VERDICT="
              << ((near0 && pair) ? "template_reproduces_the_analytic_spectrum"
                                  : "template_does_not_reproduce")
              << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  if (probe == "deflation_inner_product")
    return deflation_inner_product();
  if (probe == "dirichlet_spurious_modes")
    return dirichlet_spurious_modes();
  if (probe == "standard_vs_generalized")
    return standard_vs_generalized();
  if (probe == "constraints_on_stiffness_only")
    return constraints_on_stiffness_only();
  if (probe == "template_reference_values")
    return template_reference_values(false);
  if (probe == "degenerate_pair")
    return template_reference_values(true);
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
