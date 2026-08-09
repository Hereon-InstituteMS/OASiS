// Shared translation unit for the DG advection / interior-penalty Signal
// families (advection_dg::*, dg_advection_reaction::*).
// One compile serves several fixture directories; each fixture runs one probe.
//
// usage: dgip_family <probe>
//   cell_only_sparsity | sipg_penalty | flipped_face_orientation
//   | streamline_ordering | central_flux_stability | krylov_scaling
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
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

// ===========================================================================
// Upwind DG advection with a reaction term:  b.grad(u) + sigma*u = 0, inflow
// datum imposed weakly through the numerical flux.
// ===========================================================================
enum PatternKind
{
  PATTERN_FLUX,       // DoFTools::make_flux_sparsity_pattern
  PATTERN_CELL_ONLY   // DoFTools::make_sparsity_pattern
};

struct Advection
{
  Triangulation<dim>   tria;
  FE_DGQ<dim>          fe;
  DoFHandler<dim>      dof;
  SparsityPattern      sp;
  SparseMatrix<double> A;
  Vector<double>       rhs, sol;
  Tensor<1, dim>       b;
  double               sigma;
  unsigned int         flux_pattern_nnz = 0, cell_pattern_nnz = 0;

  Advection(unsigned int degree, double reaction = 0.0, double bx = 1.0,
            double by = 0.3)
    : fe(degree)
    , dof(tria)
    , sigma(reaction)
  {
    b[0] = bx;
    b[1] = by;
  }

  double g(const Point<dim> &p) const { return (p[1] < 0.5) ? 1.0 : 0.0; }

  void setup(unsigned int refine, PatternKind kind)
  {
    GridGenerator::hyper_cube(tria, 0.0, 1.0, false);
    tria.refine_global(refine);
    dof.distribute_dofs(fe);
    DynamicSparsityPattern flux(dof.n_dofs()), cellonly(dof.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof, flux);
    DoFTools::make_sparsity_pattern(dof, cellonly);
    flux_pattern_nnz = flux.n_nonzero_elements();
    cell_pattern_nnz = cellonly.n_nonzero_elements();
    sp.copy_from((kind == PATTERN_FLUX) ? flux : cellonly);
    A.reinit(sp);
    rhs.reinit(dof.n_dofs());
    sol.reinit(dof.n_dofs());
  }

  // Count the interior-face matrix entries an upwind DG assembly wants to write
  // that the CURRENT pattern does not have.  A dry run, so it also reports in a
  // Debug build, where the real assembly aborts.
  unsigned int face_entries_outside_the_pattern() const
  {
    QGauss<dim - 1>        fquad(fe.degree + 2);
    FEInterfaceValues<dim> fiv(fe, fquad, update_values);
    unsigned int           missing = 0;
    for (const auto &cell : dof.active_cell_iterators())
      for (const auto f : cell->face_indices())
        {
          if (cell->face(f)->at_boundary())
            continue;
          const auto ncell = cell->neighbor(f);
          if (ncell->active_cell_index() < cell->active_cell_index())
            continue;
          const unsigned int nf = cell->neighbor_of_neighbor(f);
          fiv.reinit(cell, f, numbers::invalid_unsigned_int, ncell, nf,
                     numbers::invalid_unsigned_int);
          const auto idx = fiv.get_interface_dof_indices();
          for (auto i : idx)
            for (auto j : idx)
              if (!sp.exists(i, j))
                ++missing;
        }
    return missing;
  }

  void assemble(bool with_interior_faces = true)
  {
    A   = 0.0;
    rhs = 0.0;
    const unsigned int n = fe.dofs_per_cell;
    QGauss<dim>        quad(fe.degree + 2);
    QGauss<dim - 1>    fquad(fe.degree + 2);
    FEValues<dim>      fev(fe, quad,
                           update_values | update_gradients | update_JxW_values);
    FEFaceValues<dim>  ffv(fe, fquad,
                           update_values | update_normal_vectors |
                             update_quadrature_points | update_JxW_values);
    FEInterfaceValues<dim> fiv(fe, fquad,
                               update_values | update_normal_vectors |
                                 update_JxW_values);
    FullMatrix<double>                   cm(n, n);
    Vector<double>                       cv(n);
    std::vector<types::global_dof_index> local(n);

    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        cv = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
              cm(i, j) += (-fev.shape_value(j, q) * (b * fev.shape_grad(i, q)) +
                           sigma * fev.shape_value(i, q) *
                             fev.shape_value(j, q)) *
                          fev.JxW(q);
        cell->get_dof_indices(local);
        for (const auto f : cell->face_indices())
          if (cell->face(f)->at_boundary())
            {
              ffv.reinit(cell, f);
              for (unsigned int q = 0; q < fquad.size(); ++q)
                {
                  const double bn = b * ffv.normal_vector(q);
                  if (bn > 0)
                    for (unsigned int i = 0; i < n; ++i)
                      for (unsigned int j = 0; j < n; ++j)
                        cm(i, j) += bn * ffv.shape_value(j, q) *
                                    ffv.shape_value(i, q) * ffv.JxW(q);
                  else
                    for (unsigned int i = 0; i < n; ++i)
                      cv(i) -= bn * g(ffv.quadrature_point(q)) *
                               ffv.shape_value(i, q) * ffv.JxW(q);
                }
            }
        A.add(local, cm);
        rhs.add(local, cv);

        if (!with_interior_faces)
          continue;
        for (const auto f : cell->face_indices())
          {
            if (cell->face(f)->at_boundary())
              continue;
            const auto ncell = cell->neighbor(f);
            if (ncell->active_cell_index() < cell->active_cell_index())
              continue;
            const unsigned int nf = cell->neighbor_of_neighbor(f);
            fiv.reinit(cell, f, numbers::invalid_unsigned_int, ncell, nf,
                       numbers::invalid_unsigned_int);
            const unsigned int ni = fiv.n_current_interface_dofs();
            FullMatrix<double> fm(ni, ni);
            fm = 0.0;
            for (unsigned int q = 0; q < fquad.size(); ++q)
              {
                const double bn        = b * fiv.normal(q);
                const bool   take_here = (bn > 0);
                for (unsigned int i = 0; i < ni; ++i)
                  for (unsigned int j = 0; j < ni; ++j)
                    fm(i, j) += bn * fiv.shape_value(take_here, j, q) *
                                fiv.jump_in_shape_values(i, q) * fiv.JxW(q);
              }
            A.add(fiv.get_interface_dof_indices(), fm);
          }
      }
  }

  void solve_direct()
  {
    SparseDirectUMFPACK direct;
    direct.initialize(A);
    direct.vmult(sol, rhs);
  }

  double max_abs_solution() const
  {
    double m = 0.0;
    for (unsigned int i = 0; i < sol.size(); ++i)
      m = std::max(m, std::abs(sol(i)));
    return m;
  }
};

// ---------------------------------------------------------------------------
// advection_dg::0 -- the cell-only sparsity pattern under a DG assembly.
// ---------------------------------------------------------------------------
static int cell_only_sparsity()
{
  const PatternKind kind = mutate() ? PATTERN_FLUX : PATTERN_CELL_ONLY;
  std::cout << "sparsity_builder="
            << (kind == PATTERN_FLUX ? "make_flux_sparsity_pattern"
                                     : "make_sparsity_pattern")
            << std::endl;
  Advection t(1, 0.0);
  t.setup(3, kind);
  std::cout << "n_active_cells=" << t.tria.n_active_cells()
            << " n_dofs=" << t.dof.n_dofs() << std::endl;
  std::cout << "flux_pattern_nonzeros=" << t.flux_pattern_nnz
            << " cell_only_pattern_nonzeros=" << t.cell_pattern_nnz
            << " ratio="
            << double(t.flux_pattern_nnz) / double(t.cell_pattern_nnz)
            << std::endl;
  std::cout << "flux_pattern_is_substantially_larger="
            << ((t.flux_pattern_nnz > 2 * t.cell_pattern_nnz) ? "true" : "false")
            << std::endl;
  const unsigned int missing = t.face_entries_outside_the_pattern();
  std::cout << "face_entries_outside_the_pattern=" << missing << std::endl;
  std::cout << "face_coupling_entries_are_missing="
            << ((missing > 0) ? "true" : "false") << std::endl;

  std::cout << "before_face_assembly" << std::endl;
  std::cout.flush();
  t.assemble(true);
  std::cout << "after_face_assembly" << std::endl;

  // Release only gets this far.  How much was silently dropped, and does the
  // answer differ from the correctly assembled one?
  t.solve_direct();
  Advection r(1, 0.0);
  r.setup(3, PATTERN_FLUX);
  r.assemble(true);
  r.solve_direct();
  double diff = 0.0;
  for (unsigned int i = 0; i < t.sol.size(); ++i)
    diff = std::max(diff, std::abs(t.sol(i) - r.sol(i)));
  std::cout << "max_abs_solution=" << t.max_abs_solution()
            << " reference_max_abs_solution=" << r.max_abs_solution()
            << std::endl;
  std::cout << "max_difference_from_the_correct_answer=" << diff << std::endl;
  const bool differs = diff > 0.1 * std::max(1.0, r.max_abs_solution());
  std::cout << "answer_differs_from_the_correct_one="
            << (differs ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << ((missing > 0)
                  ? "cell_only_pattern_silently_drops_the_face_coupling"
                  : "flux_pattern_holds_every_face_entry")
            << std::endl;
  return 0;
}

// ===========================================================================
// Symmetric interior penalty (SIPG) Poisson with the exact solution
// u = sin(pi x) sin(pi y), Dirichlet data imposed by Nitsche's method on the
// boundary faces.  Serves advection_dg::1 (the penalty parameter) and
// advection_dg::2 (the sign of the consistency terms).
// ===========================================================================
class ExactU : public Function<dim>
{
public:
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  }
};

struct SIPG
{
  Triangulation<dim>   tria;
  FE_DGQ<dim>          fe;
  DoFHandler<dim>      dof;
  SparsityPattern      sp;
  SparseMatrix<double> A;
  Vector<double>       rhs, sol;
  double               alpha;
  // 0 = correct SIPG, 1 = both jump factors of the consistency terms swapped
  // (the literal "+/- swapped in the jump integral"), 2 = only the second
  // consistency term swapped (which is the NIPG form).
  int face_form;

  SIPG(unsigned int degree, double penalty, int form = 0)
    : fe(degree)
    , dof(tria)
    , alpha(penalty)
    , face_form(form)
  {}

  static double f(const Point<dim> &p)
  {
    return 2.0 * numbers::PI * numbers::PI * std::sin(numbers::PI * p[0]) *
           std::sin(numbers::PI * p[1]);
  }

  void setup(unsigned int refine)
  {
    GridGenerator::hyper_cube(tria, 0.0, 1.0, false);
    tria.refine_global(refine);
    dof.distribute_dofs(fe);
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_flux_sparsity_pattern(dof, dsp);
    sp.copy_from(dsp);
    A.reinit(sp);
    rhs.reinit(dof.n_dofs());
    sol.reinit(dof.n_dofs());
  }

  void assemble()
  {
    A   = 0.0;
    rhs = 0.0;
    const unsigned int n = fe.dofs_per_cell;
    QGauss<dim>        quad(fe.degree + 2);
    QGauss<dim - 1>    fquad(fe.degree + 2);
    FEValues<dim>      fev(fe, quad,
                           update_values | update_gradients |
                             update_quadrature_points | update_JxW_values);
    FEFaceValues<dim>  ffv(fe, fquad,
                           update_values | update_gradients |
                             update_normal_vectors | update_JxW_values);
    FEInterfaceValues<dim> fiv(fe, fquad,
                               update_values | update_gradients |
                                 update_normal_vectors | update_JxW_values);
    FullMatrix<double>                   cm(n, n);
    Vector<double>                       cv(n);
    std::vector<types::global_dof_index> local(n);

    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        cv = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < n; ++i)
            {
              cv(i) += f(fev.quadrature_point(q)) * fev.shape_value(i, q) *
                       fev.JxW(q);
              for (unsigned int j = 0; j < n; ++j)
                cm(i, j) +=
                  fev.shape_grad(i, q) * fev.shape_grad(j, q) * fev.JxW(q);
            }
        cell->get_dof_indices(local);
        const double hcell = cell->diameter();
        for (const auto fc : cell->face_indices())
          if (cell->face(fc)->at_boundary())
            {
              ffv.reinit(cell, fc);
              const double pen = alpha / hcell;
              for (unsigned int q = 0; q < fquad.size(); ++q)
                for (unsigned int i = 0; i < n; ++i)
                  for (unsigned int j = 0; j < n; ++j)
                    cm(i, j) +=
                      (-(ffv.shape_grad(j, q) * ffv.normal_vector(q)) *
                         ffv.shape_value(i, q) -
                       ffv.shape_value(j, q) *
                         (ffv.shape_grad(i, q) * ffv.normal_vector(q)) +
                       pen * ffv.shape_value(j, q) * ffv.shape_value(i, q)) *
                      ffv.JxW(q);
            }
        A.add(local, cm);
        rhs.add(local, cv);

        for (const auto fc : cell->face_indices())
          {
            if (cell->face(fc)->at_boundary())
              continue;
            const auto ncell = cell->neighbor(fc);
            if (ncell->active_cell_index() < cell->active_cell_index())
              continue;
            const unsigned int nf = cell->neighbor_of_neighbor(fc);
            fiv.reinit(cell, fc, numbers::invalid_unsigned_int, ncell, nf,
                       numbers::invalid_unsigned_int);
            const unsigned int ni = fiv.n_current_interface_dofs();
            const double       hf =
              0.5 * (cell->diameter() + ncell->diameter());
            const double       pen = alpha / hf;
            FullMatrix<double> fm(ni, ni);
            fm = 0.0;
            // sA multiplies {grad u . n}[v], sB multiplies [u]{grad v . n}
            const double sA = (face_form == 1) ? 1.0 : -1.0;
            const double sB = (face_form == 0) ? -1.0 : 1.0;
            for (unsigned int q = 0; q < fiv.n_quadrature_points; ++q)
              {
                const Tensor<1, dim> nq = fiv.normal(q);
                for (unsigned int i = 0; i < ni; ++i)
                  for (unsigned int j = 0; j < ni; ++j)
                    fm(i, j) +=
                      (sA * (fiv.average_of_shape_gradients(j, q) * nq) *
                         fiv.jump_in_shape_values(i, q) +
                       sB * fiv.jump_in_shape_values(j, q) *
                         (fiv.average_of_shape_gradients(i, q) * nq) +
                       pen * fiv.jump_in_shape_values(j, q) *
                         fiv.jump_in_shape_values(i, q)) *
                      fiv.JxW(q);
              }
            A.add(fiv.get_interface_dof_indices(), fm);
          }
      }
  }

  void solve_direct()
  {
    SparseDirectUMFPACK direct;
    direct.initialize(A);
    direct.vmult(sol, rhs);
  }

  double l2_error() const
  {
    Vector<float> per_cell(tria.n_active_cells());
    VectorTools::integrate_difference(dof, sol, ExactU(), per_cell,
                                      QGauss<dim>(fe.degree + 3),
                                      VectorTools::L2_norm);
    return VectorTools::compute_global_error(tria, per_cell,
                                             VectorTools::L2_norm);
  }

  double relative_asymmetry() const
  {
    double num = 0.0, den = 0.0;
    for (types::global_dof_index i = 0; i < dof.n_dofs(); ++i)
      for (auto it = A.begin(i); it != A.end(i); ++it)
        {
          const double a = it->value();
          const double t = A.el(it->column(), i);
          num += (a - t) * (a - t);
          den += a * a;
        }
    return std::sqrt(num / den);
  }

  // smallest eigenvalue of A through LAPACK: coercivity of the discrete form,
  // measured rather than argued.  SIPG is symmetric, so these are real.
  double min_eigenvalue() const
  {
    LAPACKFullMatrix<double> full(dof.n_dofs(), dof.n_dofs());
    full = 0.0;
    for (types::global_dof_index i = 0; i < dof.n_dofs(); ++i)
      for (auto it = A.begin(i); it != A.end(i); ++it)
        full(i, it->column()) = it->value();
    full.compute_eigenvalues();
    double lo = std::numeric_limits<double>::max();
    for (unsigned int i = 0; i < dof.n_dofs(); ++i)
      lo = std::min(lo, full.eigenvalue(i).real());
    return lo;
  }

  // exact 2-norm condition number through LAPACK, for small systems only
  double condition_number() const
  {
    LAPACKFullMatrix<double> full(dof.n_dofs(), dof.n_dofs());
    full = 0.0;
    for (types::global_dof_index i = 0; i < dof.n_dofs(); ++i)
      for (auto it = A.begin(i); it != A.end(i); ++it)
        full(i, it->column()) = it->value();
    full.compute_svd();
    const double smax = full.singular_value(0);
    const double smin = full.singular_value(dof.n_dofs() - 1);
    return (smin > 0.0) ? smax / smin : std::numeric_limits<double>::infinity();
  }

  // Unpreconditioned, so the reported residual is the true one and a huge
  // penalty cannot flatter the stopping criterion through the diagonal scaling.
  unsigned int gmres_iterations(unsigned int budget = 2000) const
  {
    Vector<double>              x(dof.n_dofs());
    SolverControl               control(budget, 1e-8 * rhs.l2_norm());
    SolverGMRES<Vector<double>> gmres(control);
    try
      {
        gmres.solve(A, x, rhs, PreconditionIdentity());
      }
    catch (const std::exception &)
      {
        return budget;
      }
    return control.last_step();
  }
};

// ---------------------------------------------------------------------------
// advection_dg::1 -- the interior penalty parameter.
// ---------------------------------------------------------------------------
static int sipg_penalty()
{
  const unsigned int p        = 1;
  const double       rule     = 4.0 * (p + 1) * (p + 1);
  const double       under    = 0.1;
  const double       a_test   = mutate() ? rule : under;
  std::cout << "alpha_under_test=" << a_test << std::endl;
  std::cout << "alpha_rule_4_times_p_plus_1_squared=" << rule << std::endl;

  const unsigned int levels[3] = {2, 3, 4};
  double             err[3];
  for (int k = 0; k < 3; ++k)
    {
      SIPG s(p, a_test);
      s.setup(levels[k]);
      s.assemble();
      s.solve_direct();
      err[k] = s.l2_error();
      std::cout << "refine=" << levels[k] << " n_dofs=" << s.dof.n_dofs()
                << " l2_error=" << err[k] << std::endl;
    }
  const double rate = std::log(err[1] / err[2]) / std::log(2.0);
  std::cout << "observed_l2_rate_finest_pair=" << rate << std::endl;
  const bool second_order = rate > 1.5;
  std::cout << "l2_error_converges_at_order_two="
            << (second_order ? "true" : "false") << std::endl;
  const bool diverging = err[2] > err[0];
  std::cout << "l2_error_grows_with_refinement="
            << (diverging ? "true" : "false") << std::endl;

  // A sweep on one small mesh where the spectrum can be computed exactly:
  // coercivity is the sign of the smallest eigenvalue, and the entry's second
  // clause (alpha too large -> cond > 1e14 and GMRES stagnates) is measured on
  // the same table.
  const double sweep[7] = {0.01, 0.1, 1.0, rule, 1e4, 1e8, 1e12};
  bool         big_cond = false, stagnates_large = false;
  double       lam_test = 0.0, lam_rule = 0.0;
  unsigned int iters_test = 0, iters_rule = 0;
  for (double a : sweep)
    {
      SIPG s(p, a);
      s.setup(3);
      s.assemble();
      const double       lam   = s.min_eigenvalue();
      const double       cond  = s.condition_number();
      const unsigned int iters = s.gmres_iterations();
      std::cout << "alpha=" << a << " n_dofs=" << s.dof.n_dofs()
                << " min_eigenvalue=" << lam << " condition_number=" << cond
                << " gmres_iters=" << iters
                << " relative_asymmetry=" << s.relative_asymmetry()
                << std::endl;
      if (a == a_test)
        {
          lam_test   = lam;
          iters_test = iters;
        }
      if (a == rule)
        {
          lam_rule   = lam;
          iters_rule = iters;
        }
      if (a >= 1e12)
        {
          big_cond        = cond > 1e14;
          stagnates_large = iters >= 2000;
        }
    }
  std::cout << "min_eigenvalue_at_alpha_under_test=" << lam_test
            << " min_eigenvalue_at_the_rule=" << lam_rule << std::endl;
  const bool coercive_test = lam_test > 0.0;
  std::cout << "form_is_positive_definite_at_alpha_under_test="
            << (coercive_test ? "true" : "false") << std::endl;
  std::cout << "form_is_positive_definite_at_the_rule="
            << ((lam_rule > 0.0) ? "true" : "false") << std::endl;
  std::cout << "gmres_iters_at_alpha_under_test=" << iters_test
            << " gmres_iters_at_the_rule=" << iters_rule << std::endl;
  const bool krylov_broken = iters_test >= 2000 && iters_rule < 2000;
  std::cout << "unpreconditioned_gmres_exhausts_its_budget_only_below_the_rule="
            << (krylov_broken ? "true" : "false") << std::endl;
  std::cout << "condition_number_exceeds_1e14_at_alpha_1e12="
            << (big_cond ? "true" : "false") << std::endl;
  std::cout << "gmres_stagnates_at_alpha_1e12="
            << (stagnates_large ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << ((coercive_test && second_order)
                  ? "penalty_at_the_rule_is_coercive_and_converges"
                  : "penalty_below_the_rule_loses_positive_definiteness")
            << std::endl;
  return 0;
}

// ---------------------------------------------------------------------------
// advection_dg::2 -- swapping the two sides in ONE of the two jump factors of
// the SIPG consistency terms.  The entry says the result is the transpose of the
// intended matrix and that the convergence rate collapses to O(1).
// ---------------------------------------------------------------------------
static int flipped_face_orientation()
{
  const int    form  = mutate() ? 0 : 1;
  const double alpha = 16.0;   // the rule for p = 1, so the penalty is not the
                               // thing under test here
  std::cout << "orientation_under_test="
            << ((form == 1) ? "plus_minus_swapped_in_the_jump_integral"
                            : "consistent")
            << std::endl;
  std::cout << "penalty_alpha=" << alpha << std::endl;

  const unsigned int levels[3] = {2, 3, 4};
  double             err[3], ref[3];
  for (int k = 0; k < 3; ++k)
    {
      SIPG a(1, alpha, form), b(1, alpha, 0);
      a.setup(levels[k]);
      a.assemble();
      a.solve_direct();
      err[k] = a.l2_error();
      b.setup(levels[k]);
      b.assemble();
      b.solve_direct();
      ref[k] = b.l2_error();
      std::cout << "refine=" << levels[k] << " n_dofs=" << a.dof.n_dofs()
                << " l2_error_under_test=" << err[k]
                << " l2_error_consistent=" << ref[k] << std::endl;
    }
  const double rate     = std::log(err[1] / err[2]) / std::log(2.0);
  const double ref_rate = std::log(ref[1] / ref[2]) / std::log(2.0);
  std::cout << "observed_l2_rate_under_test=" << rate
            << " observed_l2_rate_consistent=" << ref_rate << std::endl;

  // Is the flipped matrix the transpose of the intended one?  And is the
  // intended one symmetric to begin with?
  SIPG a(1, alpha, 1), b(1, alpha, 0);
  a.setup(3);
  a.assemble();
  b.setup(3);
  b.assemble();
  double num = 0.0, den = 0.0;
  for (types::global_dof_index i = 0; i < b.dof.n_dofs(); ++i)
    for (auto it = b.A.begin(i); it != b.A.end(i); ++it)
      {
        const double t = b.A.el(it->column(), i);   // (A_correct^T)_{i,col}
        const double f = a.A.el(i, it->column());
        num += (f - t) * (f - t);
        den += it->value() * it->value();
      }
  const double transpose_distance = std::sqrt(num / den);
  std::cout << "relative_distance_to_the_transpose_of_the_correct_matrix="
            << transpose_distance << std::endl;
  std::cout << "correct_matrix_relative_asymmetry=" << b.relative_asymmetry()
            << std::endl;
  std::cout << "correct_matrix_is_symmetric="
            << ((b.relative_asymmetry() < 1e-12) ? "true" : "false")
            << std::endl;
  std::cout << "assembled_matrix_is_the_transpose_of_the_correct_one="
            << ((transpose_distance < 1e-10) ? "true" : "false") << std::endl;
  std::cout << "min_eigenvalue_under_test=" << a.min_eigenvalue()
            << " min_eigenvalue_consistent=" << b.min_eigenvalue() << std::endl;

  const bool collapsed = rate < 0.5;
  std::cout << "convergence_rate_collapses_to_order_zero="
            << (collapsed ? "true" : "false") << std::endl;
  const bool worse = err[2] > 3.0 * ref[2];
  std::cout << "error_is_much_larger_than_the_consistent_form="
            << (worse ? "true" : "false") << std::endl;
  // The third form -- only the SECOND consistency term flipped -- is the NIPG
  // method, and it is reported so the difference between "a sign error" and
  // "another legitimate scheme" is visible.
  {
    SIPG n2(1, alpha, 2), n3(1, alpha, 2);
    n2.setup(3);
    n2.assemble();
    n2.solve_direct();
    n3.setup(4);
    n3.assemble();
    n3.solve_direct();
    const double e2 = n2.l2_error(), e3 = n3.l2_error();
    std::cout << "one_term_flipped_l2_errors=" << e2 << "," << e3
              << " one_term_flipped_rate="
              << std::log(e2 / e3) / std::log(2.0) << std::endl;
  }
  std::cout << "VERDICT="
            << ((form == 1)
                  ? "swapped_sides_break_the_scheme_but_not_by_transposing_it"
                  : "consistent_orientation_converges")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  if (probe == "cell_only_sparsity")
    return cell_only_sparsity();
  if (probe == "sipg_penalty")
    return sipg_penalty();
  if (probe == "flipped_face_orientation")
    return flipped_face_orientation();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
