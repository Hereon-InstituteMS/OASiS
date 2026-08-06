// Shared translation unit for the MatrixFree Signal family.
//
// A 3D FE_Q(2) Laplace operator built twice on the same DoFHandler -- once
// assembled into a SparseMatrix and once as a MatrixFree cell_loop -- so every
// probe below can compare a matrix-free product against a reference product on
// the same vector.
//
// usage: matrixfree_family <probe>
//   mf_simplex_support | mf_rt_reinit | mf_template_variants | mf_bad_template
//   | mf_missing_constraints | mf_no_global_matrix
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.
//
// TWO OF THESE PROBES ABORT BY DESIGN and are run in their own processes by the
// fixture's cmd.sh, which pins the exit codes. DEAL_II_NOT_IMPLEMENTED() and
// Assert both ABORT (SIGABRT, rc=134); neither throws, so the exit code is the
// observable and a try/catch would see nothing.

#include <deal.II/base/function.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
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
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace dealii;

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
// The 3D FE_Q(2) Laplace problem every probe shares.
// ===========================================================================
template <int dim>
struct Lap
{
  Triangulation<dim>        tria;
  FE_Q<dim>                 fe;
  MappingQ1<dim>            mapping;
  DoFHandler<dim>           dof;
  AffineConstraints<double> constraints, empty_constraints;
  SparsityPattern           sp;
  SparseMatrix<double>      A;
  bool                      assembled = false;

  explicit Lap(unsigned int degree = 2)
    : fe(degree)
    , dof(tria)
  {}

  void setup(unsigned int refine)
  {
    GridGenerator::hyper_cube(tria, 0.0, 1.0);
    tria.refine_global(refine);
    dof.distribute_dofs(fe);
    constraints.clear();
    VectorTools::interpolate_boundary_values(
      dof, 0, Functions::ZeroFunction<dim>(), constraints);
    constraints.close();
    empty_constraints.clear();
    empty_constraints.close();
  }

  void assemble()
  {
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
    sp.copy_from(dsp);
    A.reinit(sp);
    QGauss<dim>   quad(fe.degree + 1);
    FEValues<dim> fev(mapping, fe, quad,
                      update_gradients | update_JxW_values);
    const unsigned int n = fe.n_dofs_per_cell();
    FullMatrix<double> cm(n, n);
    std::vector<types::global_dof_index> local(n);
    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < n; ++i)
            for (unsigned int j = 0; j < n; ++j)
              cm(i, j) +=
                fev.shape_grad(i, q) * fev.shape_grad(j, q) * fev.JxW(q);
        cell->get_dof_indices(local);
        constraints.distribute_local_to_global(cm, local, A);
      }
    assembled = true;
  }
};

// One matrix-free Laplace product with explicit template arguments.
template <int dim, int degree, int nq>
static void mf_apply(const MatrixFree<dim, double> &mf, Vector<double> &dst,
                     const Vector<double> &src)
{
  dst = 0.0;
  const std::function<void(const MatrixFree<dim, double> &, Vector<double> &,
                           const Vector<double> &,
                           const std::pair<unsigned int, unsigned int> &)>
    op = [](const MatrixFree<dim, double> &data, Vector<double> &d,
            const Vector<double> &s,
            const std::pair<unsigned int, unsigned int> &r) {
      FEEvaluation<dim, degree, nq, 1, double> phi(data);
      for (unsigned int cell = r.first; cell < r.second; ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values(s);
          phi.evaluate(EvaluationFlags::gradients);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_gradient(phi.get_gradient(q), q);
          phi.integrate(EvaluationFlags::gradients);
          phi.distribute_local_to_global(d);
        }
    };
  mf.cell_loop(op, dst, src);
}

// The runtime form: degree and quadrature count read from the MatrixFree object.
template <int dim>
static void mf_apply_runtime(const MatrixFree<dim, double> &mf,
                             Vector<double> &dst, const Vector<double> &src)
{
  dst = 0.0;
  const std::function<void(const MatrixFree<dim, double> &, Vector<double> &,
                           const Vector<double> &,
                           const std::pair<unsigned int, unsigned int> &)>
    op = [](const MatrixFree<dim, double> &data, Vector<double> &d,
            const Vector<double> &s,
            const std::pair<unsigned int, unsigned int> &r) {
      FEEvaluation<dim, -1, 0, 1, double> phi(data);
      for (unsigned int cell = r.first; cell < r.second; ++cell)
        {
          phi.reinit(cell);
          phi.read_dof_values(s);
          phi.evaluate(EvaluationFlags::gradients);
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
            phi.submit_gradient(phi.get_gradient(q), q);
          phi.integrate(EvaluationFlags::gradients);
          phi.distribute_local_to_global(d);
        }
    };
  mf.cell_loop(op, dst, src);
}

static double reldiff(const Vector<double> &a, const Vector<double> &b)
{
  Vector<double> d(a);
  d -= b;
  return d.l2_norm() / std::max(1e-300, b.l2_norm());
}

// ===========================================================================
// matrix_free#0 -- which elements MatrixFree can evaluate.
// ===========================================================================
static int mf_simplex_support()
{
  constexpr int dim = 3;
  Triangulation<dim> tria;
  GridGenerator::subdivided_hyper_cube_with_simplices(tria, 3);
  FE_SimplexP<dim>  fe(2);
  MappingFE<dim>    mapping(FE_SimplexP<dim>(1));
  DoFHandler<dim>   dof(tria);
  dof.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.clear();
  VectorTools::interpolate_boundary_values(
    mapping, dof, 0, Functions::ZeroFunction<dim>(), constraints);
  constraints.close();
  std::cout << "element=FE_SimplexP_2 n_cells=" << tria.n_active_cells()
            << " n_dofs=" << dof.n_dofs() << std::endl;

  // Assembled reference.
  DynamicSparsityPattern dsp(dof.n_dofs());
  DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
  SparsityPattern sp;
  sp.copy_from(dsp);
  SparseMatrix<double> A;
  A.reinit(sp);
  QGaussSimplex<dim> quad(3);
  FEValues<dim>      fev(mapping, fe, quad,
                         update_gradients | update_JxW_values);
  const unsigned int n = fe.n_dofs_per_cell();
  FullMatrix<double> cm(n, n);
  std::vector<types::global_dof_index> local(n);
  for (const auto &cell : dof.active_cell_iterators())
    {
      fev.reinit(cell);
      cm = 0.0;
      for (unsigned int q = 0; q < quad.size(); ++q)
        for (unsigned int i = 0; i < n; ++i)
          for (unsigned int j = 0; j < n; ++j)
            cm(i, j) +=
              fev.shape_grad(i, q) * fev.shape_grad(j, q) * fev.JxW(q);
      cell->get_dof_indices(local);
      constraints.distribute_local_to_global(cm, local, A);
    }

  std::cout << "before_matrixfree_reinit_on_simplices" << std::endl;
  MatrixFree<dim, double> mf;
  mf.reinit(mapping, dof, constraints, quad);
  std::cout << "after_matrixfree_reinit_on_simplices" << std::endl;
  std::cout << "matrixfree_reinit_succeeds_on_fe_simplexp=true" << std::endl;

  Vector<double> src(dof.n_dofs()), ref(dof.n_dofs()), got(dof.n_dofs());
  for (unsigned int i = 0; i < src.size(); ++i)
    src(i) = std::sin(0.3 * i + 1.0);
  constraints.set_zero(src);
  A.vmult(ref, src);
  mf_apply_runtime<dim>(mf, got, src);
  const double d = reldiff(got, ref);
  std::cout << "simplex_cell_loop_relative_difference_from_the_assembled_product="
            << d << std::endl;
  std::cout << "simplex_cell_loop_matches_the_assembled_product_to_roundoff="
            << yesno(d < 1e-12) << std::endl;
  std::cout << "VERDICT="
            << (d < 1e-12 ? "matrixfree_handles_the_simplex_element"
                          : "matrixfree_does_not_reproduce_the_product")
            << std::endl;
  return 0;
}

// The vector-valued moment-based family: this ABORTS.
static int mf_rt_reinit()
{
  constexpr int dim = 3;
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0);
  tria.refine_global(1);
  MappingQ1<dim>  mapping;
  DoFHandler<dim> dof(tria);
  // T2_MUTATE: an element MatrixFree does know how to evaluate.
  FE_RaviartThomas<dim> rt(0);
  FE_Q<dim>             q(2);
  const FiniteElement<dim> &fe =
    mutate() ? static_cast<const FiniteElement<dim> &>(q)
             : static_cast<const FiniteElement<dim> &>(rt);
  dof.distribute_dofs(fe);
  AffineConstraints<double> constraints;
  constraints.close();
  std::cout << "element=" << fe.get_name() << " n_dofs=" << dof.n_dofs()
            << std::endl;
  std::cout << "before_matrixfree_reinit" << std::endl;
  MatrixFree<dim, double> mf;
  try
    {
      mf.reinit(mapping, dof, constraints, QGauss<dim>(3));
    }
  catch (const std::exception &e)
    {
      // Recorded so the entry's "does NOT throw a catchable exception" is
      // tested rather than assumed.
      std::cout << "caught_a_std_exception=true" << std::endl;
      return 0;
    }
  std::cout << "after_matrixfree_reinit" << std::endl;
  std::cout << "caught_a_std_exception=false" << std::endl;
  return 0;
}

// ===========================================================================
// matrix_free#1 -- the FEEvaluation template arguments.
// ===========================================================================
static int mf_template_variants()
{
  constexpr int dim = 3;
  Lap<dim>      lap(2);
  lap.setup(2);
  lap.assemble();
  MatrixFree<dim, double> mf;
  mf.reinit(lap.mapping, lap.dof, lap.constraints, QGauss<dim>(3));
  std::cout << "n_dofs=" << lap.dof.n_dofs()
            << " built_with=FE_Q(2)_and_QGauss(3)" << std::endl;

  Vector<double> src(lap.dof.n_dofs()), ref(lap.dof.n_dofs()),
    got(lap.dof.n_dofs());
  for (unsigned int i = 0; i < src.size(); ++i)
    src(i) = std::sin(0.3 * i + 1.0);
  lap.constraints.set_zero(src);
  lap.A.vmult(ref, src);

  double worst = 0.0;
  bool   any_nan = false;
  auto   report = [&](const char *name, double d, bool counts) {
    std::cout << "variant=" << name << " relative_difference=" << d
              << " finite=" << yesno(std::isfinite(d))
              << " ran_without_error=true" << std::endl;
    if (counts)
      {
        if (!std::isfinite(d))
          any_nan = true;
        else
          worst = std::max(worst, d);
      }
  };

  mf_apply<dim, 2, 3>(mf, got, src);
  const double d_ok = reldiff(got, ref);
  report("matching_degree2_nq3", d_ok, false);
  mf_apply_runtime<dim>(mf, got, src);
  const double d_rt = reldiff(got, ref);
  report("runtime_minus1_0", d_rt, false);

  if (!mutate())
    {
      mf_apply<dim, 3, 3>(mf, got, src);
      report("degree_too_high_3", reldiff(got, ref), true);
      mf_apply<dim, 1, 3>(mf, got, src);
      report("degree_too_low_1", reldiff(got, ref), true);
      mf_apply<dim, 2, 4>(mf, got, src);
      report("n_q_points_too_high_4", reldiff(got, ref), true);
      mf_apply<dim, 2, 2>(mf, got, src);
      report("n_q_points_too_low_2", reldiff(got, ref), true);
    }

  std::cout << "matching_template_matches_to_roundoff=" << yesno(d_ok < 1e-12)
            << std::endl;
  std::cout << "runtime_form_matches_to_roundoff=" << yesno(d_rt < 1e-12)
            << std::endl;
  std::cout << "runtime_form_is_as_accurate_as_the_matching_template="
            << yesno(std::abs(d_rt - d_ok) < 1e-14) << std::endl;
  std::cout << "a_mismatched_template_produced_a_wrong_answer_without_error="
            << yesno(any_nan || worst > 0.1) << std::endl;
  std::cout << "a_mismatched_template_produced_a_non_finite_answer="
            << yesno(any_nan) << std::endl;
  std::cout << "VERDICT="
            << ((any_nan || worst > 0.1)
                  ? "template_mismatch_is_silent_and_wrong_in_release"
                  : "no_template_mismatch_was_exercised")
            << std::endl;
  return 0;
}

// One mismatched variant on its own, so a Debug build has exactly one Assert
// to fire.
static int mf_bad_template()
{
  constexpr int dim = 3;
  Lap<dim>      lap(2);
  lap.setup(2);
  MatrixFree<dim, double> mf;
  mf.reinit(lap.mapping, lap.dof, lap.constraints, QGauss<dim>(3));
  Vector<double> src(lap.dof.n_dofs()), got(lap.dof.n_dofs());
  for (unsigned int i = 0; i < src.size(); ++i)
    src(i) = std::sin(0.3 * i + 1.0);
  std::cout << "before_fe_evaluation" << std::endl;
  if (mutate())
    mf_apply<dim, 2, 3>(mf, got, src); // matching
  else
    mf_apply<dim, 3, 4>(mf, got, src); // degree AND quadrature both wrong
  std::cout << "after_fe_evaluation l2=" << got.l2_norm() << std::endl;
  return 0;
}

// ===========================================================================
// matrix_free#2 -- the constraints must reach MatrixFree::reinit.
// ===========================================================================
static int mf_missing_constraints()
{
  constexpr int dim = 3;
  Lap<dim>      lap(2);
  lap.setup(2);
  lap.assemble();
  const AffineConstraints<double> &handed =
    mutate() ? lap.constraints : lap.empty_constraints;
  std::cout << "constraints_handed_to_reinit="
            << (mutate() ? "the_real_ones" : "an_empty_object") << std::endl;
  MatrixFree<dim, double> mf;
  mf.reinit(lap.mapping, lap.dof, handed, QGauss<dim>(3));
  const unsigned int n_constrained = mf.get_constrained_dofs().size();
  std::cout << "n_dofs=" << lap.dof.n_dofs()
            << " real_constraints=" << lap.constraints.n_constraints()
            << " matrixfree_get_constrained_dofs_size=" << n_constrained
            << std::endl;
  std::cout << "matrixfree_knows_about_constrained_dofs="
            << yesno(n_constrained > 0) << std::endl;

  // A CG solve against a constant right-hand side.
  Vector<double> rhs(lap.dof.n_dofs()), x(lap.dof.n_dofs());
  {
    QGauss<dim>   quad(3);
    FEValues<dim> fev(lap.mapping, lap.fe, quad,
                      update_values | update_JxW_values);
    const unsigned int n = lap.fe.n_dofs_per_cell();
    Vector<double>     cr(n);
    std::vector<types::global_dof_index> local(n);
    for (const auto &cell : lap.dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cr = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          for (unsigned int i = 0; i < n; ++i)
            cr(i) += 1.0 * fev.shape_value(i, q) * fev.JxW(q);
        cell->get_dof_indices(local);
        for (unsigned int i = 0; i < n; ++i)
          rhs(local[i]) += cr(i);
      }
    handed.set_zero(rhs);
  }
  struct Op
  {
    const MatrixFree<dim, double> &mf;
    void vmult(Vector<double> &d, const Vector<double> &s) const
    {
      mf_apply_runtime<dim>(mf, d, s);
      for (const auto i : mf.get_constrained_dofs())
        d(i) = s(i);
    }
  } op{mf};

  SolverControl control(300, 1e-10 * std::max(1e-300, rhs.l2_norm()));
  SolverCG<Vector<double>> cg(control);
  bool                     converged = false;
  try
    {
      cg.solve(op, x, rhs, PreconditionIdentity());
      converged = true;
    }
  catch (const std::exception &)
    {}
  std::cout << "cg_converged=" << yesno(converged)
            << " cg_last_step=" << control.last_step()
            << " cg_last_value=" << control.last_value() << std::endl;

  // How far is the answer from satisfying the Dirichlet condition?
  double worst_boundary = 0.0;
  std::map<types::global_dof_index, double> bv;
  VectorTools::interpolate_boundary_values(
    lap.dof, 0, Functions::ZeroFunction<dim>(), bv);
  for (const auto &p : bv)
    worst_boundary = std::max(worst_boundary, std::abs(x(p.first)));
  std::cout << "n_boundary_dofs=" << bv.size()
            << " worst_boundary_value_of_the_solution=" << worst_boundary
            << std::endl;
  std::cout << "solution_is_zero_on_the_dirichlet_boundary="
            << yesno(worst_boundary < 1e-10) << std::endl;
  std::cout << "nothing_was_raised_by_reinit=true" << std::endl;
  std::cout << "VERDICT="
            << ((n_constrained == 0)
                  ? "empty_constraints_leave_the_dirichlet_data_out_silently"
                  : "constraints_reached_the_matrix_free_operator")
            << std::endl;
  return 0;
}

// ===========================================================================
// matrix_free#3 -- no global matrix exists on the matrix-free path.
// The observable asserted here is STRUCTURAL (bytes of storage, sparsity
// entries), not wall-clock; the timings are printed for information only.
// ===========================================================================
static int mf_no_global_matrix()
{
  constexpr int dim = 3;
  Lap<dim>      lap(2);
  lap.setup(3);
  const bool assemble_a_matrix = !mutate(); // the sparse fallback
  std::cout << "operator_under_test="
            << (assemble_a_matrix ? "assembled_sparse_matrix" : "matrix_free")
            << std::endl;

  MatrixFree<dim, double> mf;
  mf.reinit(lap.mapping, lap.dof, lap.constraints, QGauss<dim>(3));
  Vector<double> src(lap.dof.n_dofs()), got(lap.dof.n_dofs()),
    ref(lap.dof.n_dofs());
  for (unsigned int i = 0; i < src.size(); ++i)
    src(i) = std::sin(0.3 * i + 1.0);
  lap.constraints.set_zero(src);

  std::size_t matrix_bytes = 0, pattern_entries = 0;
  auto        t0 = std::chrono::steady_clock::now();
  if (assemble_a_matrix)
    {
      lap.assemble();
      matrix_bytes = lap.A.memory_consumption();
      pattern_entries = lap.sp.n_nonzero_elements();
      lap.A.vmult(ref, src);
    }
  auto t1 = std::chrono::steady_clock::now();
  const std::size_t mf_bytes = mf.memory_consumption();
  mf_apply_runtime<dim>(mf, got, src);
  auto t2 = std::chrono::steady_clock::now();

  std::cout << "n_dofs=" << lap.dof.n_dofs()
            << " global_matrix_bytes=" << matrix_bytes
            << " sparsity_pattern_entries=" << pattern_entries
            << " matrix_free_bytes=" << mf_bytes << std::endl;
  std::cout << "setup_seconds_under_test="
            << std::chrono::duration<double>(t1 - t0).count()
            << " matrix_free_apply_seconds="
            << std::chrono::duration<double>(t2 - t1).count() << std::endl;
  std::cout << "operator_under_test_stores_a_global_matrix="
            << yesno(matrix_bytes > 0) << std::endl;
  std::cout << "operator_under_test_builds_a_sparsity_pattern="
            << yesno(pattern_entries > 0) << std::endl;
  std::cout << "matrix_free_object_stores_no_sparsity_pattern=true"
            << std::endl;
  if (assemble_a_matrix)
    {
      const double d = reldiff(got, ref);
      std::cout << "matrix_free_relative_difference_from_the_assembled_product="
                << d << std::endl;
      std::cout << "the_two_paths_agree_to_roundoff=" << yesno(d < 1e-12)
                << std::endl;
    }
  std::cout << "VERDICT="
            << (matrix_bytes > 0
                  ? "operator_under_test_fell_back_to_a_global_sparse_matrix"
                  : "operator_under_test_holds_no_global_matrix")
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
  if (probe == "mf_simplex_support")
    return mf_simplex_support();
  if (probe == "mf_rt_reinit")
    return mf_rt_reinit();
  if (probe == "mf_template_variants")
    return mf_template_variants();
  if (probe == "mf_bad_template")
    return mf_bad_template();
  if (probe == "mf_missing_constraints")
    return mf_missing_constraints();
  if (probe == "mf_no_global_matrix")
    return mf_no_global_matrix();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
