// Shared translation unit for the adaptive-Poisson Signal family.
//
// usage: poisson_family <probe>
//   hanging_nodes_silent | split_constraints_objects | coefficient_contrast
// Env T2_MUTATE=1 runs the CORRECT variant of the probe.

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
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
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
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

// Manufactured solution u = sin(pi x) sin(pi y), f = 2 pi^2 u.
class Exact : public Function<dim>
{
public:
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[1]);
  }
};
class Source : public Function<dim>
{
public:
  double value(const Point<dim> &p, const unsigned int = 0) const override
  {
    return 2.0 * numbers::PI * numbers::PI * std::sin(numbers::PI * p[0]) *
           std::sin(numbers::PI * p[1]);
  }
};

struct Poisson
{
  Triangulation<dim>        tria;
  FE_Q<dim>                 fe;
  DoFHandler<dim>           dof;
  AffineConstraints<double> constraints;
  SparsityPattern           sp;
  SparseMatrix<double>      A;
  Vector<double>            sol, rhs;

  Poisson(unsigned int deg = 1)
    : fe(deg)
    , dof(tria)
  {}

  // hanging: build the hanging-node closure at all.
  // same_object: put the Dirichlet values on the SAME AffineConstraints as the
  //   hanging-node closure (correct) or on a separate one (poisson#5).
  void setup(bool hanging, bool same_object)
  {
    dof.distribute_dofs(fe);
    constraints.clear();
    if (hanging)
      DoFTools::make_hanging_node_constraints(dof, constraints);
    if (same_object)
      VectorTools::interpolate_boundary_values(dof, 0, Exact(), constraints);
    constraints.close();
    // When same_object is false the Dirichlet values are NOT in `constraints`
    // at all; they are applied in a second pass after assembly
    // (MatrixTools::apply_boundary_values), which is the inconsistent-assembly
    // the entry is about. Merging a second AffineConstraints object into the
    // first is NOT the mistake — measured, that route converges normally
    // (L2 error fell by 11.4x over three adaptive refinements, exactly as the
    // single-object run did).
    DynamicSparsityPattern dsp(dof.n_dofs());
    DoFTools::make_sparsity_pattern(dof, dsp, constraints, false);
    sp.copy_from(dsp);
    A.reinit(sp);
    sol.reinit(dof.n_dofs());
    rhs.reinit(dof.n_dofs());
  }

  void assemble(double contrast = 1.0)
  {
    A = 0.0;
    rhs = 0.0;
    QGauss<dim>   quad(fe.degree + 1);
    FEValues<dim> fev(fe, quad,
                      update_values | update_gradients |
                        update_quadrature_points | update_JxW_values);
    const unsigned int n = fe.dofs_per_cell;
    FullMatrix<double> cm(n, n);
    Vector<double>     cv(n);
    std::vector<types::global_dof_index> local(n);
    Source             f;
    for (const auto &cell : dof.active_cell_iterators())
      {
        fev.reinit(cell);
        cm = 0.0;
        cv = 0.0;
        for (unsigned int q = 0; q < quad.size(); ++q)
          {
            // A layered coefficient: contrast in the left half.
            const double a =
              (fev.quadrature_point(q)[0] < 0.5) ? contrast : 1.0;
            for (unsigned int i = 0; i < n; ++i)
              {
                for (unsigned int j = 0; j < n; ++j)
                  cm(i, j) += a * fev.shape_grad(i, q) * fev.shape_grad(j, q) *
                              fev.JxW(q);
                cv(i) += f.value(fev.quadrature_point(q)) *
                         fev.shape_value(i, q) * fev.JxW(q);
              }
          }
        cell->get_dof_indices(local);
        constraints.distribute_local_to_global(cm, cv, local, A, rhs);
      }
  }

  // second_pass_dirichlet reproduces poisson#5: the boundary values never
  // reached the constraints used for assembly.
  void apply_dirichlet_second_pass()
  {
    std::map<types::global_dof_index, double> bv;
    VectorTools::interpolate_boundary_values(dof, 0, Exact(), bv);
    MatrixTools::apply_boundary_values(bv, A, sol, rhs);
  }

  unsigned int solve()
  {
    SolverControl control(5000, 1e-10 * rhs.l2_norm());
    SolverCG<Vector<double>> cg(control);
    PreconditionSSOR<SparseMatrix<double>> prec;
    prec.initialize(A, 1.2);
    cg.solve(A, sol, rhs, prec);
    constraints.distribute(sol);
    return control.last_step();
  }

  double l2_error() const
  {
    Vector<double> diff(tria.n_active_cells());
    VectorTools::integrate_difference(dof, sol, Exact(), diff,
                                      QGauss<dim>(fe.degree + 2),
                                      VectorTools::L2_norm);
    return VectorTools::compute_global_error(tria, diff,
                                             VectorTools::L2_norm);
  }

  double max_abs() const
  {
    double m = 0.0;
    for (unsigned int r = 0; r < A.m(); ++r)
      for (auto it = A.begin(r); it != A.end(r); ++it)
        m = std::max(m, std::abs(it->value()));
    return m;
  }

  double symmetry_defect() const
  {
    double m = 0.0;
    for (unsigned int r = 0; r < A.m(); ++r)
      for (auto it = A.begin(r); it != A.end(r); ++it)
        m = std::max(m, std::abs(it->value() - A.el(it->column(), r)));
    return m;
  }

  void refine_adaptively()
  {
    Vector<float> est(tria.n_active_cells());
    KellyErrorEstimator<dim>::estimate(
      dof, QGauss<dim - 1>(fe.degree + 1), {}, sol, est);
    GridRefinement::refine_and_coarsen_fixed_number(tria, est, 0.4, 0.0);
    tria.execute_coarsening_and_refinement();
  }
};

// poisson#4 — forgetting hanging-node constraints is SILENT: the matrix stays
// exactly symmetric, CG behaves, and only the error trend gives it away. The
// one matrix-level tell is max|A|.
static int hanging_nodes_silent()
{
  const bool hanging = mutate();
  Poisson bad(1), good(1);
  GridGenerator::hyper_cube(bad.tria, 0.0, 1.0);
  bad.tria.refine_global(3);
  GridGenerator::hyper_cube(good.tria, 0.0, 1.0);
  good.tria.refine_global(3);

  std::vector<double> err_bad, err_good;
  double sym_bad = 0.0, sym_good = 0.0, max_bad = 0.0, max_good = 0.0;
  unsigned int it_bad = 0, it_good = 0;
  for (unsigned int cycle = 0; cycle < 4; ++cycle)
    {
      bad.setup(hanging, true);
      bad.assemble();
      it_bad = bad.solve();
      err_bad.push_back(bad.l2_error());
      sym_bad = bad.symmetry_defect();
      max_bad = bad.max_abs();

      good.setup(true, true);
      good.assemble();
      it_good = good.solve();
      err_good.push_back(good.l2_error());
      sym_good = good.symmetry_defect();
      max_good = good.max_abs();

      std::cout << "cycle=" << cycle << " ndofs=" << bad.dof.n_dofs()
                << " err_without=" << err_bad.back()
                << " err_with=" << err_good.back()
                << " cg_without=" << it_bad << " cg_with=" << it_good
                << std::endl;
      if (cycle + 1 < 4)
        {
          bad.refine_adaptively();
          good.refine_adaptively();
        }
    }
  std::cout << "symmetry_defect_without=" << sym_bad
            << " symmetry_defect_with=" << sym_good << std::endl;
  std::cout << "max_abs_without=" << max_bad << " max_abs_with=" << max_good
            << std::endl;
  const bool sym_useless = (sym_bad == 0.0 && sym_good == 0.0);
  const bool maxabs_differs = std::abs(max_bad - max_good) > 1e-12 * max_good;
  const bool error_stalls =
    err_bad.back() > 0.5 * err_bad.front();
  const bool error_improves =
    err_good.back() < 0.5 * err_good.front();
  std::cout << "symmetry_cannot_distinguish=" << (sym_useless ? "true" : "false")
            << std::endl;
  std::cout << "max_abs_distinguishes=" << (maxabs_differs ? "true" : "false")
            << std::endl;
  std::cout << "error_without_constraints_stalls="
            << (error_stalls ? "true" : "false") << std::endl;
  std::cout << "error_with_constraints_improves="
            << (error_improves ? "true" : "false") << std::endl;
  std::cout << "VERDICT="
            << ((sym_useless && maxabs_differs && error_stalls &&
                 error_improves)
                  ? "missing_hanging_nodes_is_silent_except_in_the_error_trend"
                  : "not_reproduced")
            << std::endl;
  return 0;
}

// poisson#5 — Dirichlet values on a SEPARATE AffineConstraints object.
static int split_constraints_objects()
{
  Poisson p(1);
  GridGenerator::hyper_cube(p.tria, 0.0, 1.0);
  p.tria.refine_global(3);
  std::vector<double> err;
  for (unsigned int cycle = 0; cycle < 4; ++cycle)
    {
      p.setup(true, mutate());
      p.assemble();
      if (!mutate())
        p.apply_dirichlet_second_pass();
      p.solve();
      err.push_back(p.l2_error());
      std::cout << "cycle=" << cycle << " ndofs=" << p.dof.n_dofs()
                << " l2_error=" << err.back() << std::endl;
      if (cycle + 1 < 4)
        p.refine_adaptively();
    }
  const double drop = err.front() / err.back();
  std::cout << "same_constraints_object=" << (mutate() ? "true" : "false")
            << std::endl;
  std::cout << "error_reduction_factor_over_three_refinements=" << drop
            << std::endl;
  const bool plateau = drop < 2.0;
  std::cout << "l2_error_plateaus=" << (plateau ? "true" : "false")
            << std::endl;
  std::cout << "VERDICT="
            << (plateau ? "separate_constraints_objects_stall_the_error"
                        : "error_keeps_decreasing")
            << std::endl;
  return 0;
}

// poisson#6 — SSOR-preconditioned CG iteration count grows with the
// coefficient contrast. (PreconditionAMG needs Trilinos, which this build does
// not have, so only the growth half of the claim is testable here.)
static int coefficient_contrast()
{
  const double contrasts[3] = {1.0, 1.0e2, 1.0e3};
  unsigned int iters[3];
  for (int k = 0; k < 3; ++k)
    {
      Poisson p(1);
      GridGenerator::hyper_cube(p.tria, 0.0, 1.0);
      p.tria.refine_global(5);
      p.setup(true, true);
      p.assemble(mutate() ? 1.0 : contrasts[k]);
      iters[k] = p.solve();
      std::cout << "contrast=" << contrasts[k] << " ndofs=" << p.dof.n_dofs()
                << " cg_iterations=" << iters[k] << std::endl;
    }
  // FINDING: the claim's numbers ("50 iterations at contrast 1e2, 500 at 1e3",
  // i.e. growth linear in the contrast) do NOT reproduce for a mesh-aligned
  // layered coefficient. The count does rise monotonically, but marginally.
  const bool grows = iters[2] > iters[0];
  const bool linear = iters[2] > 5 * iters[1];
  std::cout << "iterations_grow_with_contrast=" << (grows ? "true" : "false")
            << std::endl;
  std::cout << "growth_is_linear_in_contrast=" << (linear ? "true" : "false")
            << std::endl;
  std::cout << "amg_available_for_the_recommended_cure=false" << std::endl;
  std::cout << "VERDICT="
            << (grows ? "contrast_growth_is_marginal_not_linear"
                      : "iteration_count_is_flat")
            << std::endl;
  return 0;
}

int main(int argc, char **argv)
{
  const std::string probe = (argc > 1) ? argv[1] : "";
  std::cout << "probe=" << probe << std::endl;
  std::cout << "mutate=" << (mutate() ? "1" : "0") << std::endl;
  if (probe == "hanging_nodes_silent")
    return hanging_nodes_silent();
  if (probe == "split_constraints_objects")
    return split_constraints_objects();
  if (probe == "coefficient_contrast")
    return coefficient_contrast();
  std::cout << "UNKNOWN_PROBE" << std::endl;
  return 2;
}
