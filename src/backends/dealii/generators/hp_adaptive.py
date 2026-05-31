"""hp-adaptive FEM templates for deal.II.

Based on deal.II tutorial step-27 (hp-adaptive with smoothness estimation).
"""


def _hp_adaptive_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.
    All parameter defaults are placeholders.
    hp-adaptive FEM with smoothness estimation — based on step-27 pattern.
    """
    max_degree = params.get("max_degree", 7)
    min_degree = params.get("min_degree", 1)
    n_cycles = params.get("n_cycles", 6)
    refinements = params.get("refinements", 2)
    rhs_value = params.get("rhs_value", 1.0)
    return f'''\
/* hp-adaptive FEM on unit square — based on deal.II step-27 pattern
 * Solves -laplacian(u) = f with hp-adaptivity using smoothness estimation.
 * Higher polynomial degree in smooth regions, h-refinement near singularities.
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_series.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/refinement.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/smoothness_estimator.h>
#include <fstream>
#include <iostream>

using namespace dealii;

// Source term — set for your problem
template <int dim>
class RightHandSide : public Function<dim>
{{
public:
  virtual double value(const Point<dim> &p,
                       const unsigned int) const override
  {{
    // Source term value — set for your problem
    double val = {rhs_value};
    for (unsigned int d = 0; d < dim; ++d)
      val *= std::sin(numbers::PI * p[d]);
    return val;
  }}
}};

int main()
{{
  const unsigned int dim = 2;
  const unsigned int min_degree = {min_degree};
  const unsigned int max_degree = {max_degree};

  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, -1.0, 1.0);
  triangulation.refine_global({refinements});

  // Build hp FE collection with polynomial degrees min_degree..max_degree
  hp::FECollection<dim> fe_collection;
  hp::QCollection<dim>  q_collection;
  hp::QCollection<dim-1> q_collection_face;

  for (unsigned int degree = min_degree; degree <= max_degree; ++degree)
    {{
      fe_collection.push_back(FE_Q<dim>(degree));
      q_collection.push_back(QGauss<dim>(degree + 1));
      q_collection_face.push_back(QGauss<dim-1>(degree + 1));
    }}

  DoFHandler<dim> dof_handler(triangulation);

  // Fourier series for smoothness estimation
  const unsigned int N = max_degree;
  const std::vector<unsigned int> n_coefficients_per_direction(dim, N);
  FESeries::Fourier<dim> fourier(n_coefficients_per_direction,
                                  fe_collection,
                                  q_collection);

  for (unsigned int cycle = 0; cycle < {n_cycles}; ++cycle)
    {{
      // Distribute DOFs with current hp assignment
      dof_handler.distribute_dofs(fe_collection);

      // Constraints (hanging nodes + Dirichlet BCs)
      AffineConstraints<double> constraints;
      DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      VectorTools::interpolate_boundary_values(dof_handler,
                                                0,
                                                Functions::ZeroFunction<dim>(),
                                                constraints);
      constraints.close();

      // Sparsity and system
      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
      SparsityPattern sparsity_pattern;
      sparsity_pattern.copy_from(dsp);

      SparseMatrix<double> system_matrix(sparsity_pattern);
      Vector<double> solution(dof_handler.n_dofs());
      Vector<double> system_rhs(dof_handler.n_dofs());

      // Assembly with hp quadrature
      RightHandSide<dim> rhs_function;

      hp::MappingCollection<dim> mapping_collection(MappingQ1<dim>());

      for (const auto &cell : dof_handler.active_cell_iterators())
        {{
          const unsigned int dofs_per_cell = cell->get_fe().n_dofs_per_cell();
          FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
          Vector<double>     cell_rhs(dofs_per_cell);
          std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

          FEValues<dim> fe_values(cell->get_fe(),
                                  q_collection[cell->active_fe_index()],
                                  update_values | update_gradients |
                                  update_quadrature_points | update_JxW_values);
          fe_values.reinit(cell);

          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {{
              const double rhs_val = rhs_function.value(fe_values.quadrature_point(q), 0);
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {{
                  for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                          fe_values.shape_grad(j, q) *
                                          fe_values.JxW(q);
                  cell_rhs(i) += rhs_val *
                                  fe_values.shape_value(i, q) *
                                  fe_values.JxW(q);
                }}
            }}

          cell->get_dof_indices(local_dof_indices);
          constraints.distribute_local_to_global(cell_matrix, cell_rhs,
                                                  local_dof_indices,
                                                  system_matrix, system_rhs);
        }}

      // Solve
      SolverControl solver_control(dof_handler.n_dofs(), 1e-12);
      SolverCG<Vector<double>> solver(solver_control);
      PreconditionSSOR<SparseMatrix<double>> preconditioner;
      preconditioner.initialize(system_matrix, 1.2);
      solver.solve(system_matrix, solution, system_rhs, preconditioner);
      constraints.distribute(solution);

      std::cout << "Cycle " << cycle
                << ": " << dof_handler.n_dofs() << " DOFs, "
                << triangulation.n_active_cells() << " cells"
                << std::endl;

      // Error estimation
      Vector<float> estimated_error(triangulation.n_active_cells());
      KellyErrorEstimator<dim>::estimate(dof_handler,
                                          q_collection_face,
                                          {{}},
                                          solution,
                                          estimated_error);

      // Smoothness estimation for hp decision
      Vector<float> smoothness(triangulation.n_active_cells());
      SmoothnessEstimator::Fourier::coefficient_decay(fourier,
                                                       dof_handler,
                                                       solution,
                                                       smoothness);

      // Mark cells for refinement/coarsening
      GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                       estimated_error,
                                                       0.3, 0.03);

      // hp decision: smooth cells get p-refinement, rough cells get h-refinement
      hp::Refinement::p_adaptivity_from_reference(dof_handler,
                                                    smoothness,
                                                    smoothness,
                                                    0.5, 0.5);

      // Combine h and p decisions
      hp::Refinement::choose_p_over_h(dof_handler);

      triangulation.execute_coarsening_and_refinement();
    }}

  // Output final solution
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  // Track active FE index (polynomial degree)
  Vector<float> fe_degrees(triangulation.n_active_cells());
  unsigned int idx = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    fe_degrees[idx++] = static_cast<float>(cell->active_fe_index() + min_degree);
  data_out.add_data_vector(fe_degrees, "fe_degree");

  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);

  std::cout << "hp-adaptive FEM: "
            << dof_handler.n_dofs() << " DOFs, "
            << triangulation.n_active_cells() << " cells, "
            << "degree range [" << min_degree << ", " << max_degree << "]"
            << std::endl;

  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "hp-adaptive FEM with smoothness estimation (step-27, step-75)",
    "tutorial_steps": ["step-27 (hp-adaptive, Fourier smoothness)", "step-75 (matrix-free hp-GMG)"],
    "function_space": "hp::FECollection<dim> with FE_Q<dim>(1..max_degree)",
    "solver": "CG + SSOR (serial), CG + AMG (parallel)",
    "smoothness_estimation": {
        "Fourier": "FESeries::Fourier — decay of Fourier coefficients indicates regularity",
        "Legendre": "FESeries::Legendre — expansion in Legendre polynomials",
        "decay_rate": "Fast decay → smooth → increase p, slow decay → singular → refine h",
    },
    "hp_decision": {
        "p_adaptivity_from_reference": "Compare smoothness to reference values",
        "choose_p_over_h": "Prefer p-refinement when both flagged",
        "fixed_number": "Refine fraction of cells with largest error",
    },
    "pitfalls": [
        "[Syntax] hp::FECollection must include all FE_Q degrees "
        "you want to use, registered before distribute_dofs(). "
        "Missing degrees give an active_fe_index that points at "
        "nothing. Signal: `dof_handler.distribute_dofs(fe_"
        "collection)` raises ExcMessage('Index in FECollection "
        "out of range') or, worse, returns silently with "
        "n_dofs() = 0 on cells with the missing degree.",
        "[Syntax] hp::QCollection must match: each FE_Q(p) needs "
        "QGauss(p+1). A single QGauss(2) used across all degrees "
        "is the most common bug — under-integrates higher p and "
        "the assembly produces a non-symmetric stiffness. Signal: "
        "SolverCG reports 'breakdown' because the assembled K is "
        "not SPD; FECollection::size() and QCollection::size() "
        "differ.",
        "[API] Smoothness estimator needs FESeries::Fourier or "
        "FESeries::Legendre object — the smoothness decay rate "
        "drives p- vs h-refinement choice. Without it, "
        "p_adaptivity_from_smoothness silently falls back to "
        "uniform refinement. Signal: dof_handler.n_dofs() grows "
        "as O(N) instead of O(log N) on a smooth solution; "
        "VectorTools::integrate_difference against analytic "
        "reference shows pure h-convergence rate instead of "
        "exponential-in-p.",
        "[Numerical] Hanging-node constraints more complex with "
        "different p on neighbours — AffineConstraints needs the "
        "p-projection in addition to the h-projection. Forgetting "
        "this produces solution jumps at p-transitions. Signal: "
        "DataOut shows step discontinuities at cell boundaries "
        "where the neighbours have different FE_Q degree; "
        "VectorTools::integrate_difference reports O(1) error "
        "along those interfaces and ~O(h^p) elsewhere.",
        "[API] For matrix-free hp: use the step-75 pattern with "
        "MatrixFree<dim,number,VectorizedArray>. The standard "
        "MatrixFree pattern from step-37 assumes uniform p and "
        "ExcMessage('all cells must have same active_fe_index') "
        "fires on a mixed-p triangulation. Signal: MatrixFree::"
        "reinit raises ExcMessage('hp-FEValues requires hp::"
        "MappingCollection') or compiles but produces zero "
        "stiffness contribution on cells with the non-default "
        "active_fe_index.",
        "[Numerical] Transfer solution between p-levels: use "
        "SolutionTransfer or VectorTools::interpolate. Setting "
        "solution values directly across a p-change discards "
        "the high-frequency content and breaks Newton "
        "continuation. Signal: DataOut frame at the p-refinement "
        "step shows solution.linfty_norm() dropping by 10-50% "
        "(the high-frequency content lost in the transfer); "
        "next Newton step has to re-construct it from scratch.",
    ],
}
