"""MPI-parallel Poisson solver templates for deal.II.

Based on deal.II tutorial step-40 (p4est distributed mesh).
"""


def _parallel_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.
    All parameter defaults are placeholders.
    MPI-parallel Poisson solver with p4est — based on step-40 pattern.
    """
    refinements = params.get("refinements", 5)
    degree = params.get("degree", 2)
    rhs_value = params.get("rhs_value", 1.0)
    return f'''\
/* MPI-parallel Poisson equation — based on deal.II step-40 pattern
 * Solves -laplacian(u) = f using PETSc/Trilinos on a distributed mesh (p4est).
 */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/mpi.h>
#include <deal.II/lac/generic_linear_algebra.h>

// Hard precondition with ONE actionable diagnostic instead of 200
// lines of template errors from inside generic_linear_algebra.h:
// conda-forge deal.II builds ship WITHOUT MPI and WITHOUT PETSc in
// every version (config.h: '#undef DEAL_II_WITH_MPI' / '..._PETSC',
// verified 9.1.1 + 9.3.2 on 2026-06-12), so the step-40 distributed
// pattern cannot compile there at all.
#if !defined(DEAL_II_WITH_MPI) || !defined(DEAL_II_WITH_PETSC)
#  error "parallel_poisson_2d requires a deal.II build with MPI + PETSc (+ p4est). conda-forge deal.II ships without them - use the serial poisson_2d template instead, or build deal.II from source with -DDEAL_II_WITH_MPI=ON -DDEAL_II_WITH_PETSC=ON."
#endif

// Choose LA backend: PETSc or Trilinos
namespace LA
{{
  using namespace dealii::LinearAlgebraPETSc;
}}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>
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
    double val = {rhs_value};
    for (unsigned int d = 0; d < dim; ++d)
      val *= std::sin(numbers::PI * p[d]);
    return val * dim * numbers::PI * numbers::PI;
  }}
}};

int main(int argc, char *argv[])
{{
  const unsigned int dim = 2;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  const MPI_Comm mpi_communicator = MPI_COMM_WORLD;

  ConditionalOStream pcout(std::cout,
                            Utilities::MPI::this_mpi_process(mpi_communicator) == 0);

  pcout << "Running on "
        << Utilities::MPI::n_mpi_processes(mpi_communicator)
        << " MPI rank(s)" << std::endl;

  // Distributed triangulation via p4est
  parallel::distributed::Triangulation<dim> triangulation(mpi_communicator);
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global({refinements});

  const unsigned int degree = {degree};
  FE_Q<dim>       fe(degree);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  pcout << "Parallel Poisson: " << dof_handler.n_dofs() << " DOFs, "
        << triangulation.n_global_active_cells() << " cells" << std::endl;

  // Constraints
  AffineConstraints<double> constraints;
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler, 0,
                                            Functions::ZeroFunction<dim>(),
                                            constraints);
  constraints.close();

  // Distributed sparsity pattern
  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                              dof_handler.locally_owned_dofs(),
                                              mpi_communicator,
                                              locally_relevant_dofs);

  // System matrix and vectors
  LA::MPI::SparseMatrix system_matrix;
  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);

  LA::MPI::Vector solution;
  solution.reinit(locally_owned_dofs, mpi_communicator);

  LA::MPI::Vector system_rhs;
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);

  // Assembly
  const QGauss<dim> quadrature(degree + 1);
  FEValues<dim>     fe_values(fe, quadrature,
                               update_values | update_gradients |
                               update_quadrature_points | update_JxW_values);

  const RightHandSide<dim> rhs_function;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {{
        const unsigned int dpc = fe.n_dofs_per_cell();
        FullMatrix<double> cell_matrix(dpc, dpc);
        Vector<double>     cell_rhs(dpc);
        std::vector<types::global_dof_index> local_dof_indices(dpc);

        fe_values.reinit(cell);

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
          {{
            const double rhs_val = rhs_function.value(
              fe_values.quadrature_point(q), 0);
            for (unsigned int i = 0; i < dpc; ++i)
              {{
                for (unsigned int j = 0; j < dpc; ++j)
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

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  // Solve with PETSc CG + AMG preconditioner
  SolverControl solver_control(dof_handler.n_dofs(), 1e-12);
  LA::SolverCG  solver(solver_control, mpi_communicator);

  LA::MPI::PreconditionAMG preconditioner;
  LA::MPI::PreconditionAMG::AdditionalData amg_data;
  preconditioner.initialize(system_matrix, amg_data);

  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  pcout << "Solved in " << solver_control.last_step() << " iterations"
        << std::endl;

  constraints.distribute(solution);

  // Output: each process writes its part, DataOut merges via PVTU
  LA::MPI::Vector locally_relevant_solution;
  locally_relevant_solution.reinit(locally_owned_dofs,
                                    locally_relevant_dofs,
                                    mpi_communicator);
  locally_relevant_solution = solution;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(locally_relevant_solution, "solution");

  // Partition visualization
  Vector<float> subdomain(triangulation.n_active_cells());
  for (auto &val : subdomain)
    val = static_cast<float>(triangulation.locally_owned_subdomain());
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  data_out.write_vtu_with_pvtu_record("./", "result", 0, mpi_communicator);

  pcout << "Parallel Poisson: output written" << std::endl;
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "MPI-parallel Poisson with p4est distributed mesh (step-40)",
    "tutorial_steps": ["step-40 (basic parallel Poisson)", "step-50 (parallel GMG)",
                      "step-75 (parallel hp-multigrid, matrix-free)"],
    "function_space": "FE_Q<dim>(p) with parallel::distributed::Triangulation",
    "solver": "PETSc CG + AMG (BoomerAMG), or Trilinos CG + ML/MueLu",
    "parallel_infrastructure": {
        "mesh": "parallel::distributed::Triangulation (p4est backend)",
        "vectors": "LA::MPI::Vector (PETSc or Trilinos)",
        "matrix": "LA::MPI::SparseMatrix (PETSc or Trilinos)",
        "output": "DataOut::write_vtu_with_pvtu_record for parallel VTU",
    },
    "pitfalls": [
        "[Integration] Requires deal.II compiled with MPI + p4est + "
        "PETSc (or Trilinos). Without these the parallel:: classes "
        "are not instantiated. Signal: link error 'undefined "
        "reference to parallel::distributed::Triangulation<dim>::"
        "Triangulation' or `ExcMessage('parallel::distributed::"
        "Triangulation needs deal.II compiled with p4est')`.",
        "[Syntax] Only locally-owned cells are assembled. Use "
        "`cell->is_locally_owned()` to gate the assembly loop. "
        "Signal: assembly runs on every rank but "
        "`system_matrix.frobenius_norm()` is the SAME on rank 0 "
        "and rank 1 — each rank assembled the full mesh, leading "
        "to double-counted entries; Triangulation::n_locally_"
        "owned_active_cells() returns the full count instead of "
        "the per-rank partition.",
        "[API] Locally-relevant DoFs needed for ghosted vectors "
        "(AffineConstraints::distribute on parallel vectors). "
        "Forgetting `DoFTools::extract_locally_relevant_dofs` "
        "gives an empty IndexSet and constraints fail to apply. "
        "Signal: `AffineConstraints::distribute(solution)` raises "
        "ExcMessage('ghost entries not consistent') because the "
        "ghost layer is empty.",
        "[Syntax] SparsityTools::distribute_sparsity_pattern "
        "needed for parallel sparsity. Skipping it produces a "
        "globally-replicated sparsity (each rank holds all rows) "
        "and matrix assembly OOMs at scale. Signal: rss / "
        "VmPeak from /proc/self/status grows linearly with "
        "n_dofs on each MPI rank instead of being O(n_dofs / "
        "n_ranks); SparseMatrix::memory_consumption() exceeds "
        "the expected per-rank slice.",
        "[Syntax] `compress(VectorOperation::add)` required after "
        "assembly to sum contributions across ranks. Without it, "
        "system_rhs holds only this rank's contribution. Signal: "
        "DataOut shows the global solution split into per-rank "
        "regions with mismatched values at MPI subdomain "
        "interfaces; VectorTools::integrate_difference against "
        "the serial reference is O(1) at the interfaces and 0 "
        "elsewhere.",
        "[Numerical] PreconditionAMG / BoomerAMG works out-of-box "
        "for scalar Laplace via TrilinosWrappers but needs tuning "
        "(strong-threshold, smoother) for elasticity / NS / "
        "multiphysics systems. Signal: SolverCG iteration count "
        "from SolverControl::last_step() grows from O(20) on "
        "Laplace to >500 on the same mesh size for elasticity, "
        "with each iteration cost dominated by AMG-apply.",
        "[Integration] Output: write_vtu_with_pvtu_record "
        "produces one .vtu per rank plus one .pvtu index file. "
        "ParaView opens the .pvtu, NOT the per-rank .vtu — opening "
        "the latter shows only that rank's subdomain. Signal: "
        "ParaView shows fragmentary mesh (only one MPI subdomain), "
        "while reading the .pvtu yields the full distributed "
        "result; DataOut::write_vtu (no pvtu_record) produces "
        "the broken-into-pieces output.",
    ],
}
