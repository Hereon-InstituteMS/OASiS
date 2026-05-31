"""Poisson / Laplace equation templates for deal.II.

Based on deal.II tutorial steps 3, 5, 6.
"""


def _poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-3.
    """
    refinements = params.get("refinements", 5)
    return f'''\
/* Poisson equation on unit square — based on deal.II step-3
 * -laplacian(u) = 1 on [0,1]^2, u = 0 on boundary
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int main()
{{
  Triangulation<2> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global({refinements});

  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  std::cout << "Number of DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);

  Vector<double> solution;
  solution.reinit(dof_handler.n_dofs());
  Vector<double> system_rhs;
  system_rhs.reinit(dof_handler.n_dofs());

  // Assemble
  QGauss<2> quadrature_formula(fe.degree + 1);
  FEValues<2> fe_values(fe, quadrature_formula,
                        update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {{
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {{
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                   fe_values.shape_grad(j, q) *
                                   fe_values.JxW(q);
            cell_rhs(i) += fe_values.shape_value(i, q) * 1.0 *
                           fe_values.JxW(q);
          }}

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {{
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }}
    }}

  // Boundary conditions
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<2>(),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);

  // Solve
  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "Solver converged in " << solver_control.last_step()
            << " iterations." << std::endl;
  std::cout << "min(u) = " << *std::min_element(solution.begin(), solution.end())
            << ", max(u) = " << *std::max_element(solution.begin(), solution.end())
            << std::endl;

  // Output
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

  std::ofstream output("result.vtu");
  data_out.write_vtu(output);

  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}}
'''


def _poisson_3d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-3.
    """
    refinements = params.get("refinements", 3)
    return f'''\
/* Poisson equation on unit cube — based on deal.II step-3
 * -laplacian(u) = 1 on [0,1]^3, u = 0 on boundary
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int main()
{{
  Triangulation<3> triangulation;
  GridGenerator::hyper_cube(triangulation);
  triangulation.refine_global({refinements});

  FE_Q<3> fe(1);
  DoFHandler<3> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  std::cout << "Number of DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);

  Vector<double> solution;
  solution.reinit(dof_handler.n_dofs());
  Vector<double> system_rhs;
  system_rhs.reinit(dof_handler.n_dofs());

  // Assemble
  QGauss<3> quadrature_formula(fe.degree + 1);
  FEValues<3> fe_values(fe, quadrature_formula,
                        update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {{
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (unsigned int q = 0; q < quadrature_formula.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {{
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                   fe_values.shape_grad(j, q) *
                                   fe_values.JxW(q);
            cell_rhs(i) += fe_values.shape_value(i, q) * 1.0 *
                           fe_values.JxW(q);
          }}

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {{
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }}
    }}

  // Boundary conditions
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<3>(),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);

  // Solve
  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "Solver converged in " << solver_control.last_step()
            << " iterations." << std::endl;
  std::cout << "min(u) = " << *std::min_element(solution.begin(), solution.end())
            << ", max(u) = " << *std::max_element(solution.begin(), solution.end())
            << std::endl;

  // Output
  DataOut<3> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();

  std::ofstream output("result.vtu");
  data_out.write_vtu(output);

  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}}
'''


def _poisson_l_domain(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Uses deal.II built-in GridGenerator::hyper_L.
    """
    refinements = params.get("refinements", 5)
    return f'''\
/* Poisson on L-shaped domain — deal.II
 * -laplacian(u) = 1, u = 0 on boundary
 * Non-trivial geometry with re-entrant corner singularity.
 * Uses built-in GridGenerator::hyper_L (no external mesher needed).
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int main()
{{
  Triangulation<2> triangulation;
  GridGenerator::hyper_L(triangulation, -1, 1);
  triangulation.refine_global({refinements});

  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "L-domain DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  QGauss<2> quadrature(fe.degree + 1);
  FEValues<2> fe_values(fe, quadrature,
    update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {{
      fe_values.reinit(cell);
      cell_matrix = 0; cell_rhs = 0;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {{
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                   fe_values.shape_grad(j, q) *
                                   fe_values.JxW(q);
            cell_rhs(i) += fe_values.shape_value(i, q) * 1.0 * fe_values.JxW(q);
          }}
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {{
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }}
    }}

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<2>(), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "Solver: " << solver_control.last_step() << " iterations" << std::endl;
  std::cout << "min(u) = " << *std::min_element(solution.begin(), solution.end())
            << ", max(u) = " << *std::max_element(solution.begin(), solution.end()) << std::endl;

  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}}
'''


def _poisson_rectangle(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    refinements = params.get("refinements", 5)
    lx = params.get("lx", 2.0)
    ly = params.get("ly", 1.0)
    return f'''\
/* Poisson on [{lx}x{ly}] rectangle — deal.II
 * -laplacian(u) = 1, u = 0 on boundary
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int main()
{{
  Triangulation<2> triangulation;
  GridGenerator::subdivided_hyper_rectangle(triangulation,
    {{(unsigned int)({int(lx * 8)}), (unsigned int)({int(ly * 8)})}},
    Point<2>(0, 0), Point<2>({lx}, {ly}));
  triangulation.refine_global({refinements});

  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  QGauss<2> quadrature(fe.degree + 1);
  FEValues<2> fe_values(fe, quadrature,
    update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {{
      fe_values.reinit(cell);
      cell_matrix = 0; cell_rhs = 0;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {{
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                   fe_values.shape_grad(j, q) *
                                   fe_values.JxW(q);
            cell_rhs(i) += fe_values.shape_value(i, q) * 1.0 * fe_values.JxW(q);
          }}
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {{
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }}
    }}

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<2>(), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "min(u) = " << *std::min_element(solution.begin(), solution.end())
            << ", max(u) = " << *std::max_element(solution.begin(), solution.end()) << std::endl;

  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  return 0;
}}
'''


def _poisson_adaptive_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-6.
    """
    cycles = params.get("cycles", 6)
    order = params.get("order", 2)
    return f'''\
/* Poisson with AMR — step-6 based — deal.II */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <fstream>
#include <iostream>
using namespace dealii;

int main() {{
  const int dim = 2;
  Triangulation<dim> tria;
  GridGenerator::hyper_L(tria, -1, 1);

  FE_Q<dim> fe({order});
  DoFHandler<dim> dof_handler(tria);

  for (unsigned int cycle = 0; cycle < {cycles}; ++cycle) {{
    dof_handler.distribute_dofs(fe);

    AffineConstraints<double> constraints;
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler, 0,
      Functions::ZeroFunction<dim>(), constraints);
    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    SparsityPattern sp;
    sp.copy_from(dsp);

    SparseMatrix<double> system_matrix;
    system_matrix.reinit(sp);
    Vector<double> solution(dof_handler.n_dofs());
    Vector<double> system_rhs(dof_handler.n_dofs());

    QGauss<dim> quadrature(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature,
      update_values | update_gradients | update_JxW_values);

    const unsigned int dpc = fe.n_dofs_per_cell();
    FullMatrix<double> cell_matrix(dpc, dpc);
    Vector<double> cell_rhs(dpc);
    std::vector<types::global_dof_index> local_dof_indices(dpc);

    for (const auto &cell : dof_handler.active_cell_iterators()) {{
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs = 0;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dpc; ++i) {{
          for (unsigned int j = 0; j < dpc; ++j)
            cell_matrix(i, j) += fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q)
                                 * fe_values.JxW(q);
          cell_rhs(i) += 1.0 * fe_values.shape_value(i, q) * fe_values.JxW(q);
        }}
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices,
                                             system_matrix, system_rhs);
    }}

    SolverControl sc(1000, 1e-12);
    SolverCG<Vector<double>> solver(sc);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

    std::cout << "Cycle " << cycle << ": " << dof_handler.n_dofs() << " DOFs, "
              << sc.last_step() << " CG iters, max(u)=" << solution.linfty_norm() << std::endl;

    // Error estimation and refinement
    Vector<float> error_per_cell(tria.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler,
      QGauss<dim - 1>(fe.degree + 1), {{}}, solution, error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_number(tria, error_per_cell, 0.3, 0.03);
    tria.execute_coarsening_and_refinement();
  }}

  // Final output
  dof_handler.distribute_dofs(fe);
  // (re-solve on final mesh if needed)
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  // Output the mesh structure
  data_out.build_patches();
  std::ofstream mesh_output("solution.vtu");
  data_out.write_vtu(mesh_output);
  std::cout << "AMR complete." << std::endl;
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Poisson equation solved with deal.II (step-3/4/5/6/7)",
    "tutorial_steps": ["step-3 (basic)", "step-5 (variable coefficients)", "step-6 (AMR)",
                      "step-37 (matrix-free)", "step-40 (parallel)"],
    "function_space": "FE_Q<dim>(p) — Lagrange, any order p",
    "solver": "CG + SSOR/AMG. Matrix-free: MatrixFree + FEEvaluation (step-37)",
    "adaptive_refinement": "KellyErrorEstimator + refine_and_coarsen_fixed_number (step-6)",
    # ── Structured keys for Poisson — the canonical first physics.
    #    GENERAL_KNOWLEDGE in this same module enumerates the full
    #    H1/H1_enriched/nonconforming/H_div/H_curl menu; this list
    #    is the Poisson-relevant subset only.
    "elements": [
        "FE_Q<dim>(degree)            — continuous Lagrange; degree=1 (default), degree=2 for smoother solutions, higher for spectral convergence on smooth problems",
        "FE_Q_Hierarchical<dim>(degree) — hierarchical basis; required for p-adaptive refinement (coarse DoFs survive a degree change, regular Lagrange ones do not)",
        "FE_Bernstein<dim>(degree)    — Bernstein-Bezier; better-conditioned mass matrix at high p",
        "FE_Q_iso_Q1<dim>(p)          — piecewise (multi-)linear on p^dim sub-cells of each macro-cell; same continuity as FE_Q(1) but more DoFs",
        "FE_DGQ<dim>(degree)          — discontinuous Galerkin Lagrange; for DG-Poisson formulations (interior-penalty)",
        "FE_DGP<dim>(degree)          — DG with monomial pressure-like basis on hyper-cube cells",
        # deal.II ≥ 9.3 ships fe_simplex_p.h; the conda install used
        # at the time of catalog encoding (9.1.1) does NOT have it,
        # so claiming it without the version gate is a false promise.
        # When upgrading deal.II, this entry becomes immediately
        # usable.
        "FE_SimplexP<dim>(degree)     — Lagrange on simplex (triangle / tet) cells; available in deal.II ≥ 9.3 — use when the mesh is unstructured from Gmsh",
    ],
    "mesh_generators": [
        "GridGenerator::hyper_cube(tria, a, b)                        — [a,b]^dim unit cube; classic Poisson on the square",
        "GridGenerator::hyper_rectangle(tria, p1, p2)                 — axis-aligned box; non-square aspect ratio",
        "GridGenerator::subdivided_hyper_cube(tria, n_per_dir, a, b)  — pre-subdivided to avoid repeated refine_global() calls",
        "GridGenerator::hyper_L(tria, a, b)                           — L-shaped, has corner singularity u ~ r^{2/3} sin(2θ/3) — canonical test for adaptive refinement",
        "GridGenerator::hyper_ball(tria, center, radius)              — circular / spherical disk; curved boundary",
        "GridGenerator::hyper_shell(tria, center, inner, outer)       — annulus; tests boundary-conforming refinement",
        "GridGenerator::cheese(tria, ...)                             — domain with holes; multiscale / heterogeneous-coefficient demo",
        "GridGenerator::torus(tria, R, r)                             — 3D toroidal domain; for periodic-boundary studies",
    ],
    "solvers": [
        "SolverCG<>                   — Poisson is symmetric positive-definite; CG is the default",
        "SolverGMRES<>                — only needed if coefficients break symmetry (e.g. when stabilisation is added)",
        "MatrixFree + FEEvaluation    — step-37 matrix-free; needed for matrix-storage-bound problems past ~10^7 DoFs",
    ],
    "preconditioners": [
        "PreconditionSSOR             — serial default; cheap on SPD systems",
        "PreconditionAMG / BoomerAMG  — parallel; via TrilinosWrappers, scales to 10^7 DoFs",
        "PreconditionChebyshev        — used inside multigrid as smoother, also as a standalone for matrix-free",
        "MGSmootherRelaxation         — geometric multigrid smoother (step-16, step-50)",
    ],
    "pitfalls": [
        "[Syntax] Must call triangulation.refine_global() before "
        "distributing DOFs. Calling distribute_dofs on a 1-cell "
        "triangulation runs but gives a useless 4-DoF system. "
        "Signal: solver reports 'converged in 0 iterations' on a "
        "non-trivial problem because the system has only 1 DOF.",
        "[Syntax] Boundary IDs on hyper_cube: ALL faces have "
        "boundary_id=0. To distinguish faces you must iterate "
        "the cells and re-tag faces after the mesh exists. Signal: "
        "you apply different Dirichlet values on different faces "
        "but the result has the SAME value everywhere on the "
        "boundary.",
        "[Syntax] For hyper_rectangle: left=0, right=1 "
        "(in 2D: bottom=2, top=3; in 3D: front=4, back=5). The "
        "rectangle has them auto-assigned. Always check via "
        "GridTools::get_boundary_ids(tria) after creating the "
        "mesh.",
        "[Numerical] For AMR: MUST apply hanging-node constraints "
        "after assembly via constraints.condense(system_matrix, "
        "system_rhs). Forgetting this gives a non-symmetric matrix "
        "and CG breaks down. Signal: SolverCG reports 'breakdown' "
        "on iteration 2-3 on a refined mesh, but works on the "
        "globally-refined version of the same problem.",
        "[API] AffineConstraints<double> handles both Dirichlet BCs "
        "and hanging nodes — interpolate_boundary_values + the "
        "hanging-node closure on the SAME constraints object. Using "
        "two separate constraints objects produces inconsistent "
        "assembly. Signal: solution has step discontinuities at "
        "the interfaces between refinement levels.",
        # New entries shipped with this encoding pass — common Poisson
        # failure modes the catalog should warn about.
        "[Numerical] For variable-coefficient problems (a(x) ∇u), if "
        "the coefficient varies over several orders of magnitude "
        "(layered media, heterogeneous), the stiffness matrix gets "
        "ill-conditioned and PreconditionSSOR loses effectiveness. "
        "Switch to PreconditionAMG / BoomerAMG, which respects the "
        "coefficient structure. Signal: CG iteration count grows "
        "linearly with the coefficient contrast.",
        "[Integration] Mixing the 2D and 3D template instantiations "
        "in the same translation unit at unrelated polynomial orders "
        "(FE_Q<2>(2) plus FE_Q<3>(1)) does NOT explicitly instantiate "
        "the lower-degree 3D version unless it appears somewhere in "
        "the program. Signal: link errors like "
        "'undefined reference to FE_Q<3>::FE_Q(unsigned int)' even "
        "though FE_Q<3> appears to be used elsewhere.",
    ],
}

GENERAL_KNOWLEDGE = {
    "description": "deal.II general capabilities",
    "element_types": {
        "H1": "FE_Q(p), FE_Q_Hierarchical(p), FE_Bernstein(p), FE_Hermite(p), FE_SimplexP(p)",
        "H1_enriched": (
            "Strictly H1-conforming enrichments — safe to use anywhere "
            "the formulation needs H1 continuity:  "
            "FE_Q_Bubbles(p) — Q with cell-interior bubble enrichment "
            "(the bubble vanishes on the cell boundary, so inter-element "
            "continuity is preserved).  Upstream caveat: condition number "
            "grows quickly for p > 3; use the lowest applicable degree.  "
            "FE_SimplexP_Bubbles(p) — simplex analogue of FE_Q_Bubbles.  "
            "FE_Q_iso_Q1(p) — piecewise (bi-/tri-)linear functions on a "
            "macro-element of p^dim sub-cells; the cell is conceptually "
            "split into p subdivisions per coordinate direction and a Q1 "
            "basis is laid down on the resulting subcells (still globally "
            "continuous, so H1-conforming)."
        ),
        "nonconforming_and_qp_dg0": (
            "NOT H1-conforming — agent should NOT pick these for an "
            "H1-conforming formulation:  "
            "FE_Q_DG0(p) — Lagrange Qp **plus** the space of cell-wise "
            "constant functions (Qp+DG0).  The added piecewise-constant "
            "part is discontinuous across element boundaries; only the "
            "Lagrange part is continuous.  Used in mixed/stabilised "
            "discretisations that explicitly want the extra discontinuous "
            "mode.  "
            "FE_RannacherTurek(0) — classical first-order *nonconforming* "
            "element (degree argument fixed to 0 in upstream).  Continuity "
            "is enforced only at edge/face midpoints, not pointwise across "
            "faces."
        ),
        "DG": "FE_DGQ(p), FE_DGQLegendre(p), FE_DGQHermite(p), FE_DGP(p), FE_SimplexDGP(p)",
        "DG_advanced": (
            "FE_DGQArbitraryNodes(quadrature) — DG_Q on a user-chosen node "
            "set (Gauss-Lobatto, Gauss, equispaced) for matrix-free / "
            "spectral-element style discretisations; "
            "FE_DGPMonomial(p) — DG using the monomial polynomial basis "
            "rather than the standard nodal basis (kept for legacy "
            "comparison and analytic-coefficient access); "
            "FE_DGPNonparametric(p) — DG with a non-parametric mapping, "
            "i.e. polynomials defined in physical (not reference) space; "
            "FE_DGVector<PolynomialsType> — class template defined in "
            "fe_dg_vector.h that wraps a vector-valued polynomial space "
            "(PolynomialsRaviartThomas, PolynomialsNedelec, PolynomialsBDM) "
            "into a DG element.  The three concrete instantiations are: "
            "FE_DGRaviartThomas(k) — DG element built on the RT polynomial "
            "space (used in DG mixed methods); "
            "FE_DGNedelec(k) — DG element on the Nédélec polynomial space "
            "(discontinuous H(curl)-type approximation); "
            "FE_DGBDM(k) — DG element on the Brezzi-Douglas-Marini "
            "polynomial space (discontinuous H(div)-type approximation)."
        ),
        "H(div)": "FE_RaviartThomas(k), FE_BDM(k), FE_ABF(k), FE_BernardiRaugel(1)",
        "H(div)_advanced": (
            "FE_RaviartThomasNodal(k) — RT with a nodal degree-of-freedom "
            "representation (alternative to the moment-based default), "
            "convenient when interpolating from a nodal velocity field; "
            "FE_RT_Bubbles(k) — RT enriched with interior bubble functions "
            "for improved approximation order on a fixed mesh."
        ),
        "H(curl)": "FE_Nedelec(k), FE_NedelecSZ(k)",
        "H(curl)_advanced": (
            "FE_NedelecNodal(k) — Nédélec element with a nodal-interpolation "
            "DoF setup, useful when coupling against nodal H(curl) data."
        ),
        "trace_and_face": (
            "FE_FaceQ(p) — Q-polynomial face element used for "
            "hybridised DG (HDG) interface unknowns; "
            "FE_FaceP(p) — P-polynomial face element, the simplex "
            "analogue of FE_FaceQ; "
            "FE_TraceQ(p) — trace of FE_Q on element faces, used by "
            "Lagrange-multiplier and HDG stabilisations."
        ),
        "pyramid_and_wedge_3d": (
            "FE_PyramidP(p) — continuous P element on pyramidal (square-base) "
            "3D cells, used in transition meshes between hex and tet regions; "
            "FE_PyramidDGP(p) — DG counterpart of FE_PyramidP; "
            "FE_WedgeP(p) — continuous P element on wedge (triangular-prism) "
            "3D cells, the second transition shape between hex and tet; "
            "FE_WedgeDGP(p) — DG counterpart of FE_WedgeP."
        ),
        "special": "FE_FaceQ(p), FE_Nothing, FE_Enriched, FE_P1NC, FESystem, hp::FECollection",
        "internal_polynomial_bases": (
            "FE_Poly, FE_PolyFace, FE_PolyTensor, FE_Q_Base, FE_SimplexPoly, "
            "FE_PyramidPoly, FE_WedgePoly — abstract polynomial base classes "
            "that the concrete elements above are templated on (e.g. FE_Q is "
            "an FE_Q_Base; FE_PyramidP is an FE_PyramidPoly; FE_SimplexP is "
            "an FE_SimplexPoly).  Listed here only so the agent does not "
            "propose them in user code; these classes have no public "
            "stand-alone constructor and are not directly instantiated."
        ),
    },
    "mesh_generators": [
        "hyper_cube, hyper_rectangle, hyper_L, hyper_ball, hyper_shell",
        "channel_with_cylinder, plate_with_a_hole, cheese, cylinder",
        "merge_triangulations, extrude_triangulation",
        "Import: Gmsh, UCD, VTK, ExodusII, ABAQUS, OpenCASCADE",
    ],
    "parallel": "MPI (p4est) + TBB/Taskflow + CUDA/Kokkos GPU",
    "amr": "KellyErrorEstimator, DWR (step-14), hp-adaptivity (step-27/75)",
    "matrix_free": "MatrixFree + FEEvaluation, sum factorization (step-37/48/59/64/66/67/75/76/95)",
    "output": "VTU (DataOut), higher-order VTU cells, PVTU (parallel), PVD (time series)",
    "unique_features": [
        "97 tutorial programs covering almost every FEM topic",
        "hp-adaptive FEM with automatic smoothness estimation",
        "Matrix-free methods with sum factorization (10x faster than sparse)",
        "GPU support via CUDA and Kokkos",
        "Automatic differentiation via Sacado/ADOL-C for nonlinear problems",
        "Scalable to 10^12 DOFs on 300,000+ MPI processes",
    ],
}
