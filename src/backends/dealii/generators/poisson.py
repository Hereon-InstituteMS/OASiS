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
    "elements": {
        "FE_Q":
            "Canonical Poisson choice. degree=1 default; degree=2 "
            "smoother solutions; higher degree for spectral "
            "convergence on smooth problems.",
        "FE_Q_Hierarchical":
            "Required for p-adaptive refinement on Poisson — "
            "coarse-level DoFs survive a degree change.",
        "FE_Bernstein":
            "For high-p Poisson where mass-matrix conditioning "
            "matters (modal analysis, transient diffusion).",
        "FE_Q_iso_Q1":
            "Cheap multi-linear-on-sub-cells alternative to "
            "FE_Q(p); diagonal lumped mass matrix.",
        "FE_DGQ":
            "Discontinuous Galerkin Poisson via interior-penalty "
            "formulation; needed when coefficients are "
            "discontinuous across cells (heterogeneous media).",
        "FE_DGP":
            "Monomial DG basis; alternative to FE_DGQ for higher-"
            "order accurate Poisson on hyper-cube meshes.",
        "FE_SimplexP":
            "Lagrange on simplex (triangle / tet) cells — needed "
            "when the mesh comes from unstructured Gmsh / "
            "Triangle / TetGen. (Available in deal.II ≥ 9.3; "
            "the canonical element-catalog has the version gate.)",
    },
    "mesh_generators": {
        "hyper_cube": "Classic Poisson on the unit square / cube.",
        "hyper_rectangle": "Non-square aspect ratio.",
        "subdivided_hyper_cube": "Pre-subdivided to avoid repeated refine_global() calls.",
        "hyper_L": "Re-entrant-corner singularity; canonical adaptive-refinement test.",
        "hyper_ball": "Curved boundary; tests boundary-conforming refinement.",
        "hyper_shell": "Annulus; layered radial Poisson problems.",
        "cheese": "Heterogeneous-coefficient demos.",
        "torus": "3D periodic-boundary studies.",
    },
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
        "Signal: SolverControl reports 'Convergence step 0 value "
        "X.XXe-16' (already converged), `dof_handler.n_dofs()` "
        "returns 4, and KellyErrorEstimator output is a single "
        "cell-wise scalar.",
        "[Syntax] Boundary IDs on hyper_cube: with the DEFAULT "
        "colorize=false ALL faces have boundary_id=0. The fix is NOT "
        "to re-tag faces by hand — pass colorize=true: "
        "GridGenerator::hyper_cube(tria, 0, 1, /*colorize=*/true) "
        "assigns 0:x=0, 1:x=1, 2:y=0, 3:y=1 (3D adds 4:z=0, 5:z=1). "
        "Signal: `tria.get_boundary_ids()` returns `{0}` "
        "(a single id, not the 4-6 expected for a cube), and "
        "VectorTools::interpolate_boundary_values applied to "
        "different boundary_ids produces the same Dirichlet "
        "values across all faces of the cube. Measured on deal.II "
        "9.8.0-pre 2026-08-03: colorize=false -> {0}; colorize=true "
        "-> {0,1,2,3} in 2D and {0,1,2,3,4,5} in 3D. (This entry "
        "used to prescribe manual re-tagging, contradicting the "
        "poisson_mixed_bc catalog which already documented "
        "colorize=true.)",
        "[Syntax] hyper_rectangle and subdivided_hyper_rectangle do "
        "NOT auto-assign per-face ids: their colorize argument also "
        "defaults to false, so every face lands on boundary_id=0. "
        "With colorize=true you get left=0, right=1, bottom=2, "
        "top=3 (3D adds front=4, back=5). Always check via "
        "tria.get_boundary_ids() after creating the mesh. "
        "Signal: tria.get_boundary_ids() returns `{0}`, not the "
        "assumed `{0,1,2,3}` — a Dirichlet loop keyed on "
        "boundary_id=1..3 then matches no faces and those sides "
        "silently become homogeneous Neumann. Measured on deal.II "
        "9.8.0-pre 2026-08-03: hyper_rectangle<2> default -> {0}, "
        "colorize=true -> {0,1,2,3}; subdivided_hyper_rectangle<3> "
        "colorize=true -> {0,...,5}. (This entry previously stated "
        "the ids were auto-assigned and printed the OPPOSITE "
        "signal.)",
        "[Numerical] For AMR: hanging-node constraints are "
        "MANDATORY — build them with "
        "DoFTools::make_hanging_node_constraints and apply them "
        "either through AffineConstraints::"
        "distribute_local_to_global during assembly (preferred) or "
        "with the still-supported "
        "constraints.condense(sparsity) + "
        "constraints.condense(system_matrix, system_rhs) pair. "
        "But do NOT expect a loud failure if you forget: the "
        "unconstrained system is still EXACTLY SYMMETRIC and CG "
        "converges normally. The failure is silent and purely in "
        "the answer. Measured on deal.II 9.8.0-pre 2026-08-03 "
        "(unit-square MMS u=sin(pi x)sin(pi y), FE_Q(1), 6 cycles "
        "refining the lower-left quadrant, CG + PreconditionSSOR): "
        "without constraints CG converged in 8/11/16/26/47/89 "
        "steps and with constraints in 8/10/16/26/47/86 — "
        "essentially the same cost, no blow-up either way — while "
        "the L2 error GREW 3.04e-2 -> 6.74e-2 without them and "
        "fell 3.04e-2 -> 1.35e-2 with them. Do NOT use matrix "
        "symmetry as the tell: max|A_ij-A_ji| is bitwise 0.0 with "
        "AND without the constraints, so it cannot distinguish the "
        "two. max|A| does (2.67 unconstrained vs 3.17 constrained: "
        "condensation adds to the diagonal). Signal: an L2 error "
        "that stops decreasing (or increases) under refinement "
        "while the solver looks healthy. (This entry previously claimed a "
        "non-symmetric matrix and 'SolverCG breakdown on iteration "
        "2-3'; neither reproduces.)",
        "[API] AffineConstraints<double> handles both Dirichlet BCs "
        "and hanging nodes — interpolate_boundary_values + the "
        "hanging-node closure on the SAME constraints object. Using "
        "two separate constraints objects produces inconsistent "
        "assembly. Signal: DataOut output shows step "
        "discontinuities of order 1e-2 to 1e-1 at refinement-level "
        "interfaces (hanging-node faces); SolverCG converges but "
        "the L2-error against an analytic reference plateaus "
        "instead of decreasing as h is refined.",
        # New entries shipped with this encoding pass — common Poisson
        # failure modes the catalog should warn about.
        "[Numerical] For variable-coefficient problems (a(x) ∇u), if "
        "the coefficient varies over several orders of magnitude "
        "(layered media, heterogeneous), the stiffness matrix gets "
        "ill-conditioned and PreconditionSSOR loses effectiveness. "
        "Switch to PreconditionAMG / BoomerAMG, which respects the "
        "coefficient structure. Signal: SolverCG iteration count "
        "from SolverControl::last_step() grows linearly with the "
        "max/min ratio of the coefficient (e.g. 50 iterations at "
        "contrast 1e2, 500 at contrast 1e3); switching to "
        "PreconditionAMG drops it back to O(log(ndof)).",
        "[Integration] The installed deal.II is a RELEASE-only build "
        "on this machine (CMAKE_BUILD_TYPE=Release, only "
        "libdeal_II.so, no libdeal_II.g.so). Every downstream target "
        "is therefore compiled with -DNDEBUG and every `Assert(...)` "
        "in the library is compiled out — only `AssertThrow` "
        "survives. Requesting a debug build does not help: "
        "cmake -DCMAKE_BUILD_TYPE=Debug prints "
        "'#  WARNING: ... CMAKE_BUILD_TYPE was forced to \"Release\"' "
        "and still emits -DNDEBUG (measured 2026-08-03). "
        "Consequence for this whole catalog: any pitfall promising "
        "'raises ExcMessage(...)' will NOT fire here — the same "
        "misuse instead returns silently with a wrong answer or "
        "segfaults. Signal: `grep CMAKE_BUILD_TYPE "
        "$DEAL_II_DIR/CMakeCache.txt` prints "
        "'CMAKE_BUILD_TYPE:STRING=Release', "
        "`ls $DEAL_II_DIR/lib` shows libdeal_II.so with no "
        "libdeal_II.g.so, and the generated "
        "CMakeFiles/<target>.dir/flags.make carries "
        "'CXX_DEFINES = -DNDEBUG'. Verified examples on 9.8.0-pre: "
        "SparseMatrix::add() into an entry outside the sparsity "
        "pattern returns normally and DROPS the value; "
        "FEValues::get_function_gradients() with a scalar-shaped "
        "container on a vector FESystem returns normally; setting "
        "an active_fe_index beyond hp::FECollection::size() and "
        "calling distribute_dofs SEGFAULTS (rc=139). "
        "(This slot previously held a claim that FE_Q<2>(2) plus "
        "FE_Q<3>(1) in one translation unit produces "
        "'undefined reference to FE_Q<3>::FE_Q(unsigned int)'. "
        "It does not: deal.II pre-instantiates FE_Q<1>, FE_Q<2> "
        "and FE_Q<3>, and the two-dimension program compiled, "
        "linked and ran on 2026-08-03.)",
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
    # ── Cross-physics gaps found by the 2026-08-03 execution pass on
    #    the INSTALLED deal.II 9.8.0-pre. Every entry below was
    #    reproduced by compiling and running a program; none of them
    #    was covered anywhere in this catalog before.
    "install_and_build_gotchas": [
        "[Build] The installed deal.II is RELEASE-only "
        "(CMAKE_BUILD_TYPE=Release in "
        "/home/alexander/dealii/build/CMakeCache.txt; only "
        "libdeal_II.so exists, there is no libdeal_II.g.so). Every "
        "downstream target therefore compiles with -DNDEBUG and every "
        "`Assert(...)` inside deal.II is compiled out — only "
        "`AssertThrow` survives. Asking for a debug build does not "
        "help: `cmake -DCMAKE_BUILD_TYPE=Debug` prints the banner "
        "'#  WARNING: / CMAKE_BUILD_TYPE \"Debug\" unsupported by "
        "current installation! / deal.II was configured with "
        "\"Release\". / CMAKE_BUILD_TYPE was forced to \"Release\".' "
        "and still emits -DNDEBUG (verified 2026-08-03). CONSEQUENCE: "
        "any pitfall in this catalog whose Signal is 'raises "
        "ExcMessage(...)' will NOT fire on this install; the same "
        "mistake instead returns silently with a wrong answer, or "
        "segfaults. Three verified instances: (a) SparseMatrix::add() "
        "into an entry outside the sparsity pattern returns normally "
        "and DROPS the value; (b) FEValues::get_function_gradients() "
        "called with a scalar-shaped std::vector<Tensor<1,dim>> on a "
        "vector-valued FESystem returns normally; (c) setting an "
        "active_fe_index >= hp::FECollection::size() and then calling "
        "distribute_dofs SEGFAULTS with rc=139 and no message. Check "
        "with `grep CMAKE_BUILD_TYPE $DEAL_II_DIR/CMakeCache.txt` "
        "before trusting any Assert-based diagnostic.",

        "[API] Non-interpolatory elements have no support points, and "
        "VectorTools::interpolate_boundary_values cannot be used on "
        "them. `FiniteElement::has_support_points()` is FALSE for "
        "FE_RannacherTurek (and for the H(div)/H(curl) families, whose "
        "DoFs are moments, not point values). On this Release build "
        "the call does not assert — it SEGFAULTS. Verified "
        "2026-08-03: FESystem<2>(FE_RannacherTurek<2>(), 2) on a "
        "2-refinement colorized hyper_cube reports "
        "has_support_points()=0, has_generalized_support_points()=1, "
        "and interpolate_boundary_values(dh, 0, ZeroFunction<2>(2), "
        "constraints) dies with rc=139. The working replacement is "
        "VectorTools::project_boundary_values(dh, {{0, &f}}, "
        "QGauss<1>(3), boundary_values), which returned 8 boundary "
        "values on the identical setup. Prefer has_support_points() "
        "as the guard before choosing interpolate vs project.",

        "[API] deal.II 9.1 -> 9.8 removed several APIs that older "
        "templates still use. Each of these is a hard compile error, "
        "which is the good case — but they are easy to mistake for a "
        "broken install. Verified by compiling this catalog's own "
        "generators on 9.8.0-pre 2026-08-03: "
        "DoFTools::count_dofs_per_block -> count_dofs_per_fe_block "
        "(now RETURNS the vector, no out-parameter); "
        "MGTransferPrebuilt::build_matrices -> build; "
        "MatrixFree::reinit(dof, constraints, quad, data) -> the "
        "mapping must be the first argument; "
        "MatrixFree::n_macro_cells -> n_cell_batches; "
        "FEEvaluation::evaluate/integrate(bool, bool) -> "
        "EvaluationFlags::values / EvaluationFlags::gradients; "
        "SolutionTransfer::interpolate(in, out) -> interpolate(out). "
        "DoFTools::extract_locally_relevant_dofs still accepts BOTH "
        "the modern returning form and the old (dof, IndexSet&) "
        "out-parameter form on 9.8.",

        "[Integration] Availability probes that only check whether a "
        "header includes are WRONG on a SOURCE build. A source install "
        "ships every header regardless of which optional dependencies "
        "were enabled, and the bodies sit behind `#ifdef "
        "DEAL_II_WITH_<FEATURE>`. Verified 2026-08-03 on this build "
        "(MPI, P4EST, PETSC, SLEPC, TRILINOS, SUNDIALS, ADOLC, "
        "SYMENGINE, COMPLEX_VALUES all undefined): "
        "`#include <deal.II/lac/slepc_solver.h>` compiles and links "
        "fine, and only naming SLEPcWrappers fails "
        "(\"'dealii::SLEPcWrappers' has not been declared\"); "
        "`#include <deal.II/distributed/tria.h>` also compiles, and "
        "constructing parallel::distributed::Triangulation fails at "
        "COMPILE time with 'use of deleted function'; "
        "Utilities::MPI::MPI_InitFinalize constructs happily and "
        "reports n_mpi_processes == 1. The only reliable probe is to "
        "grep $DEAL_II_DIR/include/deal.II/base/config.h for the "
        "'/* #undef DEAL_II_WITH_<X> */' lines.",
    ],
    "parallel": "MPI (p4est) + TBB/Taskflow + CUDA/Kokkos GPU",
    "amr": "KellyErrorEstimator, DWR (step-14), hp-adaptivity (step-27/75)",
    "matrix_free": "MatrixFree + FEEvaluation, sum factorization (step-37/48/59/64/66/67/75/76/95)",
    "output": "VTU (DataOut), higher-order VTU cells, PVTU (parallel), PVD (time series)",
    "unique_features": [
        "88 tutorial programs covering almost every FEM topic (numbered to step-97 with gaps; counted on the installed 9.8.0-pre source tree 2026-08-03)",
        "hp-adaptive FEM with automatic smoothness estimation",
        "Matrix-free methods with sum factorization (10x faster than sparse)",
        "GPU support via CUDA and Kokkos",
        "Automatic differentiation via Sacado/ADOL-C for nonlinear problems",
        "Scalable to 10^12 DOFs on 300,000+ MPI processes",
    ],
    "cmake_user_macros": {
        "description": (
            "User-callable DEAL_II_* CMake macros that downstream users "
            "invoke in their CMakeLists.txt. Source: "
            "dealii/cmake/macros/macro_deal_ii_*.cmake."
        ),
        "DEAL_II_INITIALIZE_CACHED_VARIABLES": {
            "signature": "DEAL_II_INITIALIZE_CACHED_VARIABLES()",
            "purpose": "Inherit deal.II's compiler + build settings.",
            "order_constraint": (
                "MUST be called AFTER find_package(deal.II) — needs "
                "DEAL_II_PROJECT_CONFIG_INCLUDED. AND MUST be called "
                "BEFORE project() — sets CMAKE_CXX_COMPILER + build-type "
                "cache that project() must consume."
            ),
            "Signal": (
                "[Input] Wrong-order use of DEAL_II_INITIALIZE_CACHED_VARIABLES "
                "fails with FATAL_ERROR literal text "
                "'DEAL_II_INITIALIZE_CACHED_VARIABLES can only be called in "
                "external projects after the inclusion of deal.IIConfig.cmake. "
                "It is not intended for internal use.' "
                "Two subtle silent side effects: "
                "(1) If user's -DCMAKE_BUILD_TYPE=Debug doesn't match the "
                "dealii install's DEAL_II_BUILD_TYPE (e.g. user wants Debug "
                "but dealii built Release-only), the macro FORCEs "
                "CMAKE_BUILD_TYPE to a valid mode and emits a banner "
                "starting with '#  WARNING:' and the literal "
                "'CMAKE_BUILD_TYPE was forced to'. "
                "(2) The macro WIPES user-set CMAKE_CXX_FLAGS / "
                "CMAKE_CXX_FLAGS_DEBUG / CMAKE_CXX_FLAGS_RELEASE to empty "
                "strings — any -O3 / -march=native / etc. set BEFORE the "
                "macro is lost. To customise flags, set them AFTER calling "
                "the macro. (File walk macro_deal_ii_initialize_cached_variables.cmake "
                "2026-06-02.)"
            ),
        },
        "DEAL_II_SETUP_TARGET": {
            "signature": (
                "DEAL_II_SETUP_TARGET(<target> [DEBUG|RELEASE])"),
            "purpose": (
                "Append deal.II's INCLUDE_DIRECTORIES, COMPILE_FLAGS, "
                "LINK_FLAGS, COMPILE_DEFINITIONS, and link interface "
                "to <target>. Must be called AFTER add_executable / "
                "add_library on <target> and AFTER "
                "find_package(deal.II). Source: "
                "cmake/macros/macro_deal_ii_setup_target.cmake."),
            "Signal": (
                "[Input] DEAL_II_SETUP_TARGET has FIVE distinct "
                "failure / silent-surprise modes users routinely hit: "
                "(1) Called BEFORE find_package(deal.II) — "
                "FATAL_ERROR with literal text "
                "'DEAL_II_SETUP_TARGET can only be called in external "
                "projects after the inclusion of deal.IIConfig.cmake. "
                "It is not intended for internal use.' (gated on "
                "DEAL_II_PROJECT_CONFIG_INCLUDED). "
                "(2) CMAKE_BUILD_TYPE is set to something other than "
                "'Debug' or 'Release' (e.g. RelWithDebInfo, "
                "MinSizeRel, empty) and no explicit DEBUG|RELEASE arg "
                "is passed — FATAL_ERROR 'DEAL_II_SETUP_TARGET cannot "
                "determine DEBUG, or RELEASE flavor for target. "
                "CMAKE_BUILD_TYPE \"<X>\" is neither equal to "
                "\"Debug\", nor \"Release\"'. Common with Ninja "
                "multi-config / Visual Studio defaults. Fix: set "
                "CMAKE_BUILD_TYPE explicitly OR call "
                "DEAL_II_SETUP_TARGET(<target> DEBUG|RELEASE). "
                "(3) DANGEROUS SILENT DOWNGRADE: if user requests "
                "DEBUG (via arg or CMAKE_BUILD_TYPE=Debug) but the "
                "INSTALLED deal.II was built RELEASE-only "
                "(DEAL_II_BUILD_TYPE doesn't contain 'Debug'), the "
                "macro silently overrides `_build` to RELEASE without "
                "any warning. The user's debug-flagged target links "
                "against the optimized deal.II — Assert macros are "
                "compiled out, the debugger steps through optimized "
                "frames, and bug-hunters waste hours wondering why "
                "DealiiAssert isn't firing. Fix: rebuild deal.II "
                "with -DCMAKE_BUILD_TYPE=DebugRelease, or accept the "
                "release build. "
                "(4) Second arg is anything besides empty / DEBUG / "
                "RELEASE — FATAL_ERROR 'invalid second argument. "
                "Valid arguments are (empty), DEBUG, or RELEASE'. "
                "Common: passing 'Debug' (lowercase d, capitalized "
                "rest) thinking the macro is case-insensitive — it "
                "isn't (string MATCHES is case-sensitive). "
                "(5) Target is an OBJECT_LIBRARY — the link-"
                "interface block (TARGET_LINK_LIBRARIES) is SKIPPED "
                "silently (gated on `_type != OBJECT_LIBRARY`). The "
                "object library compiles fine but linking it into "
                "the final executable without explicitly "
                "TARGET_LINK_LIBRARIES(<exe> ${DEAL_II_TARGET_<build>}) "
                "yields undefined-symbol errors at link time. "
                "Plus: this is a CMake MACRO (not FUNCTION), so "
                "internal vars _build, _flags, _cuda_flags, "
                "_cxx_flags LEAK into the caller scope and can "
                "shadow user-set variables of the same names. "
                "(File walk macro_deal_ii_setup_target.cmake "
                "2026-06-03.)"),
        },
        "DEAL_II_INVOKE_AUTOPILOT": {
            "signature": "DEAL_II_INVOKE_AUTOPILOT()",
            "purpose": (
                "All-in-one shortcut for the canonical tutorial template. "
                "Reads CALLER-SET variables, creates an executable + 8 "
                "custom CMake targets."),
            "caller_variables": {
                "TARGET":         "REQUIRED — project + executable name",
                "TARGET_SRC":     "REQUIRED — list of .cc source files",
                "TARGET_RUN":     "optional — `make run` command (defaults to ${TARGET}); empty disables",
                "CLEAN_UP_FILES": "optional — files removed by runclean/distclean (defaults to glob `*.log *.gmv *.gnuplot *.gpl *.eps *.pov *.vtk *.ucd *.d2`)",
            },
            "targets_created": {
                "run":            "compile + execute (gated by TARGET_RUN!=empty)",
                "sign":           "Mac OSX codesign (requires OSX_CERTIFICATE_NAME)",
                "debug":          "switch CMAKE_BUILD_TYPE to Debug (only if DEAL_II_BUILD_TYPE matches Debug)",
                "release":        "switch CMAKE_BUILD_TYPE to Release (only if matches Release)",
                "runclean":       "remove output files matching CLEAN_UP_FILES glob",
                "distclean":      "remove CMakeCache.txt, CMakeFiles/, Makefile, build.ninja, .ninja_* (NUKES build state)",
                "strip_comments": "IRREVERSIBLE: runs `perl -pi -e 's#^[ \\t]*//.*\\n##g;' ${TARGET_SRC}` in place",
                "info":           "print usage message",
            },
            "Signal": (
                "[API] DEAL_II_INVOKE_AUTOPILOT creates a target named "
                "`strip_comments` that the macro itself documents as "
                "'irreversible'. The implementation is "
                "ADD_CUSTOM_TARGET(strip_comments COMMAND perl -pi -e "
                "'s#^[ \\t]*//.*\\n##g;' ${TARGET_SRC}) — Perl rewrites "
                "the SOURCE FILES in place, dropping all // line comments. "
                "There is NO confirmation prompt and NO backup. Run "
                "`make strip_comments` once and your tutorial's comments "
                "are gone. Also: `make distclean` nukes CMakeCache.txt + "
                "Makefile + .ninja_* — far more aggressive than `make "
                "clean`. Default CLEAN_UP_FILES glob may delete unrelated "
                "*.log / *.vtk files the user produced manually in the "
                "build dir. (File walk "
                "macro_deal_ii_invoke_autopilot.cmake 2026-06-03.)"
            ),
        },
        "DEAL_II_PICKUP_TESTS": {
            "signature": "DEAL_II_PICKUP_TESTS()",
            "purpose": (
                "Glob *.output files in the current source dir, parse each "
                "filename for feature constraints, and register each as a "
                "ctest case via DEAL_II_ADD_TEST(<dir-name>, <test>, <output>)."),
            "test_filename_grammar": (
                "Tests are identified by a *.output file. The filename can "
                "encode constraints between dots: "
                "  <name>.with_<feature><op><value>.output "
                "Operators: = .ge. .le. .geq. .leq. "
                "Values: boolean (on/off/yes/no/true/false) for = comparisons; "
                "        version number (e.g. 11.2) for .ge./.le./.geq./.leq. "
                "Examples: "
                "  mytest.with_petsc=on.output           — needs DEAL_II_WITH_PETSC "
                "  mytest.with_cuda.geq.11.2.output      — needs CUDA >= 11.2 "
                "  mytest.mpirun=4.output                — runs under mpirun -np 4; "
                "                                          SKIPPED if DEAL_II_WITH_MPI=OFF"),
            "env_vars": {
                "TEST_PICKUP_REGEX": "regex filter on '<category>/<test>' names; empty = catchall (default)",
                "TEST_TIME_LIMIT":   "wall clock limit per test in seconds (default 600)",
                "DIFF_DIR":          "hint path for diff executable",
                "NUMDIFF_DIR":       "hint path for numdiff executable (preferred over diff)",
                "TEST_LIBRARIES":    "extra libs/targets to link against",
                "TEST_LIBRARIES_DEBUG / _RELEASE":  "per-config link list",
                "TEST_TARGET":       "test target name (or _DEBUG / _RELEASE pair)",
            },
            "Signal": (
                "[Input] DEAL_II_PICKUP_TESTS has three FATAL_ERROR traps: "
                "(1) calling it outside an external project (DEAL_II_PROJECT_CONFIG_INCLUDED "
                "not set) — literal 'DEAL_II_PICKUP_TESTS can only be called in "
                "external (test sub-)projects after the inclusion of "
                "deal.IIConfig.cmake'; "
                "(2) neither diff nor numdiff on PATH — 'Could not find diff "
                "or numdiff. One of those are required'; "
                "(3) numdiff IS a symlink to diff (common on minimal installs) — "
                "macro runs a relative-tolerance probe and dies with 'The "
                "detected numdiff executable was not able to pass a simple "
                "relative tolerance test. This usually means that either "
                "numdiff was misconfigured or that it is a symbolic link to "
                "diff.' Workaround: install real numdiff from "
                "savannah.gnu.org/projects/numdiff or set NUMDIFF_DIR. "
                "Additionally: an unknown `with_<feature>` in a test filename "
                "(neither DEAL_II_WITH_<F> nor DEAL_II_<F> defined) silently "
                "drops the test from ctest's discovery — easy way to lose "
                "tests after a dealii config option rename. (File walk "
                "macro_deal_ii_pickup_tests.cmake 2026-06-03.)"
            ),
        },
        "DEAL_II_QUERY_GIT_INFORMATION": {
            "description": (
                "Populate GIT_BRANCH / GIT_REVISION / GIT_SHORTREV / "
                "GIT_TAG from the source dir's .git metadata. The "
                "macro has an OPTIONAL positional PREFIX argument: "
                "called as DEAL_II_QUERY_GIT_INFORMATION() variables "
                "are unprefixed; called as "
                "DEAL_II_QUERY_GIT_INFORMATION(MYAPP) they become "
                "MYAPP_GIT_BRANCH etc. Source: "
                "cmake/macros/macro_deal_ii_query_git_information.cmake."),
            "Signal": (
                "[Output] Four sharp edges users routinely hit with "
                "DEAL_II_QUERY_GIT_INFORMATION: "
                "(1) The default variables are UNPREFIXED — GIT_BRANCH, "
                "GIT_REVISION, GIT_SHORTREV, GIT_TAG. There is NO "
                "DEAL_II_GIT_* prefix unless the user explicitly "
                "passes a prefix argument; the prefix is "
                "${ARGN}_-style and lives in the macro body. "
                "(2) The variable set is GIT_BRANCH / GIT_REVISION / "
                "GIT_SHORTREV / GIT_TAG — there is NO GIT_TIMESTAMP "
                "and NO GIT_COMMIT_DATE; older documentation that "
                "claims a _TIMESTAMP slot is wrong. "
                "(3) If ${CMAKE_SOURCE_DIR}/.git/HEAD doesn't exist "
                "(tarball install, shallow CI checkout without .git/, "
                "or downstream app embedded in a non-git workspace) "
                "the macro is a SILENT NO-OP — no warning, no error, "
                "all four variables remain unset. Subsequent "
                "configure_file expansions on ${GIT_REVISION} produce "
                "the empty string. "
                "(4) GIT_TAG depends on the auxiliary shell script "
                "${DEAL_II_SHARE_RELDIR}/scripts/get_latest_tag.sh; "
                "if that script isn't on disk (some packagers strip "
                "it), GIT_TAG is silently left unset with only a "
                "MESSAGE(STATUS) line — easy to miss in CMake "
                "configure noise. "
                "(5) In detached-HEAD state `git symbolic-ref HEAD` "
                "returns non-zero, so GIT_BRANCH is NOT populated "
                "even though .git/HEAD exists — common in CI runs "
                "that checkout a tag or specific commit. "
                "(File walk macro_deal_ii_query_git_information.cmake "
                "2026-06-03.)"),
        },
    },
}
