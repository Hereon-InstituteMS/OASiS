"""Heat equation templates for deal.II.

Based on deal.II tutorial step-26 (transient) and steady-state variants.
"""


def _heat_2d_transient(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-26 pattern (simplified).
    """
    refinements = params.get("refinements", 4)
    n_steps = params.get("n_steps", 20)
    dt = params.get("dt", 0.05)
    return f'''\
/* Transient heat equation on unit square — deal.II
 * dT/dt - laplacian(T) = 0, prescribed temperatures on boundaries
 * Backward Euler time stepping.
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
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iostream>

using namespace dealii;

// Boundary value function — returns prescribed temperature on left edge
template <int dim>
class LeftBoundary : public Function<dim>
{{
public:
  virtual double value(const Point<dim> &p, const unsigned int) const override
  {{
    return (p[0] < 1e-10) ? 1.0 : 0.0;
  }}
}};

int main()
{{
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, Point<2>(0,0), Point<2>(1,1), true);
  triangulation.refine_global({refinements});

  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "Heat DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> mass_matrix, stiffness_matrix, system_matrix;
  mass_matrix.reinit(sparsity_pattern);
  stiffness_matrix.reinit(sparsity_pattern);
  system_matrix.reinit(sparsity_pattern);

  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> old_solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  // Assemble mass and stiffness
  MatrixTools::create_mass_matrix(dof_handler, QGauss<2>(fe.degree+1), mass_matrix);
  MatrixTools::create_laplace_matrix(dof_handler, QGauss<2>(fe.degree+1), stiffness_matrix);

  // Initial condition: T=0
  solution = 0;

  double dt = {dt};
  int n_steps = {n_steps};

  for (int step = 0; step < n_steps; ++step)
    {{
      old_solution = solution;

      // System: (M + dt*K) * T^(n+1) = M * T^n
      system_matrix.copy_from(mass_matrix);
      system_matrix.add(dt, stiffness_matrix);
      mass_matrix.vmult(system_rhs, old_solution);

      // BCs: prescribed temperature on left (boundary_id=0) and right (boundary_id=1)
      std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(dof_handler, 0,
        Functions::ConstantFunction<2>(1.0), boundary_values);
      VectorTools::interpolate_boundary_values(dof_handler, 1,
        Functions::ZeroFunction<2>(), boundary_values);
      MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

      SolverControl solver_control(1000, 1e-12);
      SolverCG<Vector<double>> solver(solver_control);
      solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
    }}

  std::cout << "Heat at t=" << n_steps * dt << ": min(T)="
            << *std::min_element(solution.begin(), solution.end())
            << ", max(T)=" << *std::max_element(solution.begin(), solution.end()) << std::endl;

  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "temperature");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}}
'''


def _heat_2d_steady(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    refinements = params.get("refinements", 5)
    T_left = params.get("T_left", 100.0)
    T_right = params.get("T_right", 0.0)
    return f'''\
/* Steady heat conduction on unit square — deal.II
 * -laplacian(T) = 0, T={T_left} on left, T={T_right} on right
 * Insulated top/bottom (natural Neumann BC).
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
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int main()
{{
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, Point<2>(0,0), Point<2>(1,1), true);
  triangulation.refine_global({refinements});

  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "Heat DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  // Assemble Laplace operator
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
          }}
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {{
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }}
    }}

  // BCs: T={T_left} on left (boundary_id=0), T={T_right} on right (boundary_id=1)
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ConstantFunction<2>({T_left}), boundary_values);
  VectorTools::interpolate_boundary_values(dof_handler, 1,
    Functions::ConstantFunction<2>({T_right}), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "Heat steady: min(T)="
            << *std::min_element(solution.begin(), solution.end())
            << ", max(T)=" << *std::max_element(solution.begin(), solution.end()) << std::endl;

  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "temperature");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}}
'''


def _heat_rectangle(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    refinements = params.get("refinements", 3)
    lx = params.get("lx", 2.0)
    ly = params.get("ly", 1.0)
    T_left = params.get("T_left", 100.0)
    T_right = params.get("T_right", 0.0)
    nx = int(lx * 4)
    ny = int(ly * 4)
    return f'''\
/* Heat conduction on [{lx}x{ly}] rectangle — deal.II
 * T={T_left} on left, T={T_right} on right
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
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int main()
{{
  Triangulation<2> triangulation;
  GridGenerator::subdivided_hyper_rectangle(triangulation,
    {{{nx}u, {ny}u}}, Point<2>(0, 0), Point<2>({lx}, {ly}), true /*colorize*/);
  triangulation.refine_global({refinements});

  FE_Q<2> fe(1);
  DoFHandler<2> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "Heat DOFs: " << dof_handler.n_dofs() << std::endl;

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
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            cell_matrix(i, j) += fe_values.shape_grad(i, q) *
                                 fe_values.shape_grad(j, q) *
                                 fe_values.JxW(q);
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
    Functions::ConstantFunction<2>({T_left}), boundary_values);
  VectorTools::interpolate_boundary_values(dof_handler, 1,
    Functions::ConstantFunction<2>({T_right}), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  SolverControl solver_control(5000, 1e-10);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "Heat: min(T)=" << *std::min_element(solution.begin(), solution.end())
            << ", max(T)=" << *std::max_element(solution.begin(), solution.end()) << std::endl;

  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "temperature");
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Heat equation: transient (step-26) with AMR, steady-state, time-stepping",
    "tutorial_steps": ["step-26 (transient + AMR)", "step-86 (SUNDIALS ARKode)"],
    "function_space": "FE_Q<dim>(1)",
    "solver": "CG + SSOR for each time step. SUNDIALS for adaptive time stepping",
    "time_stepping": "Theta method (0=forward Euler, 0.5=Crank-Nicolson, 1=backward Euler)",
    "elements": [
        "FE_Q<dim>(degree)            — continuous Lagrange; degree=1 is default for transient heat",
        "FE_Q_Hierarchical<dim>(degree) — for p-adaptive refinement with smooth coefficient transitions",
        "FE_DGQ<dim>(degree)          — DG variant; useful for sharp-front problems (welding, phase change)",
        "FE_SimplexP<dim>(degree)     — Lagrange on simplex (triangle/tet) meshes from Gmsh; available in deal.II ≥ 9.3",
    ],
    "mesh_generators": [
        "GridGenerator::hyper_cube(tria, a, b)               — unit cube/square; default heat-equation test domain",
        "GridGenerator::hyper_rectangle(tria, p1, p2)        — non-square; e.g. {0,0}-{L,H} for thin plates",
        "GridGenerator::hyper_L(tria, a, b)                  — L-shaped with re-entrant corner; tests adaptive refinement near singularity",
        "GridGenerator::cylinder(tria, radius, half_length)  — radial heat-equation problems (heat-pipe, cylindrical bar)",
        "GridGenerator::hyper_shell(tria, center, in, out)   — annular heat conduction (pipe insulation)",
        "GridGenerator::plate_with_a_hole(tria, ...)         — local thermal stress concentration via temperature gradient",
        "GridGenerator::extrude_triangulation(t_in, n_slices, h, t_out) — extrude a 2D heat-eq mesh into 3D, layer count = n_slices",
    ],
    "solvers": [
        "SolverCG<>                   — Heat-equation stiffness K and mass matrix M are SPD; CG works for all theta in (0,1]",
        "SolverGMRES<>                — needed only when the time-step matrix becomes non-symmetric (rare; happens with full-coupled nonlinear source terms)",
        "SUNDIALS::ARKode             — adaptive multi-step time integrator; step-86. Use when wall-time / step-count matters and dt is hard to estimate a priori",
    ],
    "preconditioners": [
        "PreconditionSSOR             — works on (M + dt*theta*K) at each step for moderate dt",
        "PreconditionAMG / BoomerAMG  — needed for stiff systems (small dt with large K eigenvalues, or wide dynamic range in thermal diffusivity)",
        "Use a SINGLE preconditioner factorisation across multiple time steps when dt and the mesh are fixed — re-building per step is the most common performance pitfall in transient heat code",
    ],
    "pitfalls": [
        "[Syntax] Time-step system is (M + dt*theta*K). Assemble M "
        "and K once at startup, build the LHS each step. Re-assembling "
        "M and K every time step is the most common transient-heat "
        "performance bug. Signal: per-step wall time scales with "
        "ndof^2 instead of ndof*log(ndof) for AMG.",
        "[Physics] RHS at each step is "
        "M*u_old - dt*(1-theta)*K*u_old + dt*theta*f_new + "
        "dt*(1-theta)*f_old. Forgetting the (1-theta) terms gives a "
        "fully-implicit scheme regardless of the theta value; "
        "Crank-Nicolson (theta=0.5) then degrades to backward Euler "
        "and the time-discretisation error scales O(dt) not O(dt^2). "
        "Signal: temperature convergence on a manufactured solution "
        "is first-order in dt even though you set theta=0.5.",
        "[API] For AMR in time: interpolate the solution to the new "
        "mesh via SolutionTransfer between refine_grid() and "
        "distribute_dofs(). Skipping SolutionTransfer gives a "
        "zero solution on the new mesh and the next time step "
        "evolves from zero — looks like a sudden cool-down. "
        "Signal: a sharp temperature drop coinciding with the "
        "first AMR-step.",
        "[Numerical] Non-zero initial conditions must be set via "
        "interpolate (for piecewise polynomial ICs) or project (for "
        "general functions). Setting node values directly bypasses "
        "the AffineConstraints and gives an IC inconsistent with "
        "the boundary conditions. Signal: temperature jumps at "
        "Dirichlet-boundary nodes between t=0 and the first "
        "time-step solution.",
        "[Numerical] theta=0 (forward Euler) requires "
        "dt < ~h^2 / (2*alpha) where alpha is the thermal "
        "diffusivity — fine mesh + small alpha makes this dt tiny. "
        "Use theta=0.5 (Crank-Nicolson, 2nd-order, unconditionally "
        "stable) or theta=1 (backward Euler, 1st-order, "
        "unconditionally stable) for any production run. "
        "Signal: forward Euler with too-large dt produces "
        "oscillating solution that grows exponentially.",
    ],
}
