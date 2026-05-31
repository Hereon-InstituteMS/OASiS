"""Linear elasticity templates for deal.II.

Based on deal.II tutorial step-8.
"""


def _elasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-8.
    """
    refinements = params.get("refinements", 4)
    E_val = params.get("E", 1000.0)
    nu_val = params.get("nu", 0.3)
    lx = params.get("lx", 10.0)
    ly = params.get("ly", 1.0)
    nx_cells = int(lx * 4)
    ny_cells = max(int(ly * 4), 1)
    return f'''\
/* Linear elasticity — based on deal.II step-8
 * 2D plane strain, fixed left edge, body force pointing down
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
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
#include <deal.II/fe/component_mask.h>
#include <fstream>
#include <iostream>

using namespace dealii;

// Body force — set direction and magnitude for your problem
template <int dim>
class BodyForce : public Function<dim>
{{
public:
  BodyForce() : Function<dim>(dim) {{}}
  virtual void vector_value(const Point<dim> &, Vector<double> &values) const override
  {{
    values    = 0;
    values[1] = -1.0; // downward
  }}
}};

int main()
{{
  const int dim = 2;

  // Domain
  Triangulation<dim> triangulation;
  GridGenerator::subdivided_hyper_rectangle(triangulation,
    {{{nx_cells}u, {ny_cells}u}}, Point<dim>(0, 0), Point<dim>({lx}, {ly}), true /*colorize*/);

  FESystem<dim> fe(FE_Q<dim>(1), dim);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  std::cout << "Number of DOFs: " << dof_handler.n_dofs() << std::endl;

  // Sparsity
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  // Material
  const double E  = {E_val};
  const double nu = {nu_val};
  const double mu     = E / (2.0 * (1.0 + nu));
  const double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

  // Assemble
  QGauss<dim> quadrature(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature,
    update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dpc = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dpc, dpc);
  Vector<double>     cell_rhs(dpc);
  std::vector<types::global_dof_index> local_dof_indices(dpc);

  BodyForce<dim> body_force;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {{
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {{
          // Body force at quadrature point
          Vector<double> f_val(dim);
          body_force.vector_value(fe_values.quadrature_point(q), f_val);

          for (unsigned int i = 0; i < dpc; ++i)
            {{
              const unsigned int ci = fe.system_to_component_index(i).first;

              for (unsigned int j = 0; j < dpc; ++j)
                {{
                  const unsigned int cj = fe.system_to_component_index(j).first;

                  cell_matrix(i, j) +=
                    (fe_values.shape_grad(i, q)[ci] *
                     fe_values.shape_grad(j, q)[cj] * lambda
                     +
                     fe_values.shape_grad(i, q) *
                     fe_values.shape_grad(j, q) *
                     (ci == cj ? mu : 0.0)
                     +
                     fe_values.shape_grad(i, q)[cj] *
                     fe_values.shape_grad(j, q)[ci] * mu
                    ) * fe_values.JxW(q);
                }}

              cell_rhs(i) += fe_values.shape_value(i, q) * f_val[ci] *
                             fe_values.JxW(q);
            }}
        }}

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dpc; ++i)
        {{
          for (unsigned int j = 0; j < dpc; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }}
    }}

  // BC: fix left edge (x=0), all components
  std::map<types::global_dof_index, double> boundary_values;
  // Left boundary = id 0 for hyper_rectangle
  VectorTools::interpolate_boundary_values(dof_handler,
    0, Functions::ZeroFunction<dim>(dim), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
    system_matrix, solution, system_rhs);

  // Solve
  SolverControl solver_control(5000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "Solver converged in " << solver_control.last_step()
            << " iterations." << std::endl;

  // Output
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  std::vector<std::string> names = {{"ux", "uy"}};
  data_out.add_data_vector(solution, names);
  data_out.build_patches();

  std::ofstream output("result.vtu");
  data_out.write_vtu(output);

  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}}
'''


def _elasticity_thick_beam(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    lx = params.get("lx", 5.0)
    ly = params.get("ly", 2.0)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    nx = int(lx * 8)
    ny = int(ly * 8)
    return f'''\
/* Linear elasticity on {lx}x{ly} domain — deal.II
 * Fixed left edge, body force.
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
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

template <int dim>
class BodyForce : public Function<dim>
{{
public:
  BodyForce() : Function<dim>(dim) {{}}
  virtual void vector_value(const Point<dim> &, Vector<double> &values) const override
  {{
    values = 0;
    values[1] = -1.0;
  }}
}};

int main()
{{
  const int dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::subdivided_hyper_rectangle(triangulation,
    {{{nx}u, {ny}u}}, Point<dim>(0, 0), Point<dim>({lx}, {ly}), true /*colorize*/);

  FESystem<dim> fe(FE_Q<dim>(1), dim);
  DoFHandler<dim> dof_handler(triangulation);
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

  const double E  = {E};
  const double nu = {nu};
  const double mu     = E / (2.0 * (1.0 + nu));
  const double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

  QGauss<dim> quadrature(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature,
    update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dpc = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dpc, dpc);
  Vector<double> cell_rhs(dpc);
  std::vector<types::global_dof_index> local_dof_indices(dpc);
  BodyForce<dim> body_force;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {{
      fe_values.reinit(cell);
      cell_matrix = 0; cell_rhs = 0;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {{
          Vector<double> f_val(dim);
          body_force.vector_value(fe_values.quadrature_point(q), f_val);
          for (unsigned int i = 0; i < dpc; ++i)
            {{
              const unsigned int ci = fe.system_to_component_index(i).first;
              for (unsigned int j = 0; j < dpc; ++j)
                {{
                  const unsigned int cj = fe.system_to_component_index(j).first;
                  cell_matrix(i, j) +=
                    (fe_values.shape_grad(i, q)[ci] *
                     fe_values.shape_grad(j, q)[cj] * lambda
                     + fe_values.shape_grad(i, q) *
                       fe_values.shape_grad(j, q) * (ci == cj ? mu : 0.0)
                     + fe_values.shape_grad(i, q)[cj] *
                       fe_values.shape_grad(j, q)[ci] * mu
                    ) * fe_values.JxW(q);
                }}
              cell_rhs(i) += fe_values.shape_value(i, q) * f_val[ci] * fe_values.JxW(q);
            }}
        }}
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dpc; ++i)
        {{
          for (unsigned int j = 0; j < dpc; ++j)
            system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }}
    }}

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<dim>(dim), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  SolverControl solver_control(5000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "Solver: " << solver_control.last_step() << " iterations" << std::endl;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  std::vector<std::string> names = {{"ux", "uy"}};
  data_out.add_data_vector(solution, names);
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Linear elasticity (step-8, step-17 parallel, step-18 quasi-static)",
    "tutorial_steps": ["step-8 (basic)", "step-17 (MPI parallel)", "step-18 (incremental)"],
    "function_space": "FESystem<dim>(FE_Q<dim>(1), dim) — vector Lagrange",
    "solver": "CG + PreconditionSSOR (serial), SolverCG + BoomerAMG (parallel)",
    # ── Structured catalog keys (encoded 2026-05-31 from Layer A
    #    scan vs catalog diff — see data/postmortems/
    #    dealii-elasticity-catalog-structure.json). Each entry pairs
    #    the upstream class name with a short note on why a user
    #    would choose it for elasticity specifically — generic
    #    "FE_Q is a Lagrange element" notes go in the deal.II
    #    documentation, this catalog is physics-keyed.
    "elements": [
        "FE_Q<dim>(degree)            — continuous Lagrange, the default; wrap in FESystem<dim>(FE_Q<dim>(degree), dim) for vector-valued u",
        "FE_Q_Bubbles<dim>(degree)    — bubble-enriched Lagrange; useful when the linear elasticity is paired with an incompressible Stokes pressure (avoids LBB instability without dropping to mixed elements)",
        "FE_Q_Hierarchical<dim>(degree) — hierarchical basis; preferred for p-adaptive refinement so coarse-level DoFs survive a polynomial-degree change",
        "FE_Q_DG0<dim>(degree)        — continuous Lagrange enriched with one piecewise-constant DG mode per cell; mass-matrix lumping",
        "FE_Bernstein<dim>(degree)    — Bernstein-Bezier basis; better-conditioned mass matrix at high p, used in IGA-adjacent workflows",
        "FE_RannacherTurek<dim>()     — P1-non-conforming on quads/hexes; locking-free for nearly-incompressible elasticity (Poisson ratio approaching 0.5)",
        "FE_Nothing<dim>()            — placeholder element with zero DoFs; used inside FESystem on subdomains where displacement is fully constrained",
        "FESystem<dim>(...)           — the vector wrapper itself; do not forget it (a bare FE_Q gives scalar u, not the elasticity displacement field)",
    ],
    "mesh_generators": [
        "GridGenerator::hyper_cube(tria, a, b)                    — unit-cube / square in [a,b]^dim; smallest reproducer for any elasticity problem",
        "GridGenerator::hyper_rectangle(tria, p1, p2)             — axis-aligned box from p1 to p2; cantilever beams (typical {0,0}-{L,h})",
        "GridGenerator::subdivided_hyper_rectangle(tria, repetitions, p1, p2) — same box but with per-direction element counts; aspect-ratio control",
        "GridGenerator::plate_with_a_hole(tria, ...)              — Kirsch problem, classic stress-concentration-factor test",
        "GridGenerator::hyper_L(tria, a, b)                       — L-shaped domain with re-entrant corner; standard benchmark for singularity-driven adaptive refinement",
        "GridGenerator::hyper_cube_with_cylindrical_hole(tria, inner_radius, outer_radius) — generalisation of plate_with_a_hole to 3D",
        "GridGenerator::cylinder(tria, radius, half_length)       — circular cylinder; useful for axisymmetric beam tests",
        "GridGenerator::hyper_shell(tria, center, inner, outer)   — spherical / cylindrical shell; pressure vessels",
        "GridGenerator::merge_triangulations(t1, t2, result)      — combine two domains; needed for inclusion / dissimilar-material problems",
    ],
    "preconditioners": [
        "PreconditionSSOR<>          — serial default for symmetric positive-definite elasticity stiffness; cheap, works well up to ~10^5 DoFs",
        "PreconditionAMG / BoomerAMG — parallel AMG for >10^5 DoFs; via TrilinosWrappers (BoomerAMG is HYPRE through Trilinos)",
        "PreconditionJacobi          — diagonal scaling only; useful baseline when debugging convergence stall",
        "PreconditionChebyshev       — for smoothing inside multigrid; not a top-level preconditioner for direct CG use",
    ],
    "solvers": [
        "SolverCG<>                  — symmetric positive-definite stiffness; the canonical choice for linear elasticity",
        "SolverGMRES<>               — non-symmetric (rare in pure elasticity but needed when coupling with advection terms)",
        "SolverMinRes<>              — symmetric but indefinite; mixed displacement-pressure formulations",
    ],
    "pitfalls": [
        "[Syntax] Use FEValuesExtractors::Vector(0) for velocity-like "
        "access in assembly. Plain fe_values.shape_value(i,q) returns a "
        "scalar for each component — the elasticity strain tensor needs "
        "the vector extractor. Signal: assembly compiles but stiffness "
        "matrix is rank-deficient (det == 0 on every cell).",
        "[Physics] Lame parameters: mu = E/(2(1+nu)), "
        "lambda = E*nu/((1+nu)(1-2nu)). Computing one and forgetting "
        "the other (or swapping their roles in the bilinear form) is "
        "a common silent error. Signal: displacement field is off by a "
        "constant factor independent of mesh refinement.",
        "[Physics] For plane stress, modify lambda to "
        "lambda_star = 2*mu*lambda/(2*mu+lambda). Code: "
        "`double lam_star = 2*mu*lam / (2*mu + lam);`. Forgetting this "
        "is plane STRAIN, not plane STRESS — the response is too stiff "
        "in 2D. Signal: tip deflection on a cantilever 2D beam is "
        "smaller than the 1D Euler-Bernoulli prediction by ~30%.",
        "[Syntax] Body force is added to cell_rhs via "
        "`fe_values[velocities].value(i,q)`. Using fe_values.shape_value "
        "alone gives the wrong scalar component. Signal: only the first "
        "vector component sees the body force.",
        "[Integration] deal.II 2D ONLY reads QUADS from Gmsh — no "
        "triangles. Always set `gmsh.option.setNumber"
        "('Mesh.RecombineAll', 1)` to produce quads. Signal: "
        "`ExcMessage` from grid_in.cc that the mesh contains "
        "unsupported cell type.",
        "[API] Gmsh element order != FE polynomial degree. ALWAYS use "
        "first-order geometry elements in Gmsh (default). The FE degree "
        "(Q1, Q2) is set in the C++ code via `FE_Q<dim>(degree)`. Do "
        "NOT set `Mesh.ElementOrder=2` in Gmsh — deal.II cannot read "
        "second-order geometry elements (Tri6, Quad9). Signal: GridIn "
        "reports 'unsupported cell type 25' (Tri6) or '28' (Quad9).",
        # New entries shipped with this encoding pass — each lifted
        # from concrete deal.II tutorials or upstream issues.
        "[Numerical] FE_Q<dim>(1) (linear elements) locks in "
        "nearly-incompressible elasticity (Poisson ratio nu close to "
        "0.5). Use FE_Q<dim>(2) (quadratic), FE_RannacherTurek<dim>() "
        "(P1-NC), or a mixed displacement-pressure formulation with "
        "FE_Q+FE_DGQ for pressure. Signal: tip deflection on a "
        "compressible beam converges with refinement, but converges to "
        "the wrong value as nu → 0.5 (volumetric locking).",
        "[API] FE_Nothing<dim>() inside an FESystem on a subdomain "
        "where displacement should be inactive does NOT skip "
        "assembly on those cells — it just makes the DoF count zero "
        "there. You still need to mark the cells with a manifold ID "
        "or a hp::DoFHandler<dim> active_fe_index switch. Signal: "
        "assembly enters cells you thought were 'off' and accumulates "
        "garbage into the rhs.",
        "[Numerical] Use SolverCG only when the stiffness matrix is "
        "symmetric positive-definite. Adding a Dirichlet penalty "
        "(rather than constraining the DoFs) keeps it SPD; using "
        "asymmetric face stabilisation or a Nitsche-style boundary "
        "term breaks symmetry — switch to SolverGMRES. Signal: SolverCG "
        "reports 'breakdown' or stalls at 1e-2 residual reduction.",
    ],
}
