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
    # ── Structured catalog keys, post-canonical-element refactor.
    #    Each entry is { class_name: physics-keyed applicability
    #    note }. The canonical class description / header / version
    #    gate lives in `backends.dealii.element_catalog.ELEMENTS`;
    #    `get_knowledge('linear_elasticity')` joins them at
    #    retrieval time. Encoded 2026-05-31; refactored after the
    #    senior-AI-scientist critic flagged per-physics duplication
    #    as the top-1 risk.
    "elements": {
        "FE_Q":
            "Default for linear elasticity. Wrap as "
            "FESystem<dim>(FE_Q<dim>(degree), dim) for the vector "
            "displacement field. degree=2 strongly preferred when "
            "the Poisson ratio approaches 0.5 to avoid volumetric "
            "locking; degree=1 is fine away from that limit.",
        "FE_Q_Bubbles":
            "Useful when elasticity is paired with an "
            "incompressible Stokes pressure (FSI / poromechanics) — "
            "the bubble enrichment helps LBB stability without "
            "dropping to a full mixed formulation.",
        "FE_Q_Hierarchical":
            "Use when running p-adaptive refinement during load "
            "stepping — coarse-level DoFs survive a polynomial-"
            "degree change.",
        "FE_Q_DG0":
            "Lumped-mass dynamics; the DG0 enrichment gives a "
            "diagonal block in the time-stepping matrix.",
        "FE_Bernstein":
            "Higher-p elasticity where mass-matrix conditioning "
            "matters (modal analysis, transient with implicit "
            "integration).",
        "FE_RannacherTurek":
            "Locking-free P1-non-conforming element on quads/hexes — "
            "the cheap alternative to degree-2 FE_Q for nearly-"
            "incompressible elasticity.",
        "FE_Nothing":
            "Placeholder inside FESystem on subdomains where "
            "displacement should be inactive (FSI solid region "
            "when modelling the fluid, or vice versa). Zero "
            "DoFs there, but you still need a manifold-id or "
            "hp::DoFHandler::active_fe_index switch to actually "
            "skip assembly.",
        "FESystem":
            "The vector wrapper. A bare FE_Q gives scalar u, "
            "NOT the displacement field — forgetting FESystem is "
            "the single most common deal.II elasticity bug.",
    },
    "mesh_generators": {
        "hyper_cube": "Smallest reproducer for any elasticity problem; the [0,1]^dim cube.",
        "hyper_rectangle": "Cantilever beams (typical {0,0}-{L,h}) — non-square aspect ratios.",
        "subdivided_hyper_rectangle": "Per-direction element counts; aspect-ratio control for beam tests.",
        "plate_with_a_hole": "Kirsch problem — classic stress-concentration-factor test.",
        "hyper_L": "Re-entrant corner — singularity-driven adaptive refinement benchmark.",
        "hyper_cube_with_cylindrical_hole": "3D generalisation of plate_with_a_hole.",
        "cylinder": "Axisymmetric beam tests, torsion problems.",
        "hyper_shell": "Pressure vessels — spherical / cylindrical shells.",
        "merge_triangulations": "Combine two domains for inclusion / dissimilar-material problems.",
    },
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
        "the vector extractor. Signal: assembly compiles but SolverCG "
        "reports 'iterative method failed to converge' (ExcSolverFail) "
        "because the stiffness matrix is rank-deficient; "
        "system_matrix.frobenius_norm() / system_rhs.l2_norm() is "
        "dominated by spurious diagonal entries.",
        "[Physics] Lame parameters: mu = E/(2(1+nu)), "
        "lambda = E*nu/((1+nu)(1-2nu)). Computing one and forgetting "
        "the other (or swapping their roles in the bilinear form) is "
        "a common silent error. Signal: tip displacement from "
        "DataOut differs from the Euler-Bernoulli reference "
        "u_max = P*L^3 / (3*E*I) by a constant factor (typically "
        "2-5x) that does NOT decrease with refinement; the ratio "
        "u_computed / u_reference is mesh-independent.",
        "[Physics] For plane stress, modify lambda to "
        "lambda_star = 2*mu*lambda/(2*mu+lambda). Code: "
        "`double lam_star = 2*mu*lam / (2*mu + lam);`. Forgetting this "
        "is plane STRAIN, not plane STRESS — the response is too stiff "
        "in 2D. Signal: tip deflection from `DataOut` on a cantilever "
        "2D beam is ~30% smaller than the Euler-Bernoulli reference "
        "P*L^3 / (3*E*I); the discrepancy is bias not noise — it "
        "persists under mesh refinement.",
        "[Syntax] Body force is added to cell_rhs via "
        "`fe_values[velocities].value(i,q)`. Using fe_values.shape_value "
        "alone gives the wrong scalar component. Signal: DataOut "
        "writes a displacement field where only the first component "
        "is non-zero (u_x has the expected gravity-driven profile, "
        "u_y is identically zero); per-component norm "
        "`solution.block(1).l2_norm() == 0` on a vector FESystem.",
        "[Integration] deal.II reads BOTH triangles and quads from "
        "Gmsh in 2D. `gmsh.option.setNumber"
        "('Mesh.RecombineAll', 1)` is a preference (tensor-product "
        "cells, matrix-free eligibility), NOT a requirement. "
        "Signal: after GridIn<2>::read_msh, count "
        "cell->n_vertices() over the active cells — 3 means "
        "triangles, 4 means quads; no exception is raised either "
        "way. Measured on deal.II 9.8.0-pre 2026-08-03 with "
        "hand-written MSH 2.2 files: a 4-quad mesh (Gmsh element "
        "type 3) reads back as 4 quad cells and an 8-triangle mesh "
        "(type 2) as 8 triangle cells. The real constraint is downstream: a simplex "
        "mesh needs FE_SimplexP / FE_SimplexDGP plus QGaussSimplex, "
        "and MatrixFree does not support simplices. (This entry "
        "asserted 'ONLY QUADS — no triangles', false since deal.II "
        "9.3 added simplex support.)",
        "[API] Gmsh element order != FE polynomial degree. ALWAYS use "
        "first-order geometry elements in Gmsh (default). The FE degree "
        "(Q1, Q2) is set in the C++ code via `FE_Q<dim>(degree)`. Do "
        "NOT set `Mesh.ElementOrder=2` in Gmsh — deal.II cannot read "
        "second-order geometry elements (Tri6, Quad9). Signal: GridIn "
        "reports \"The Element Identifier <9> is not supported in "
        "the deal.II library when reading meshes in 2 dimensions\" "
        "for Tri6, and the same message with <10> for Quad9 — those "
        "numbers are GMSH element types. Measured 2026-08-03 on "
        "9.8.0-pre. (This entry used to quote '25' and '28', which "
        "are VTK cell-type codes and never appear in a GridIn "
        "message.)",
        # New entries shipped with this encoding pass — each lifted
        # from concrete deal.II tutorials or upstream issues.
        "[Numerical] FE_Q<dim>(1) (linear elements) locks in "
        "nearly-incompressible elasticity (Poisson ratio nu close to "
        "0.5). Use FE_Q<dim>(2) (quadratic) or a mixed displacement-"
        "pressure formulation with FE_Q+FE_DGQ for pressure. "
        "Signal: tip deflection differs from a rich-space "
        "reference by a factor that GROWS as nu increases, and the "
        "gap does not close under mesh refinement. Measured on "
        "deal.II 9.8.0-pre 2026-08-03 — plane-strain cantilever "
        "8x1, 16x2 cells, body-force loaded, tip deflection "
        "against an FE_Q(3) reference on the SAME mesh: "
        "nu=0.3 -> FE_Q(1) 0.870 of reference, FE_Q(2) 0.998; "
        "nu=0.499 -> FE_Q(1) 0.0496 (95% too stiff), FE_Q(2) 0.962; "
        "nu=0.49999 -> FE_Q(1) 0.0215, FE_Q(2) 0.959. The locking "
        "claim and the FE_Q(2) cure both hold; the FE_Q(1) error at "
        "nu=0.3 is ~13%, not the '<5%' this entry used to promise. "
        "CAVEAT on the third option it used to recommend: "
        "FE_RannacherTurek is locking-free in theory but is NOT "
        "interpolatory (has_support_points() == false), so "
        "VectorTools::interpolate_boundary_values on "
        "FESystem(FE_RannacherTurek<dim>(), dim) SEGFAULTS on this "
        "Release build (rc=139, verified 2026-08-03; on a Debug "
        "build it would be an Assert). Use "
        "VectorTools::project_boundary_values instead — it returns "
        "8 boundary values on the same setup and works.",
        "[API] FE_Nothing<dim>() inside an FESystem on a subdomain "
        "where displacement should be inactive does NOT skip "
        "assembly on those cells — it just makes the DoF count zero "
        "there. You still need to mark the cells with a manifold ID "
        "or a hp::DoFHandler<dim> active_fe_index switch. Signal: "
        "DataOut shows non-zero residual values on cells that should "
        "be 'off'; `system_rhs.l2_norm()` is larger than expected "
        "even though `dof_handler.n_dofs_on_subdomain()` reports the "
        "FE_Nothing region has 0 DoFs.",
        "[Numerical] Use SolverCG only when the stiffness matrix is "
        "symmetric positive-definite. Adding a Dirichlet penalty "
        "(rather than constraining the DoFs) keeps it SPD; using "
        "asymmetric face stabilisation or a Nitsche-style boundary "
        "term breaks symmetry — switch to SolverGMRES. Signal: SolverCG "
        "reports 'breakdown' or stalls at 1e-2 residual reduction.",
    ],
}
