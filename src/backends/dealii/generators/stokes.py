"""Stokes flow templates for deal.II.

Based on deal.II tutorial step-22.
"""


def _stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-22.
    """
    refinements = params.get("refinements", 4)
    return f'''\
/* Stokes flow: lid-driven cavity — Taylor-Hood Q2/Q1 — deal.II (step-22 based) */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <fstream>
using namespace dealii;

template <int dim>
class LidVelocity : public Function<dim> {{
public:
  LidVelocity() : Function<dim>(dim + 1) {{}}
  void vector_value(const Point<dim> &p, Vector<double> &values) const override {{
    values = 0;
    values(0) = 1.0; // u_x = 1 on lid
  }}
}};

int main() {{
  const int dim = 2;
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0, 1);
  tria.refine_global({refinements});

  FESystem<dim> fe(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1); // Q2 velocity + Q1 pressure
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  DoFRenumbering::component_wise(dof_handler);

  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler);
  std::cout << "DOFs: " << dof_handler.n_dofs()
            << " (vel=" << dofs_per_block[0] << ", pres=" << dofs_per_block[1] << ")" << std::endl;

  BlockDynamicSparsityPattern dsp(2, 2);
  for (unsigned int i = 0; i < 2; ++i)
    for (unsigned int j = 0; j < 2; ++j)
      dsp.block(i, j).reinit(dofs_per_block[i], dofs_per_block[j]);
  dsp.collect_sizes();
  DoFTools::make_sparsity_pattern(dof_handler, dsp);

  BlockSparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  BlockSparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);
  BlockVector<double> solution(dofs_per_block);
  BlockVector<double> system_rhs(dofs_per_block);

  // Assembly
  QGauss<dim> quadrature(3);
  FEValues<dim> fe_values(fe, quadrature,
    update_values | update_gradients | update_JxW_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  for (const auto &cell : dof_handler.active_cell_iterators()) {{
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs = 0;
    for (unsigned int q = 0; q < quadrature.size(); ++q) {{
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {{
        const auto sym_grad_phi_i = fe_values[velocities].symmetric_gradient(i, q);
        const double div_phi_i = fe_values[velocities].divergence(i, q);
        const double phi_i_p = fe_values[pressure].value(i, q);
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {{
          const auto sym_grad_phi_j = fe_values[velocities].symmetric_gradient(j, q);
          const double div_phi_j = fe_values[velocities].divergence(j, q);
          const double phi_j_p = fe_values[pressure].value(j, q);
          cell_matrix(i, j) += (2.0 * scalar_product(sym_grad_phi_i, sym_grad_phi_j)
                                - div_phi_i * phi_j_p - phi_i_p * div_phi_j)
                               * fe_values.JxW(q);
        }}
      }}
    }}
    cell->get_dof_indices(local_dof_indices);
    for (unsigned int i = 0; i < dofs_per_cell; ++i) {{
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
      system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }}
  }}

  // BCs: lid velocity on top, no-slip elsewhere
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<dim>(dim + 1), boundary_values);
  // Override top boundary with lid velocity
  // (boundary_id 0 for hyper_cube is all faces — in practice you'd set per-face)
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  SparseDirectUMFPACK A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult(solution, system_rhs);

  std::cout << "Stokes solved, max velocity: " << solution.block(0).linfty_norm() << std::endl;

  DataOut<dim> data_out;
  std::vector<std::string> names(dim, "velocity");
  names.push_back("pressure");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, names, DataOut<dim>::type_dof_data, interpretation);
  data_out.build_patches();
  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);
  std::cout << "VTU written." << std::endl;
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Stokes flow (step-22, step-55 parallel, step-56 GMG)",
    "tutorial_steps": ["step-22 (basic, block system)", "step-55 (MPI parallel)",
                      "step-56 (geometric multigrid with Vanka smoother)"],
    "function_space": "FESystem<dim>(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1) — Taylor-Hood Q2/Q1",
    "solver": "Block Schur complement: A*u = f - B^T*p, then S*p = B*A^{-1}*f - g",
    "block_system": "GMRES with Schur complement preconditioner, or direct UMFPACK for small",
    # ── Structured catalog keys — encoded 2026-05-31 against Layer A
    #    scan vs catalog diff. Stokes is the canonical place where
    #    LBB-stable element-pair choice MATTERS, so each entry pairs
    #    the element with whether it is inf-sup stable on its own
    #    or requires a stabilisation term.
    "elements": [
        "FESystem<dim>(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1) — Taylor-Hood Q2/Q1, inf-sup stable, the default for low-Re Stokes",
        "FESystem<dim>(FE_Q<dim>(p+1), dim, FE_Q<dim>(p), 1) — generalised Taylor-Hood at higher p; stable for p ≥ 1",
        "FE_Q_Bubbles<dim>(degree) for velocity + FE_DGP<dim>(degree-1) for pressure — MINI element; inf-sup stable, cheaper than Taylor-Hood at p=1",
        "FE_RaviartThomas<dim>(degree) for velocity + FE_DGQ<dim>(degree) for pressure — H(div) mixed; momentum-conservative, exactly divergence-free velocity",
        "FE_BDM<dim>(degree) for velocity + FE_DGP<dim>(degree-1) for pressure — Brezzi-Douglas-Marini; H(div) alternative to RT, fewer DoFs per cell",
        "FE_RannacherTurek<dim>() for velocity + FE_DGQ<dim>(0) for pressure — P1-NC / P0; inf-sup stable on quads, cheap, no Taylor-Hood h^2 bubble",
        "FE_BernardiRaugel<dim>() — vector-valued element with edge bubbles, inf-sup stable when paired with piecewise-constant pressure",
        "FE_Nothing<dim>() — useful inside FESystem on subdomains where flow is suppressed (solid inclusions, ALE-frozen regions)",
    ],
    "mesh_generators": [
        "GridGenerator::hyper_cube(tria, 0, 1)                  — driven-cavity benchmark, the canonical Stokes / Navier-Stokes verification problem",
        "GridGenerator::channel_with_cylinder(tria, ...)        — Schäfer-Turek benchmark (2D-1, 2D-2, 2D-3); paper-quality reference for Stokes/NS up to Re~100",
        "GridGenerator::hyper_rectangle(tria, p1, p2)           — generic channel domain; pair with inflow on the left face, outflow on the right",
        "GridGenerator::hyper_L(tria, a, b)                     — backward-facing step; classic recirculating-flow test",
        "GridGenerator::subdivided_hyper_rectangle(tria, reps, p1, p2) — anisotropic refinement (taller in y than long in x), for boundary-layer resolution",
        "GridGenerator::hyper_cube_with_cylindrical_hole(tria, inner, outer) — flow around a cylinder in a box; lift/drag benchmarks",
        "GridGenerator::cheese(tria, ...)                       — domain with holes; multiscale / porous-flow demo",
    ],
    "solvers": [
        "SolverGMRES<>                — the canonical choice; Stokes is indefinite so SolverCG WILL fail.",
        "SolverMinRes<>               — symmetric indefinite; preferable to GMRES when a symmetric Schur preconditioner is used.",
        "SolverFGMRES<>               — flexible GMRES; needed when the inner preconditioner is itself an iterative solver (e.g. Schur complement built from an AMG-preconditioned CG on the velocity block).",
        "TrilinosWrappers::SolverDirect (UMFPACK / KLU) — direct for small problems; useful as a reference truth when iterative chains start producing wrong answers.",
    ],
    "preconditioners": [
        "BlockSchurPreconditioner (step-22 §) — the textbook approach; A_inv via inner CG on the velocity block, S_inv via the pressure mass matrix scaled by 1/viscosity.",
        "PreconditionAMG / BoomerAMG on the velocity block — TrilinosWrappers; needed to scale beyond ~10^5 DoFs.",
        "Vanka smoother for the FULL block system in geometric multigrid (step-56) — point Jacobi DOES NOT work because the saddle-point structure has zero diagonal in the pressure block.",
    ],
    "pitfalls": [
        "[Numerical] System is INDEFINITE — cannot use SolverCG, use "
        "SolverGMRES / SolverMinRes / a direct solver. Signal: "
        "SolverCG reports 'breakdown' on iteration 2-3 with a "
        "negative inner product, before the iterative residual has "
        "dropped at all.",
        "[Syntax] Block structure: use "
        "DoFRenumbering::component_wise + BlockSparseMatrix. Without "
        "the renumbering the velocity and pressure DoFs are "
        "interleaved and the BlockSparseMatrix indexing is wrong. "
        "Signal: assembly succeeds but the saddle-point block "
        "B^T does not have the expected zero-on-diagonal structure.",
        "[Physics] Pressure is determined up to a constant for pure "
        "Neumann (closed-cavity) problems — pin one DoF or add a "
        "zero-mean constraint. Signal: pressure field drifts to "
        "huge magnitude (~1e15) while velocity converges normally.",
        "[Numerical] For geometric multigrid: Vanka-type smoothers "
        "needed (step-56), NOT point Jacobi. Point Jacobi diverges "
        "on saddle-point systems because the pressure block has "
        "zero diagonal. Signal: GMG residual stagnates or diverges "
        "at the first V-cycle, even though direct solver on the "
        "same matrix converges.",
        "[Numerical] Inf-sup (Ladyzhenskaya-Babuška-Brezzi) stability "
        "is REQUIRED. Equal-order pairs (Q1/Q1, Q2/Q2) are NOT "
        "inf-sup stable — they look like they converge in 1D tests "
        "but produce checkerboard pressure modes in 2D. Use "
        "Taylor-Hood (Q_{p+1}/Q_p), MINI (Q1+bubble/Q1), "
        "RT/DGQ or BDM/DGP. Signal: pressure field has a regular "
        "high-frequency checkerboard pattern superimposed on the "
        "smooth solution.",
        "[Physics] FE_RaviartThomas-based H(div) velocity pairs "
        "produce EXACTLY divergence-free velocity at the discrete "
        "level — this is correct, not a bug. If the user is "
        "comparing against a Q2/Q1 result where div(u) ≈ 1e-3, the "
        "RT/DGQ result will report div(u) ≈ 1e-15. Signal: "
        "post-processing 'is my solver more accurate?' — yes, "
        "but only on mass conservation; other error metrics scale "
        "as usual.",
        "[Integration] channel_with_cylinder is the Schäfer-Turek "
        "benchmark — set the cylinder centre to (0.2, 0.2) and the "
        "channel size to (2.2 × 0.41) to match the published "
        "lift/drag values; off-by-one on these dimensions makes "
        "the reference values not match. Signal: reported "
        "C_D differs from the Schäfer-Turek table by 10+%.",
    ],
}
