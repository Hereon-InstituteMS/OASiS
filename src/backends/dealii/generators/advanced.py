"""deal.II advanced physics generators and knowledge.

Covers: mixed_laplacian, compressible_euler, time_dependent, matrix_free,
multigrid, multiphysics, obstacle, topology_opt, error_estimation, phase_field.
"""


def _placeholder_dealii(name: str, steps: str, desc: str) -> str:
    """Generate a placeholder C++ file referencing the deal.II tutorial."""
    return f'''\
/* {desc} — deal.II
 * Reference tutorials: {steps}
 * See: https://www.dealii.org/current/doxygen/deal.II/{steps.split("(")[0].strip().replace(" ", "_")}.html
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <iostream>
using namespace dealii;
int main() {{
    std::cout << "{name} solver — reference: {steps}" << std::endl;
    std::cout << "See deal.II tutorial for full implementation" << std::endl;
    return 0;
}}
'''


def _mixed_laplacian_2d(p: dict) -> str:
    return _placeholder_dealii("Mixed Laplacian", "step-20 (Raviart-Thomas)",
        "Mixed formulation with H(div) elements")

def _compressible_euler_2d(p: dict) -> str:
    return _placeholder_dealii("Compressible Euler", "step-33 (conservation laws), step-69 (Euler)",
        "Compressible gas dynamics with shock capturing")

def _time_dependent_heat_2d(p: dict) -> str:
    return _placeholder_dealii("Time-dependent heat", "step-26 (adaptive heat eq)",
        "Transient heat equation with adaptive mesh refinement")

def _time_dependent_wave_2d(p: dict) -> str:
    return _placeholder_dealii("Time-dependent wave", "step-23 (wave eq), step-48 (parallel)",
        "Second-order wave equation with Newmark time stepping")

def _time_dependent_ns_2d(p: dict) -> str:
    return _placeholder_dealii("Time-dependent NS", "step-35 (Boussinesq)",
        "Transient buoyancy-driven flow with Boussinesq approximation")

def _matrix_free_2d(p: dict) -> str:
    return _placeholder_dealii("Matrix-free", "step-37, step-59",
        "Matrix-free operator evaluation for high-performance FEM")

def _multigrid_2d(p: dict) -> str:
    return _placeholder_dealii("Multigrid", "step-16 (GMG), step-50 (parallel GMG)",
        "Geometric multigrid preconditioner for iterative solvers")

def _multiphysics_2d(p: dict) -> str:
    return _placeholder_dealii("Multiphysics", "step-21 (two-phase), step-43 (two-phase NS)",
        "Two-phase flow with Darcy or Navier-Stokes")

def _obstacle_2d(p: dict) -> str:
    return _placeholder_dealii("Obstacle/contact", "step-41",
        "Variational inequality / obstacle problem (contact)")

def _topology_opt_2d(p: dict) -> str:
    return _placeholder_dealii("Topology optimization", "step-79 (SIMP)",
        "SIMP topology optimization for compliance minimization")

def _error_estimation_2d(p: dict) -> str:
    return _placeholder_dealii("Error estimation", "step-14 (DWR), step-74 (refinement)",
        "Dual-weighted residual error estimation and adaptive refinement")

def _phase_field_2d(p: dict) -> str:
    return _placeholder_dealii("Phase field", "step-63",
        "Phase-field / advection-diffusion-reaction with SUPG stabilization")

def _dg_advection_2d(p: dict) -> str:
    return _placeholder_dealii("DG advection-reaction", "step-12 (DG), step-39 (DG+MG)",
        "Discontinuous Galerkin for advection with upwind flux")

def _cg_dg_coupled_2d(p: dict) -> str:
    return _placeholder_dealii("CG-DG coupled", "step-46",
        "Mixed continuous-discontinuous Galerkin methods")

def _optimal_control_2d(p: dict) -> str:
    return _placeholder_dealii("Optimal control / AD", "step-72 (automatic differentiation)",
        "Automatic differentiation for tangent assembly and optimization")


KNOWLEDGE = {
    "mixed_laplacian": {
        "description": "Mixed Laplacian with Raviart-Thomas H(div) elements (step-20)",
        "function_space": "FE_RaviartThomas + FE_DGQ for flux-pressure formulation",
        "pitfalls": [
            (
                "[API] H(div) elements (FE_RaviartThomas) have a "
                "DIFFERENT DOF structure from H1 — DoFs live on "
                "faces, not vertices. Signal: post-processing "
                "treating RT DoFs as nodal (e.g. "
                "DataOut::add_data_vector(..., DataOutBase::"
                "vertex_data)) raises an `ExcInternalError` or "
                "produces a per-vertex flux field that does not "
                "match the cell-face flux integral. Use "
                "DataOutBase::DG output or interpolate to a P1 "
                "post-processing space. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Schur complement solver for saddle-"
                "point system. Signal: a plain CG on the full "
                "(u, p) block matrix diverges because the system "
                "is indefinite; standard recipe is "
                "S = -B M^{-1} B^T and a CG on S with M^{-1} as "
                "preconditioner (step-20 / step-22). Without "
                "Schur complement reformulation, MINRES (or GMRES "
                "with a block preconditioner) is the only "
                "robust option. (Audit 2026-06-02.)"
            ),
        ],
    },
    "compressible_euler": {
        "description": "Compressible Euler equations — shock-capturing DG (step-33, step-69)",
        "methods": ["Lax-Friedrichs flux", "HLLC Riemann solver", "entropy viscosity"],
        "pitfalls": [
            (
                "[Numerical] Shock capturing requires artificial "
                "viscosity or limiting. Signal: high-order DG "
                "without limiting shows Gibbs over/undershoot of "
                "5-20% at shocks that does not decay with mesh "
                "refinement; entropy viscosity (step-69) or "
                "TVB-Minmod limiting eliminates them. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] CFL condition for explicit time "
                "stepping. Signal: dt > h/(|u|+c) in a "
                "SUNDIALS::ARKode integration of the FEEvaluation "
                "matrix-free residual gives NaN within ~10 steps; "
                "the SSP_RK3 scheme needs safety factor ~0.3. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Mach number scaling affects "
                "conditioning. Signal: low-Mach limit (M < 0.1) "
                "makes the compressible Euler stiffness matrix "
                "scale with 1/M^2; SolverGMRES / SolverCG "
                "iteration count grows linearly with 1/M "
                "unless a preconditioned-flux variant is used. "
                "PreconditionAMG can stabilise low-Mach "
                "regimes. (Audit 2026-06-02.)"
            ),
        ],
    },
    "time_dependent_heat": {
        "description": "Transient heat equation with AMR (step-26)",
        "time_integration": ["backward Euler", "Crank-Nicolson", "BDF2"],
        "pitfalls": [
            (
                "[API] Adaptive mesh refinement requires solution "
                "transfer between meshes. Signal: refining the "
                "mesh between time steps without "
                "SolutionTransfer<dim>::interpolate produces a "
                "zero-vector or random-noise solution on the new "
                "mesh — the previous solution lives on the OLD "
                "DoFHandler and is invalidated by refinement. "
                "step-26 shows the canonical SolutionTransfer "
                "(prepare_for_coarsening_and_refinement -> "
                "refine -> interpolate) sequence. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] CFL for explicit; unconditionally "
                "stable for implicit. Signal: explicit Euler on "
                "the heat equation diverges (NaN within ~10 "
                "steps) at dt > h^2/(2*alpha) — SUNDIALS::ARKode "
                "reports step rejection or SolverControl::"
                "failure; backward Euler / Crank-Nicolson via "
                "SUNDIALS::IDA are unconditionally stable but "
                "CN can oscillate at sharp fronts. Choose "
                "implicit for any production heat problem; use "
                "explicit only for didactic comparisons. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "time_dependent_wave": {
        "description": "Second-order wave equation (step-23, step-48)",
        "time_integration": ["Newmark-beta", "leapfrog"],
        "pitfalls": [
            (
                "[Numerical] Energy conservation — use symplectic "
                "integrators. Signal: integrating the wave "
                "equation with implicit Euler shows a "
                "monotonically DECAYING total energy "
                "(0.5 ||u_t||^2 + 0.5 ||grad u||^2) computed via "
                "VectorTools::integrate_difference — non-"
                "physical numerical dissipation. Leapfrog / "
                "Newmark-beta with beta=0.25, gamma=0.5 "
                "(step-23 demonstrates via DoFHandler + "
                "AffineConstraints) conserve energy to "
                "roundoff. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] CFL: dt < h/c for explicit schemes. "
                "Signal: leapfrog at dt > h/c oscillates with "
                "exponentially growing amplitude (factor of ~2 "
                "per step) — classic explicit-wave instability. "
                "Safety factor 0.5*h/c is conservative; CFL=1 is "
                "the strict bound. (Audit 2026-06-02.)"
            ),
        ],
    },
    "time_dependent_ns": {
        "description": "Transient Boussinesq flow — buoyancy-driven convection (step-35)",
        "pitfalls": [
            (
                "[Numerical] Rayleigh number controls flow "
                "regime. Signal: at Ra < 1707 (critical Rayleigh "
                "for Bénard convection) the simulation correctly "
                "shows a conductive (pure-diffusion) steady "
                "state; above that, convective cells appear. "
                "Computing at Ra > 1e6 without a turbulence model "
                "produces visibly chaotic transient behaviour "
                "that does not match a laminar DNS — switch to "
                "LES or RANS for very-high-Ra regimes. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Requires NS + energy equation "
                "coupling. Signal: solving NS in isolation (no "
                "buoyancy term in momentum) gives ZERO flow on a "
                "side-heated cavity — the canonical Boussinesq "
                "test. The momentum equation needs "
                "-rho * beta * (T - T_ref) * g_hat as a "
                "FEValuesExtractors::Vector source; without it "
                "the BlockVector temperature component stays "
                "decoupled. step-35 implements via "
                "BlockSparseMatrix and DoFTools::"
                "make_sparsity_pattern. (Audit 2026-06-02.)"
            ),
        ],
    },
    "matrix_free": {
        "description": "Matrix-free operator evaluation — high performance FEM (step-37, step-59)",
        "performance": "10-100x faster than sparse matrix for high-order elements",
        "pitfalls": [
            (
                "[API] Requires tensor-product elements (FE_Q, "
                "FE_DGQ). Signal: instantiating MatrixFree<dim> "
                "with FE_RaviartThomas / FE_BDM / non-tensor-"
                "product elements raises `MatrixFree: element "
                "type not supported` or silently disables "
                "vectorization. The performance gain (10-100x) "
                "depends entirely on tensor-product evaluation. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Performance] No matrix assembly — operator is "
                "applied on-the-fly. Signal: profiling shows zero "
                "time in SparseMatrix::add() (no global matrix); "
                "the bulk of wall-clock should be in MatrixFree::"
                "cell_loop and FEEvaluation::evaluate / "
                "integrate. If matrix-related calls appear, the "
                "code accidentally falls back to a sparse "
                "path. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Geometric multigrid essential for "
                "preconditioning. Signal: SolverCG without "
                "PreconditionAMG/MGSmootherRelaxation on a "
                "MatrixFree Laplace problem converges in "
                "~O(h^-1) iterations (gets worse with refinement, "
                "visible in SolverControl::log_history()); "
                "GMG keeps it at ~10-20 iterations independent "
                "of h. step-37 / step-50 show the canonical "
                "MatrixFree + GMG combination. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "multigrid": {
        "description": "Geometric multigrid preconditioner (step-16, step-50)",
        "types": ["h-multigrid (mesh hierarchy)", "p-multigrid (polynomial degree)"],
        "pitfalls": [
            (
                "[Numerical] Smoother choice: PreconditionChebyshev "
                "for SPD, SolverGMRES for indefinite. Signal: "
                "applying PreconditionChebyshev to an indefinite "
                "Stokes-type system produces diverging multigrid "
                "V-cycles (norm grows by factor ~1.5 per cycle, "
                "visible in MGSmootherRelaxation residuals); "
                "switching to a few smoothing steps of "
                "SolverGMRES restores convergence. Conversely, "
                "GMRES smoothing on SPD is slower than "
                "PreconditionChebyshev. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Coarse grid solver: SolverDirect "
                "(UMFPACK / Trilinos Amesos) or iterative. "
                "Signal: leaving the coarse-grid smoother as a "
                "default PreconditionJacobi gives V-cycle "
                "convergence rate proportional to coarse-grid "
                "DOFs; TrilinosWrappers::SolverDirect or "
                "PETScWrappers MUMPS on the coarsest level "
                "restores h-independent multigrid convergence. "
                "For very large meshes use an iterative coarse "
                "solver to avoid the direct solver memory "
                "blowup. (Audit 2026-06-02.)"
            ),
        ],
    },
    "multiphysics_dealii": {
        "description": "Two-phase flow and multi-physics coupling (step-21, step-43)",
        "pitfalls": [
            (
                "[Numerical] Darcy flow (step-21) vs full NS "
                "(step-43) — choose by Reynolds number. Signal: "
                "applying step-21 (Darcy momentum law u = -K * "
                "grad p) to a high-Re channel flow gives a "
                "Poiseuille-like profile that under-predicts "
                "advective effects by orders of magnitude. "
                "Conservation: Darcy ignores inertia (rho * "
                "Du/Dt); step-43 retains it. Use Darcy only for "
                "porous-media subsurface flow, NS for free-flow. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Interface tracking: level-set or "
                "phase-field. Signal: a sharp-interface VoF "
                "reconstruction on a coarse mesh smears the "
                "interface across 3-5 cells with mass-loss > 1% "
                "per period of an interface oscillation; level-"
                "set with reinitialisation or a phase-field with "
                "Cahn-Hilliard regularisation conserves mass to "
                "machine precision. step-43 is level-set, step-"
                "60 ish is phase-field. (Audit 2026-06-02.)"
            ),
        ],
    },
    "obstacle_problem": {
        "description": "Variational inequality / contact / obstacle problem (step-41)",
        "method": "Active set strategy — project onto feasible set each Newton step",
        "pitfalls": [
            (
                "[Numerical] Non-smooth problem — requires special "
                "solver (active set, penalty). Signal: a "
                "vanilla Newton SolverControl loop on a "
                "variational inequality (elastic body pressing "
                "into a rigid obstacle) either diverges or "
                "oscillates between two active-set "
                "AffineConstraints states without converging; "
                "step-41's active-set strategy iterates "
                "(IndexSet constraint-detection -> linear solve) "
                "until two consecutive active sets are identical, "
                "typically 3-10 outer iterations. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "topology_opt_dealii": {
        "description": "SIMP topology optimization (step-79)",
        "method": "SIMP with density filtering and MMA optimizer",
        "pitfalls": [
            (
                "[Numerical] Penalization factor p=3 is the SIMP "
                "default (intermediate-density penalisation). "
                "Signal: p < 2 leaves the optimisation with too "
                "much grey-scale intermediate density (volume "
                "fraction outside [0.05, 0.95] for > 30% of "
                "cells); p > 4 can over-penalise and freeze the "
                "topology in the wrong configuration. Standard "
                "SIMP literature ranges 3-5. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Filter radius prevents checkerboard. "
                "Signal: without density filtering, the optimal "
                "topology shows alternating high/low density on "
                "adjacent elements (checkerboard pattern visible "
                "via DataOut::write_vtu cell-data output); a "
                "density-Vector filter computed via VectorTools::"
                "interpolate with radius ~1.5*h removes it. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Mesh-dependent without proper "
                "regularization. Signal: refining the dealii "
                "Triangulation and re-running gives a DIFFERENT "
                "optimal topology_opt (more thin struts) — the "
                "SIMP problem is ill-posed without a length "
                "scale. A Helmholtz_filter density_filter with "
                "a FIXED physical radius (not h-scaled) "
                "restores mesh-independent optimal design. "
                "(Audit 2026-06-02.)"
            ),
        ],
    },
    "error_estimation": {
        "description": "Dual-weighted residual (DWR) error estimation (step-14, step-74)",
        "method": "Solve dual/adjoint problem, weight residual for goal-oriented refinement",
        "pitfalls": [
            (
                "[Numerical] Dual problem requires adjoint "
                "assembly. Signal: a goal-oriented refinement "
                "loop driven only by the PRIMAL residual (no "
                "dual) refines uniformly toward singularities "
                "regardless of the goal functional; the "
                "effectivity index (estimated / true error) is "
                "typically O(1) to 10x off without the dual. "
                "step-14 / step-74 show the canonical DWR "
                "(residual * weighted-dual) loop. (Audit "
                "2026-06-02.)"
            ),
            (
                "[Numerical] Higher-order dual solution needed "
                "for effectivity index. Signal: solving the dual "
                "in the SAME finite-element space as the primal "
                "makes the dual-weighted residual collapse to "
                "zero (Galerkin orthogonality) — the effectivity "
                "index is trivially 1 but the estimator gives "
                "zero refinement information. Use one order "
                "higher (or a patch-wise enrichment) for the "
                "dual. (Audit 2026-06-02.)"
            ),
        ],
    },
    "phase_field": {
        "description": "Phase-field / advection-diffusion-reaction with SUPG (step-63)",
        "pitfalls": [
            (
                "[Numerical] SUPG stabilization for advection-"
                "dominated problems. Signal: a Galerkin "
                "discretisation at Peclet number > 1 produces "
                "wiggles in the boundary layer (visible in "
                "DataOut::write_vtu output as high-frequency "
                "oscillations near walls) that do not damp with "
                "refinement. Add a SUPG term "
                "tau * (b . grad phi) * (b . grad u - source) "
                "inside the FEValues quadrature loop; "
                "oscillations disappear. step-63 demonstrates "
                "this stabilisation. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Peclet number determines "
                "stabilization strength. Signal: a fixed SUPG "
                "tau = h/(2*|b|) over-stabilises in diffusion-"
                "dominated zones (smearing of sharp gradients "
                "by ~20% even in clearly resolved regions, "
                "diagnosable via KellyErrorEstimator). Use "
                "tau = h/(2*|b|) * f(Pe_h) where f(Pe) is the "
                "doubly-asymptotic switch (coth(Pe) - 1/Pe). "
                "(Audit 2026-06-02.)"
            ),
        ],
    },
    "dg_advection_reaction": {
        "description": "DG for advection with upwind flux (step-12, step-39)",
        "pitfalls": [
            (
                "[Numerical] Upwind flux for stability. Signal: a "
                "DG advection discretisation with a CENTRAL "
                "numerical flux (0.5 * (u^+ + u^-)) on a "
                "pure-advection problem is unstable — the "
                "solution amplitude grows like ~exp(t) regardless "
                "of mesh size. Use upwind: u_hat = u^- (downstream "
                "cell takes upstream value); stability is "
                "recovered. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] DG + multigrid in step-39. Signal: a "
                "plain SolverCG / SolverGMRES on the DG-"
                "advection system shows iteration count "
                "proportional to mesh Reynolds number (Pe_h) — "
                "not h-independent (visible in SolverControl "
                "history). Geometric multigrid with "
                "PreconditionBlock (block-Jacobi over the DG "
                "block + DoFRenumbering::downstream) restores "
                "h-independent convergence; see step-39 / "
                "step-50. (Audit 2026-06-02.)"
            ),
        ],
    },
    "cg_dg_coupled": {
        "description": "Mixed CG-DG methods (step-46)",
        "pitfalls": [
            (
                "[API] Different FE spaces in different "
                "subdomains. Signal: assembling on a "
                "DoFHandler<dim> that wraps an FECollection "
                "(FE_Q + FE_DGQ) without setting active_fe_index "
                "per cell raises `ExcMessage(\"Two cells have "
                "different active_fe_index\")` at the first "
                "subdomain boundary. Set "
                "cell->set_active_fe_index(0 or 1) by subdomain "
                "BEFORE distribute_dofs. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Interface conditions between CG and "
                "DG regions. Signal: a CG-DG coupling without "
                "explicit interface flux (mortar penalty or "
                "Nitsche, assembled via FEInterfaceValues) "
                "shows a jump in the solution across the CG/DG "
                "boundary that does NOT decay with refinement — "
                "DataOut::write_vtu reveals the jump visually; "
                "the FECollection spaces are incompatible "
                "there. step-46's mortar enforces continuity "
                "weakly via AffineConstraints. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "optimal_control": {
        "description": "Automatic differentiation for tangent/residual (step-72)",
        "method": "Sacado AD for automatic tangent assembly",
        "pitfalls": [
            (
                "[Numerical] AD adds overhead but eliminates "
                "hand-coded tangent errors. Signal: a Newton "
                "SolverControl loop with a hand-coded analytic "
                "tangent that is OFF by a sign or factor-of-2 "
                "shows linear (not quadratic) convergence — the "
                "residual norm halves per Newton iteration "
                "instead of squaring (visible in SolverControl::"
                "log_history). Switching to a "
                "Differentiation::AD Sacado::Fad-derived tangent "
                "restores quadratic convergence at the cost of "
                "~2-4x per-element assembly time. (Audit "
                "2026-06-02.)"
            ),
            (
                "[API] Requires Trilinos with Sacado support. "
                "Signal: #include <deal.II/differentiation/ad/"
                "sacado_number_types.h> followed by a compile "
                "error `Sacado/Sacado.hpp: No such file or "
                "directory` means the local Trilinos was built "
                "without -DTrilinos_ENABLE_Sacado=ON. Re-cmake "
                "with Sacado enabled (or use the deal.II "
                "PETSc/Trilinos conda-forge package which "
                "includes it). (Audit 2026-06-02.)"
            ),
        ],
    },
}

GENERATORS = {
    "mixed_laplacian_2d": _mixed_laplacian_2d,
    "compressible_euler_2d": _compressible_euler_2d,
    "time_dependent_heat_2d": _time_dependent_heat_2d,
    "time_dependent_wave_2d": _time_dependent_wave_2d,
    "time_dependent_ns_2d": _time_dependent_ns_2d,
    "matrix_free_2d": _matrix_free_2d,
    "multigrid_2d": _multigrid_2d,
    "multiphysics_dealii_2d": _multiphysics_2d,
    "obstacle_problem_2d": _obstacle_2d,
    "topology_opt_dealii_2d": _topology_opt_2d,
    "error_estimation_2d": _error_estimation_2d,
    "phase_field_2d": _phase_field_2d,
    "dg_advection_reaction_2d": _dg_advection_2d,
    "cg_dg_coupled_2d": _cg_dg_coupled_2d,
    "optimal_control_2d": _optimal_control_2d,
}
