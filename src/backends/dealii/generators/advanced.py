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
        "pitfalls": ["H(div) elements — different DOF structure from H1",
                     "Schur complement solver for saddle-point system"],
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
                "stepping. Signal: dt > h/(|u|+c) gives NaN "
                "within ~10 steps; SSP-RK3 needs safety factor "
                "~0.3. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Mach number scaling affects "
                "conditioning. Signal: low-Mach limit (M < 0.1) "
                "makes the compressible Euler stiffness matrix "
                "scale with 1/M^2; iterative solver iteration "
                "count grows linearly with 1/M unless a "
                "preconditioned-flux variant is used. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "time_dependent_heat": {
        "description": "Transient heat equation with AMR (step-26)",
        "time_integration": ["backward Euler", "Crank-Nicolson", "BDF2"],
        "pitfalls": ["Adaptive mesh refinement requires solution transfer between meshes",
                     "CFL for explicit; unconditionally stable for implicit"],
    },
    "time_dependent_wave": {
        "description": "Second-order wave equation (step-23, step-48)",
        "time_integration": ["Newmark-beta", "leapfrog"],
        "pitfalls": ["Energy conservation — use symplectic integrators",
                     "CFL: dt < h/c for explicit schemes"],
    },
    "time_dependent_ns": {
        "description": "Transient Boussinesq flow — buoyancy-driven convection (step-35)",
        "pitfalls": ["Rayleigh number controls flow regime",
                     "Requires NS + energy equation coupling"],
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
                "preconditioning. Signal: CG without multigrid on "
                "a matrix-free Laplace problem converges in "
                "~O(h^-1) iterations (gets worse with refinement); "
                "GMG keeps it at ~10-20 iterations independent "
                "of h. step-37 and step-50 show the canonical "
                "matrix-free + GMG combination. (Audit "
                "2026-06-02.)"
            ),
        ],
    },
    "multigrid": {
        "description": "Geometric multigrid preconditioner (step-16, step-50)",
        "types": ["h-multigrid (mesh hierarchy)", "p-multigrid (polynomial degree)"],
        "pitfalls": [
            (
                "[Numerical] Smoother choice: Chebyshev for SPD, "
                "GMRES for indefinite. Signal: applying Chebyshev "
                "to an indefinite Stokes-type system produces "
                "diverging multigrid V-cycles (norm grows by "
                "factor ~1.5 per cycle); switching to a few "
                "smoothing steps of GMRES restores convergence. "
                "Conversely, GMRES smoothing on SPD is slower "
                "than Chebyshev. (Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Coarse grid solver: direct (Amesos) "
                "or iterative. Signal: leaving the coarse-grid "
                "solver as a default Jacobi gives V-cycle "
                "convergence rate proportional to coarse-grid "
                "DOFs; Amesos / SparseDirectUMFPACK on the "
                "coarsest level restores h-independent multigrid "
                "convergence. For very large meshes use an "
                "iterative coarse solver to avoid the direct "
                "solver memory blowup. (Audit 2026-06-02.)"
            ),
        ],
    },
    "multiphysics_dealii": {
        "description": "Two-phase flow and multi-physics coupling (step-21, step-43)",
        "pitfalls": ["Darcy flow (step-21) vs full NS (step-43)",
                     "Interface tracking: level-set or phase-field"],
    },
    "obstacle_problem": {
        "description": "Variational inequality / contact / obstacle problem (step-41)",
        "method": "Active set strategy — project onto feasible set each Newton step",
        "pitfalls": [
            (
                "[Numerical] Non-smooth problem — requires special "
                "solver (active set, penalty). Signal: a "
                "vanilla Newton solve on a variational inequality "
                "(elastic body pressing into a rigid obstacle) "
                "either diverges or oscillates between two "
                "active-set states without converging; step-41's "
                "active-set strategy iterates "
                "(constraint-detection -> linear solve) until two "
                "consecutive active sets are identical, typically "
                "3-10 outer iterations. (Audit 2026-06-02.)"
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
                "adjacent elements (checkerboard pattern); a "
                "density filter with radius ~1.5*h removes it. "
                "(Audit 2026-06-02.)"
            ),
            (
                "[Numerical] Mesh-dependent without proper "
                "regularization. Signal: refining the mesh and "
                "re-running gives a DIFFERENT optimal topology "
                "(more thin struts) — physical solution is "
                "ill-posed without a length scale. Density "
                "filtering with a FIXED physical radius (not "
                "h-scaled) restores mesh-independent optimal "
                "design. (Audit 2026-06-02.)"
            ),
        ],
    },
    "error_estimation": {
        "description": "Dual-weighted residual (DWR) error estimation (step-14, step-74)",
        "method": "Solve dual/adjoint problem, weight residual for goal-oriented refinement",
        "pitfalls": ["Dual problem requires adjoint assembly",
                     "Higher-order dual solution needed for effectivity index"],
    },
    "phase_field": {
        "description": "Phase-field / advection-diffusion-reaction with SUPG (step-63)",
        "pitfalls": ["SUPG stabilization for advection-dominated problems",
                     "Peclet number determines stabilization strength"],
    },
    "dg_advection_reaction": {
        "description": "DG for advection with upwind flux (step-12, step-39)",
        "pitfalls": ["Upwind flux for stability", "DG + multigrid in step-39"],
    },
    "cg_dg_coupled": {
        "description": "Mixed CG-DG methods (step-46)",
        "pitfalls": ["Different FE spaces in different subdomains",
                     "Interface conditions between CG and DG regions"],
    },
    "optimal_control": {
        "description": "Automatic differentiation for tangent/residual (step-72)",
        "method": "Sacado AD for automatic tangent assembly",
        "pitfalls": ["AD adds overhead but eliminates hand-coded tangent errors",
                     "Requires Trilinos with Sacado support"],
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
