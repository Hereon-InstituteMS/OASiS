"""deal.II Navier-Stokes generators and knowledge.

Based on step-57 (stationary NS), step-35 (Boussinesq), step-55 (Stokes MPI).
"""


def _navier_stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE — Stationary Navier-Stokes (Newton iteration)."""
    return '''\
/* Stationary Navier-Stokes — deal.II (based on step-57)
 * Newton iteration for nonlinear convective term.
 * Taylor-Hood Q2/Q1 elements. */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;

// Placeholder: implement Newton iteration for NS
// Reference: step-57 in deal.II tutorials
int main() {
    std::cout << "Navier-Stokes solver — see step-57 for full implementation" << std::endl;
    return 0;
}
'''


KNOWLEDGE = {
    "description": "Navier-Stokes (stationary and transient) — step-57, step-35, step-55",
    "tutorial_steps": ["step-57 (stationary NS, Newton)", "step-35 (Boussinesq buoyancy)",
                       "step-55 (Stokes, MPI parallel)"],
    "function_space": "FESystem<dim>(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1) — Taylor-Hood Q2/Q1",
    "solver": "Newton iteration for nonlinear convection term, UMFPACK or GMRES+ILU for linear sub-problems",
    "elements": [
        "FESystem<dim>(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1) — Taylor-Hood Q2/Q1, default for moderate-Re NS",
        "FESystem<dim>(FE_Q<dim>(p+1), dim, FE_Q<dim>(p), 1) — generalised Taylor-Hood; p ≥ 2 for high-fidelity DNS-like runs",
        "FE_Q + FE_DGP (continuous velocity + discontinuous pressure) — MINI-like; cheaper at p=1 if combined with bubble-enrichment on velocity",
        "FE_RaviartThomas<dim>(degree) for velocity + FE_DGQ<dim>(degree) for pressure — H(div) mixed; exactly divergence-free at the discrete level (momentum-conservative; the right pick when conservation matters)",
        "FE_DGQ<dim>(degree) for velocity AND pressure — fully DG formulation; for advection-dominated NS where upwinding helps",
    ],
    "mesh_generators": [
        "GridGenerator::channel_with_cylinder(tria, ...) — Schäfer-Turek benchmark; (0.2, 0.2) cylinder, (2.2 × 0.41) channel matches the published Re=20/100 lift/drag table",
        "GridGenerator::hyper_cube(tria, 0, 1) — driven-cavity benchmark; reference values (Ghia, Ghia, Shin 1982) at Re=100/400/1000/3200/5000/7500/10000",
        "GridGenerator::hyper_L(tria, a, b) — backward-facing step; reattachment-length benchmark",
        "GridGenerator::hyper_cube_with_cylindrical_hole(tria, inner, outer) — flow around cylinder; vortex-shedding at Re > 47",
        "GridGenerator::subdivided_hyper_rectangle(tria, reps, p1, p2) — channel flow with prescribed-aspect-ratio elements (boundary-layer resolution)",
        "GridGenerator::cheese(tria, ...) — porous-media-like NS demo",
    ],
    "solvers": [
        "Newton iteration — outer loop for the stationary nonlinear problem (step-57); converges quadratically near the solution",
        "Picard/Oseen iteration — first-order linearisation, larger basin of convergence than Newton; useful as a Newton warm-start",
        "BDF2 / Crank-Nicolson — time-stepping for transient NS; BDF2 is the canonical 2nd-order multi-step choice",
        "SparseDirectUMFPACK / MUMPS — robust linear sub-solver for moderate problem sizes",
        "SolverGMRES + ILU — iterative linear sub-solver; needs preconditioning beyond ~10^5 DoFs",
    ],
    "preconditioners": [
        "PreconditionILU / ILUT — for GMRES on the linearised NS tangent; cheap, works up to ~10^5 DoFs",
        "PreconditionAMG / BoomerAMG on the velocity block — combined with Schur-complement for parallel scaling (step-55)",
        "BlockSchurPreconditioner — block-triangular preconditioner; pressure Schur approximated by 1/mu * mass_p; the canonical step-22/step-57 choice",
    ],
    "pitfalls": [
        "[Numerical] NS is NONLINEAR — requires Newton iteration "
        "or Picard/Oseen linearisation. A naive linear solve gives "
        "the Stokes solution at zero Reynolds, regardless of the "
        "user's intended Re. Signal: solver converges in one "
        "iteration but the result has no advection-driven "
        "structure.",
        "[Numerical] Taylor-Hood Q2/Q1 satisfies inf-sup — Q1/Q1 "
        "DOES NOT and produces checkerboard pressure unless "
        "stabilised (SUPG, GLS, VMS). Signal: pressure field has "
        "a regular high-frequency checkerboard pattern.",
        "[Numerical] Reynolds number affects convergence — Newton "
        "diverges at high Re from a cold start. Continuation in Re: "
        "solve at Re=10, ramp through Re=50, 100, 200, ...; use "
        "each solution as the next starting guess. Signal: Newton "
        "fails to converge at Re=200 but converges from the Re=100 "
        "solution as initial guess.",
        "[Numerical] For time-dependent: BDF2 or Crank-Nicolson "
        "(2nd-order, A-stable). Backward Euler is robust but "
        "introduces O(dt) numerical viscosity that contaminates "
        "high-Re results. Signal: time-averaged Reynolds stress "
        "underpredicts a known reference by O(dt) — disappears "
        "with smaller dt.",
        "[Physics] Pressure is determined up to a constant for "
        "closed-cavity NS — pin at one point or use mean-free "
        "constraint. Signal: pressure field drifts to ~1e15 while "
        "velocity converges normally.",
        "[Integration] SUPG/GLS stabilisation parameter tau must "
        "be tuned to the local element size h and local advection "
        "speed |u|. The textbook tau = h / (2*|u|) is correct only "
        "in 1D; multi-dimensional NS needs tau = h / (2*|u|*sqrt(2)) "
        "or a more elaborate formula. Signal: spurious oscillations "
        "near boundary layers despite stabilisation being 'on'.",
    ],
}

GENERATORS = {
    "navier_stokes_2d": _navier_stokes_2d,
}
