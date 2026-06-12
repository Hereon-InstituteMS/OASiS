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
    "elements": {
        "FESystem":
            "Block wrapper for ALL NS pairs. Standard Taylor-Hood "
            "shape: FESystem<dim>(FE_Q<dim>(2), dim, FE_Q<dim>(1), "
            "1). Generalised Q_{p+1}/Q_p for higher-order runs.",
        "FE_Q":
            "Velocity (degree p+1) and pressure (degree p) in "
            "Taylor-Hood. Equal-order Q1/Q1 needs SUPG / GLS / "
            "VMS stabilisation to be inf-sup stable.",
        "FE_Q_Bubbles":
            "Velocity component of MINI-like (Q1+bubble / Q1) — "
            "cheaper than Taylor-Hood Q2/Q1 in DoF count, "
            "inf-sup stable.",
        "FE_DGP":
            "Pressure component of MINI-like (Q1+bubble / DGP0).",
        "FE_RaviartThomas":
            "Velocity of RT/DGQ H(div) pair. Exactly "
            "divergence-free velocity — the right pick when "
            "momentum conservation matters (geophysical flow, "
            "groundwater coupled with advection).",
        "FE_DGQ":
            "Pressure of RT/DGQ pair, or velocity AND pressure "
            "in fully-DG NS for advection-dominated problems "
            "where upwinding helps stability.",
    },
    "mesh_generators": {
        "channel_with_cylinder": "Schäfer-Turek benchmark — cylinder (0.2, 0.2) in (2.2 × 0.41) channel matches published Re=20/100 lift/drag.",
        "hyper_cube": "Driven-cavity benchmark; Ghia/Ghia/Shin (1982) reference values at Re=100/400/1000/3200/5000/7500/10000.",
        "hyper_L": "Backward-facing step; reattachment-length benchmark.",
        "hyper_cube_with_cylindrical_hole": "Flow around cylinder; vortex-shedding at Re > 47.",
        "subdivided_hyper_rectangle": "Channel flow with prescribed-aspect elements (boundary-layer resolution).",
        "cheese": "Porous-media-like NS demo.",
    },
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
        "user's intended Re. Signal: SolverGMRES with no outer "
        "Newton/Picard loop converges in 1 iteration to a "
        "symmetric, advection-free velocity profile — DataOut "
        "shows no recirculation behind a cylinder at Re=100 (which "
        "should have a visible Karman vortex street); "
        "VectorTools::integrate_difference against a Stokes "
        "reference is ~0, against a NS reference is O(1).",
        "[Numerical] Taylor-Hood Q2/Q1 satisfies inf-sup — Q1/Q1 "
        "DOES NOT and produces checkerboard pressure unless "
        "stabilised (SUPG, GLS, VMS). Signal: DataOut output for "
        "the BlockVector pressure block shows a regular "
        "high-frequency checkerboard pattern; "
        "VectorTools::point_value at adjacent cell centroids "
        "alternates sign with O(1) magnitude.",
        "[Numerical] Reynolds number affects convergence — Newton "
        "diverges at high Re from a cold start. Continuation in Re: "
        "solve at Re=10, ramp through Re=50, 100, 200, ...; use "
        "each solution as the next starting guess. Signal: "
        "SolverControl reports residual.l2_norm() > 1e3 on Newton "
        "iteration 1 at Re=200 from a zero initial guess, ending "
        "in ExcMessage('iterative method failed to converge'); "
        "rerunning with the Re=100 solution stored in BlockVector "
        "as the initial guess converges in 4-6 Newton steps.",
        "[Numerical] For time-dependent: BDF2 or Crank-Nicolson "
        "(2nd-order, A-stable). Backward Euler is robust but "
        "introduces O(dt) numerical viscosity that contaminates "
        "high-Re results. Signal: time-averaged Reynolds-stress "
        "magnitude from VectorTools::integrate_difference differs "
        "by 20-40% from a Schäfer-Turek reference at Re=100 — "
        "halving dt drops the error proportionally (O(dt) "
        "scaling); switching to BDF2 / Crank-Nicolson at the "
        "same dt reduces the error below 5%.",
        "[Physics] Pressure is determined up to a constant for "
        "closed-cavity NS — pin at one point or use mean-free "
        "constraint via AffineConstraints. Signal: "
        "`solution.block(1).linfty_norm()` (pressure) drifts to "
        ">1e10 magnitude across Newton iterations while "
        "`solution.block(0).l2_norm()` (velocity) converges "
        "normally; SolverGMRES iteration count for the linearised "
        "tangent grows each outer step as the pressure null space "
        "pollutes the Krylov basis.",
        "[Integration] SUPG/GLS stabilisation parameter tau must "
        "be tuned to the local element size h and local advection "
        "speed |u|. The textbook tau = h / (2*|u|) is correct only "
        "in 1D; multi-dimensional NS needs tau = h / (2*|u|*sqrt(2)) "
        "or a more elaborate formula. Signal: DataOut shows visible "
        "spatial oscillations of magnitude 0.05-0.2 (relative to "
        "max velocity) within 2-3 cells of a no-slip wall despite "
        "the SUPG term being active; refining h eliminates them "
        "only locally; comparing tau values between 1D and 2D "
        "shows a sqrt(2) discrepancy at the wall.",
    ],
}

GENERATORS = {
    "navier_stokes_2d": _navier_stokes_2d,
}
