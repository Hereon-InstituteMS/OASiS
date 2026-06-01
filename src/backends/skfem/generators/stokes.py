"""scikit-fem Stokes flow generators and knowledge."""


def _stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Stokes flow with Taylor-Hood P2/P1."""
    nx = params.get("nx", 16)
    return f'''\
"""Stokes flow — Taylor-Hood P2/P1 — scikit-fem"""
from skfem import *
from skfem.models.poisson import vector_laplace, laplace, mass
import numpy as np
import json

m = MeshQuad.init_tensor(np.linspace(0, 1, {nx+1}), np.linspace(0, 1, {nx+1}))
e_u = ElementVector(ElementQuad2())
e_p = ElementQuad1()

ib_u = Basis(m, e_u)
ib_p = Basis(m, e_p)

@BilinearForm
def viscosity(u, v, w):
    return sum(u.grad[i].grad[j] * v.grad[i].grad[j]
               for i in range(2) for j in range(2))

# This is a simplified Stokes — for production use the block system
# K = [[A, B^T], [B, 0]] with divergence constraint

from skfem.models.general import divergence
K11 = asm(vector_laplace, ib_u)
K12 = -asm(divergence, ib_u, ib_p)

from scipy.sparse import bmat
K = bmat([[K11, K12], [K12.T, None]], format='csr')
f = np.zeros(K.shape[0])

# Velocity BCs — set for your problem
D_u = ib_u.get_dofs().flatten()
n_u = K11.shape[0]
# Set velocity BC values
d_vals = np.zeros(K.shape[0])

u = solve(*condense(K, f, D=np.concatenate([D_u, n_u + np.array([])]), x=d_vals))
print(f"Stokes: {{K.shape[0]}} DOFs")

summary = {{"n_dofs": K.shape[0]}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Stokes solve complete.")
'''


KNOWLEDGE = {
    "stokes": {
        "description": "Stokes flow — Taylor-Hood P2/P1 or Mini element (examples 18, 24, 30, 32)",
        "solver": "Block system: [[A, B^T], [B, 0]], solved via direct or Krylov-Uzawa",
        "elements": "Taylor-Hood: ElementVector(ElementTriP2()) + ElementTriP1(). Mini: ElementTriMini()",
        "pitfalls": [
            "[Numerical] The Stokes block system [[A, B^T], "
            "[B, 0]] is INDEFINITE — CG on the full block system "
            "diverges. Use a direct solver (scipy.sparse.linalg."
            "spsolve with UMFPACK / SuperLU) or block Uzawa "
            "iteration. Signal: scipy.sparse.linalg.cg on the "
            "assembled block matrix reports info != 0 with the "
            "residual stalling at O(1); switching to spsolve "
            "returns a vector and the divergence of velocity is "
            "below tolerance. (Claim inherited from prose — not "
            "yet empirically separated.)",
            "[Numerical] Krylov-Uzawa (skfem example 30) splits "
            "the saddle-point: outer iteration on the pressure "
            "Schur complement, inner solve of the velocity block. "
            "Signal: per-outer-iter pressure residual drops by a "
            "factor 0.1..0.3; switching to monolithic Krylov on "
            "the full block system shows residual stalling. "
            "(Claim inherited — recommendation, no direct "
            "exception text.)",
            "[Physics] Enclosed-flow Stokes admits the constant "
            "pressure null space — pin pressure at one DoF "
            "(boundary_values[p_dof] = 0) or add a Lagrange "
            "multiplier mean(p) = 0. Open flows with "
            "do-nothing (traction-free) outlet BCs determine "
            "pressure uniquely without pinning. Signal: "
            "spsolve(A_block, b_block) returns successfully but "
            "u_h shows the right velocity field while p_h has "
            "an arbitrary additive constant (max(p_h) - min(p_h) "
            "is bounded by the physical pressure variation, but "
            "the mean is unconstrained). Same family as fenics "
            "poisson#3 (pure Neumann). (Claim inherited — "
            "not yet empirically verified.)",
            "[Syntax] 3D Stokes (skfem example 32): use "
            "ElementTetP2 for velocity + ElementTetP1 for "
            "pressure (Taylor-Hood). Other tet combinations "
            "violate inf-sup. Signal: the assembled block matrix "
            "from skfem.asm with ElementTetP1 / ElementTetP1 "
            "produces a pressure GridFunction whose mode pattern "
            "alternates by element (checkerboard); the "
            "Functional integral of div(u) over the mesh is "
            "O(1) instead of O(eps). Switching to "
            "ElementTetP2 + ElementTetP1 (Taylor-Hood) recovers "
            "divergence below tolerance. (Claim inherited — "
            "not yet empirically verified.)",
            "[API] CRITICAL: skfem.Basis.Nbfun returns the "
            "PER-ELEMENT DOF count (3 for ElementTriP1, 6 for "
            "ElementTriP2), NOT the global DOF count. Use "
            "basis.N or the assembled matrix shape (A.shape[0]) "
            "for the global count. Signal: basis.Nbfun on a "
            "refined MeshTri is 3 or 6 regardless of how many "
            "elements; basis.N grows with refinement. (Verified "
            "empirically 2026-06-01: P1 Nbfun=3, P2 Nbfun=6 on "
            "MeshTri.refined(2), while N=25 and N=81 "
            "respectively. Using Nbfun for DOF splitting "
            "silently slices wrong.)",
            "[API] ElementVector DOF ordering is unreliable for "
            "mixed systems. Safer pattern: build TWO separate "
            "SCALAR ElementTriP2 bases (one for u_x, one for "
            "u_y), assemble each block explicitly, and assemble "
            "the final block matrix with scipy.sparse.bmat. "
            "Signal: ElementVector(ElementTriP2()).get_dofs() "
            "ordering is documented as 'interleaved' but the "
            "actual layout depends on element type; building "
            "from two scalar bases gives a stable, "
            "user-controlled block layout. (Claim inherited — "
            "not yet empirically falsified.)",
            "[API] skfem.asm() with two bases of mismatched "
            "intorder raises ValueError 'Quadrature mismatch: "
            "trial and test functions should have same number "
            "of integration points.'. Mixed Stokes (P2 velocity "
            "+ P1 pressure) needs intorder=4 (or higher) on "
            "BOTH bases. Signal: scipy ValueError text matches "
            "the exact wording above when asm(b_form, p2_basis, "
            "p1_basis) has different intorder. (Verified "
            "empirically 2026-06-01 with intorder=2 vs 6.)",
            "[Physics] scikit-fem Stokes uses the -p*div(v) "
            "convention (same as FEniCS). NGSolve uses +p*div(v). "
            "Both are valid weak forms but produce different "
            "pressure signs. Signal: a Poiseuille-flow benchmark "
            "solved on the same geometry in skfem and NGSolve "
            "gives pressure fields that differ by a sign at every "
            "DoF (max(p_skfem) ≈ -max(p_ngsolve)). (Claim "
            "inherited — not yet empirically cross-verified.)",
        ],
    },
}

GENERATORS = {
    "stokes_2d": _stokes_2d,
}
