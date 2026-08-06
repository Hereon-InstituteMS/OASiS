"""scikit-fem Stokes flow generators and knowledge."""


def _stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Stokes flow with Taylor-Hood P2/P1.

    Fixed 2026-06-01 (was unrunnable):
      * Mesh: MeshTri.refined(N) (vs prior MeshQuad —
        ElementQuad2 has assembly issues with the
        divergence form in skfem 12).
      * intorder=4 on BOTH bases (trial/test mismatch
        on the divergence bilinear form was the immediate
        ValueError on import).
      * B-block sign: skfem.models.general.divergence
        returns +∫q·div(u); the catalog negates to form
        the standard saddle-point [[K, -B^T], [-B, 0]].
      * Pressure pin: a single pressure DOF is fixed at
        0 to remove the constant-pressure null space
        (without this, spsolve raises a singular-matrix
        error even with the correct BC).
      * Driven-cavity BC: u = (1, 0) on top edge, no-slip
        elsewhere — replaces the prior all-zero BC which
        gave the trivial u=0 solution.
    """
    nx_refine = int(params.get("refine", 4))
    return f'''\
"""Stokes flow — Taylor-Hood P2/P1 — scikit-fem"""
from skfem import (
    MeshTri, Basis, BilinearForm,
    ElementVector, ElementTriP1, ElementTriP2,
    asm, condense, solve,
)
from skfem.helpers import grad, ddot
from skfem.models.general import divergence
from scipy.sparse import bmat
import numpy as np
import json

# Mesh + Taylor-Hood spaces.
# Trial / test bases must share quadrature for the mixed
# divergence(basis_u, basis_p) bilinear form. Default
# intorder differs between P2 and P1 → ValueError
# 'Quadrature mismatch'. Force intorder=4 on both.
m = MeshTri().refined({nx_refine})
basis_u = Basis(m, ElementVector(ElementTriP2()), intorder=4)
basis_p = Basis(m, ElementTriP1(), intorder=4)


@BilinearForm
def stiffness(u, v, w):
    return ddot(grad(u), grad(v))


K_block = asm(stiffness, basis_u)
# B[q, u] = ∫q·div(u) dx (skfem convention is +div, so
# we negate to form the standard saddle-point block
# [[K, -B^T], [-B, 0]]).
B_block = -asm(divergence, basis_u, basis_p)

A = bmat([[K_block, B_block.T], [B_block, None]], format='csr')
F = np.zeros(A.shape[0])

# Driven-cavity BC: u = (1, 0) on top edge, no-slip elsewhere.
# ElementVector interleaves x/y dofs at each node:
# dof[2i]=x-component, dof[2i+1]=y-component.
doflocs_u = basis_u.doflocs
by = doflocs_u[1]
top_x = np.isclose(by[0::2], 1.0)
u_bc = np.zeros(basis_u.N)
u_bc[0::2] = np.where(top_x, 1.0, 0.0)
u_bc[1::2] = 0.0

# Pin a single pressure DOF (closest to the origin)
# to fix the constant-pressure null space.
pdofs = basis_p.doflocs.T
pin_p_local = int(np.argmin(
    np.linalg.norm(pdofs[:, :2], axis=1)))
pin_p_global = basis_u.N + pin_p_local

D_u = basis_u.get_dofs().flatten()
D = np.concatenate([
    D_u,
    np.array([pin_p_global], dtype=np.int64),
])
x_full = np.zeros(A.shape[0])
x_full[:basis_u.N] = u_bc

sol = solve(*condense(A, F, D=D, x=x_full))
u_h = sol[:basis_u.N]
p_h = sol[basis_u.N:]
print(f"Stokes Taylor-Hood: total DOFs = {{A.shape[0]}}")
print(f"  ||u_x||_inf = {{np.abs(u_h[0::2]).max():.4f}}")
print(f"  ||u_y||_inf = {{np.abs(u_h[1::2]).max():.4f}}")
print(f"  ||p||_inf   = {{np.abs(p_h).max():.4f}}")

summary = {{
    "n_dofs": int(A.shape[0]),
    "max_u_x": float(np.abs(u_h[0::2]).max()),
    "max_u_y": float(np.abs(u_h[1::2]).max()),
    "max_p":   float(np.abs(p_h).max()),
}}
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
            "violate inf-sup. THERE IS NO GridFunction IN "
            "scikit-fem — hasattr(skfem, 'GridFunction') is "
            "False; that is NGSolve vocabulary. A skfem solution "
            "is a plain 1-D numpy.ndarray over the stacked "
            "[u; p] block, and you slice the pressure out of it "
            "at offset basis_u.N. "
            "Signal: do not look for a checkerboard by eye — "
            "count the spurious pressure modes instead, which "
            "needs no solve at all. Assemble the block, condense "
            "the velocity boundary and one pinned pressure DOF, "
            "and take the nullity of the condensed matrix: for "
            "the Taylor-Hood pair that count stays CONSTANT as "
            "the mesh is refined, while for an equal-order "
            "P1/P1 pair it GROWS with the mesh, which is the "
            "inf-sup failure made countable. A second, cheaper "
            "check is the pressure range from a lightly "
            "regularised solve: bounded and converging for "
            "Taylor-Hood, growing by an order of magnitude per "
            "refinement for equal order. (Verified empirically "
            "2026-08-06 on skfem 12.0.1 — the GridFunction "
            "wording was foreign vocabulary and is removed; the "
            "previous 'integral of div(u) is O(1) instead of "
            "O(eps)' wording did NOT reproduce cleanly under a "
            "regularised solve and has been replaced by the "
            "mode-count signature.)",
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
            "[API] ElementVector DOF ordering IS reliable and IS "
            "interleaved — the previous 'unreliable, depends on "
            "element type' wording is wrong and was never "
            "checked. Component 1 occupies the even global "
            "indices and component 2 the odd ones, with "
            "dof(u^2) = dof(u^1) + 1 exactly, and that layout is "
            "IDENTICAL for ElementTriP1, ElementTriP2 and "
            "ElementTriP3 and for the tetrahedral family: "
            "basis.doflocs[:, 0::dim] reproduces the scalar "
            "basis doflocs for every component. You may still "
            "prefer two scalar bases for a block layout you "
            "control, but it buys layout preference, not "
            "correctness. Signal: assert the interleaving rather "
            "than avoiding it — check basis.N == dim * "
            "scalar_basis.N and that get_dofs(...).nodal['u^1'] "
            "and ['u^2'] differ by exactly one at every entry. "
            "Do NOT slice a vector solution as [:N] and [N:] "
            "expecting components; that is the BLOCKED layout, "
            "which skfem does not use, and it silently mixes the "
            "two components. (Verified empirically 2026-08-06 on "
            "skfem 12.0.1 — claim falsified.)",
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
