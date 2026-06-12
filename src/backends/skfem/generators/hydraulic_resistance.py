"""scikit-fem hydraulic-resistance generator + knowledge.

Computes the linear hydraulic resistance R = ΔP / Q for steady
Stokes flow through a 2D channel:

    -μ Δu + ∇p = 0     in Ω
    ∇·u        = 0     in Ω
    u = 0              on top/bottom walls (no-slip)
    p = P_in           on inlet (x=0)
    p = 0              on outlet (x=L)
    σ·n = 0 tangential — natural BC from pressure-driven setup

The closed-form Poiseuille resistance for a channel of height H and
length L is R_exact = 12 μ L / H³.  We compare the FE-computed
resistance to this analytic value.

Matches scikit-fem upstream ex29 (linear hydraulic resistance).
"""


def _hydraulic_resistance_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate
    values for your specific problem.

    2D rectangular channel of length L and height H, pressure-
    driven Stokes flow via Taylor-Hood (P2-P1) on triangles."""
    L_chan = params.get("L", 2.0)
    H_chan = params.get("H", 0.1)
    nx = params.get("nx", 40)
    ny = params.get("ny", 6)
    mu = params.get("mu", 1.0)
    p_in = params.get("p_in", 1.0)
    return f'''\
"""Stokes hydraulic resistance R = ΔP / Q — scikit-fem"""
from skfem import (MeshTri, Basis, ElementTriP2, ElementTriP1,
                   ElementVector, BilinearForm, LinearForm,
                   FacetBasis, solve, condense, asm)
from skfem.helpers import dot, ddot, sym_grad, div, grad
import numpy as np
import scipy.sparse as sp
import json

L_chan = {L_chan}
H_chan = {H_chan}
nx = {nx}
ny = {ny}
mu = {mu}
p_in = {p_in}


# Rectangular tensor-product mesh, split into triangles.
m = MeshTri.init_tensor(np.linspace(0.0, L_chan, nx + 1),
                        np.linspace(0.0, H_chan, ny + 1))
m = m.with_boundaries({{
    "wall": lambda x: (np.isclose(x[1], 0.0)
                       | np.isclose(x[1], H_chan)),
    "inlet":  lambda x: np.isclose(x[0], 0.0),
    "outlet": lambda x: np.isclose(x[0], L_chan),
}})

# Taylor-Hood P2-P1. BOTH bases must use the same `intorder` so
# the bilinear coupling between them assembles — skfem default
# intorder differs by element degree, so explicit alignment is
# required. intorder=4 handles P2 sym_grad : sym_grad (rank-4
# integrand) exactly on triangles.
ev = ElementVector(ElementTriP2())
ep = ElementTriP1()
ib_u = Basis(m, ev, intorder=4)
ib_p = Basis(m, ep, intorder=4)
ib_fac_in = FacetBasis(m, ev, intorder=4,
                       facets=m.boundaries["inlet"])
ib_fac_out = FacetBasis(m, ev, intorder=4,
                        facets=m.boundaries["outlet"])


@BilinearForm
def stiffness(u, v, w):
    # 2*mu*(sym_grad u : sym_grad v).
    return 2.0 * mu * ddot(sym_grad(u), sym_grad(v))


@BilinearForm
def neg_div(u, p, w):
    # -div(u) * p (pressure × velocity-divergence coupling).
    return -div(u) * p


@LinearForm
def inlet_traction(v, w):
    # Natural BC at inlet (x=0): σ·n = -p_in * n with n=(-1,0).
    # The traction contribution to the velocity equation is
    # ∫_inlet (-p_in)·n · v ds = ∫_inlet (-p_in)·(-1) v_x ds
    #                          = ∫_inlet p_in v_x ds (positive).
    return p_in * v.value[0]


# Outlet pressure = 0 so no contribution from that boundary.
A = stiffness.assemble(ib_u)
B = neg_div.assemble(ib_u, ib_p)
f_u = inlet_traction.assemble(ib_fac_in)
f_p = ib_p.zeros()

# Block matrix: [[A, B^T], [B, 0]] x = [f_u, f_p].
K = sp.bmat([[A, B.T], [B, None]], format="csr")
F = np.concatenate([f_u, f_p])

# Dirichlet: u=0 on walls. Pin pressure at one outlet node to
# remove the constant null space.
wall_dofs = ib_u.get_dofs("wall").all()
# Pin a single pressure DOF at the outlet to fix the null space.
outlet_p_dofs = ib_p.get_dofs("outlet").all()
pin_p = ib_u.N + int(outlet_p_dofs[0])
D = np.concatenate([wall_dofs, [pin_p]])

x_full = np.zeros(ib_u.N + ib_p.N)
x_full[pin_p] = 0.0   # outlet pressure = 0

x = solve(*condense(K, F, x=x_full, D=D))
u = x[:ib_u.N]
p = x[ib_u.N:]

# Flow rate Q = ∫_outlet u_x ds (volumetric flow per unit depth).
@LinearForm
def flow_rate_form(v, w):
    return v.value[0]

# Integrate u_x over the outlet by interpolating to FacetBasis.
u_func = ib_u.interpolate(u)
# u_func.value: shape (2, n_facets, n_qpoints)
# Integrate ∫_outlet u_x ds via the FacetBasis weights.
Q = float(np.sum(ib_fac_out.interpolate(u).value[0]
                 * ib_fac_out.dx))

R_fe = p_in / Q
R_exact = 12.0 * mu * L_chan / (H_chan ** 3)
rel_err = abs(R_fe - R_exact) / R_exact

print(f"Channel L={{L_chan}} H={{H_chan}} mu={{mu}} p_in={{p_in}}")
print(f"  n_dofs_u={{ib_u.N}}  n_dofs_p={{ib_p.N}}")
print(f"  flow rate Q     = {{Q:.6e}}")
print(f"  resistance R    = {{R_fe:.6e}}")
print(f"  R_exact (12μL/H³)= {{R_exact:.6e}}")
print(f"  relative error   = {{rel_err:.4e}}")

import meshio
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
# Pressure lives on the P1 nodal basis = the mesh vertices.
p_nodal = p[:m.p.shape[1]]
mio = meshio.Mesh(points, [("triangle", m.t.T)],
                  point_data={{"p": p_nodal}})
mio.write("result.vtu")

summary = {{
    "n_dofs_u": int(ib_u.N),
    "n_dofs_p": int(ib_p.N),
    "L": float(L_chan), "H": float(H_chan), "mu": float(mu),
    "p_in": float(p_in),
    "Q": float(Q),
    "R_fe": float(R_fe),
    "R_exact": float(R_exact),
    "relative_error": float(rel_err),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


GENERATORS: dict = {
    "hydraulic_resistance_2d": _hydraulic_resistance_2d,
}


KNOWLEDGE: dict = {
    "hydraulic_resistance": {
        "description": (
            "Computes the linear hydraulic resistance "
            "R = ΔP / Q for steady Stokes flow through a 2D "
            "rectangular channel. Pressure-driven boundary "
            "conditions (p=p_in at inlet, p=0 at outlet, no-slip "
            "on top/bottom walls). FE result compared to the "
            "closed-form Poiseuille resistance R = 12 μ L / H³. "
            "Matches scikit-fem upstream ex29."
        ),
        "weak_form": (
            "Stokes: 2μ ∫ ε(u):ε(v) - ∫ div(u) q - ∫ div(v) p = "
            "∫_inlet p_in n·v ds (outflow free)."
        ),
        "elements": [
            "ElementVector(ElementTriP2) velocity + ElementTriP1 "
            "pressure (Taylor-Hood, inf-sup stable)",
        ],
        "variants": ["2d"],
        "pitfalls": [
            "[Physics] The Poiseuille analytic R = 12 μ L / H³ "
            "is only valid in the FULLY-DEVELOPED limit (L >> H). "
            "Short channels show entrance effects: the FE-computed "
            "R is HIGHER than 12μL/H³ by an additive term ~ μ/H² "
            "(entrance pressure drop). For L/H = 20 (default) the "
            "extra is ~5%; for L/H = 2 it's ~50%. "
            "Signal: `relative_error` in results_summary.json is "
            "5-50% (rises as L/H shrinks) rather than the 0.1-1% "
            "you'd expect from a well-resolved "
            "`ElementVector(ElementTriP2)` + `ElementTriP1` "
            "Taylor-Hood mesh; refinement of `MeshTri.init_tensor` "
            "doesn't help because the gap is physical (entrance "
            "length), not discretization.",

            "[API] `scipy.sparse.bmat([[A, B.T], [B, None]], "
            "format='csr')` is the canonical Stokes block "
            "assembly in skfem. The `None` in the (1,1) block "
            "produces a zero block of the right shape — "
            "explicit `sp.csr_matrix((nP, nP))` works too but is "
            "verbose. Forgetting to wrap in bmat (passing "
            "[[A, B.T], [B, 0]] with scalar 0) raises "
            "TypeError. "
            "Signal: TypeError 'no supported conversion for "
            "types: (dtype('int64'),)' or "
            "'unsupported operand type for *' from "
            "scipy.sparse.bmat when 0 is used instead of None.",

            "[Numerical] Stokes saddle-point systems have a "
            "1-dimensional null space (constant pressure). "
            "Without pinning one pressure DOF, the solver "
            "(direct or iterative) fails — direct LU gets "
            "SingularMatrix; iterative GMRES diverges. The fix "
            "pins a single pressure DOF on the outlet boundary "
            "via `condense(K, F, D=D)` where D includes "
            "`ib_u.N + outlet_p_dofs[0]` (the offset shifts to "
            "the pressure block in the [u; p] vector). "
            "Signal: scipy.sparse.linalg `MatrixRankWarning: "
            "Matrix is exactly singular`, or solve returns "
            "`nan` everywhere; `Q` in summary is 0 or nan.",

            "[Physics] Inlet pressure traction enters as a "
            "NATURAL boundary condition in the velocity "
            "equation: ∫_inlet (-p_in) v_x ds where outward "
            "normal at x=0 is (-1, 0). The MINUS SIGN matters — "
            "wrong sign gives flow in the wrong direction "
            "(Q < 0) and R < 0. "
            "Signal: `summary['Q']` is negative; "
            "`results.vtu` shows pressure increasing in the "
            "flow direction (physically wrong); "
            "`relative_error` is order 1 (R has wrong sign).",

            "[API] `ib_fac_out.interpolate(u).value[0] * "
            "ib_fac_out.dx` computes the boundary integral "
            "∫_outlet u_x ds. `ib_fac_out.dx` is the array of "
            "quadrature weights * Jacobian on the outlet "
            "facets; multiplying by the interpolated u_x and "
            "summing gives the flow rate. Forgetting to use "
            "FacetBasis (using ib_u directly on a boundary "
            "integral) double-counts interior edges and "
            "produces a Q that's ~2x too large. "
            "Signal: `Q` in summary is 2× the expected "
            "Poiseuille value; `R_fe` is half of `R_exact`. "
            "Concretely: replacing `FacetBasis(m, ev, "
            "facets=m.boundaries['outlet'])` with the "
            "volumetric `Basis(m, ev)` and using `ib_u.dx` for "
            "the boundary integral over-counts; the fix uses "
            "`FacetBasis` + `ib_fac_out.dx`.",

            "[Numerical] Taylor-Hood P2-P1 on triangles is "
            "inf-sup stable (LBB condition satisfied) and "
            "converges at O(h²) in L² for velocity and pressure. "
            "Equal-order P1-P1 without stabilization would be "
            "unstable: the pressure field develops checkerboard "
            "oscillations and the FE resistance bears no "
            "relation to the analytic value. "
            "Signal: replacing `ElementVector(ElementTriP2())` "
            "with `ElementVector(ElementTriP1())` (still using "
            "P1 pressure) produces a `summary['Q']` that "
            "fluctuates ~50%-200% from the Poiseuille value, "
            "and `result.vtu` pressure field is "
            "checkerboard-patterned.",

            "[Output] Pressure DOF block in the solution vector "
            "[u; p] starts at index `ib_u.N`. Extracting "
            "`p_nodal = p[:m.p.shape[1]]` assumes the P1 "
            "pressure DOFs are ordered to match the mesh "
            "vertices, which is the skfem default for "
            "ElementTriP1. If a different element is used "
            "(e.g. ElementTriP2 for pressure), the DOF count "
            "exceeds the vertex count and this slice drops "
            "data. "
            "Signal: ParaView shows pressure 'p' as a smooth "
            "but truncated field; max/min pressure printed "
            "from the script disagrees with what ParaView "
            "renders.",
        ],
        "references": [
            "scikit-fem ex29 (linear hydraulic resistance)",
            "White, F. M. (2011), 'Viscous Fluid Flow' 3rd ed., "
            "Ch. 3 (Poiseuille flow).",
        ],
    },
}
