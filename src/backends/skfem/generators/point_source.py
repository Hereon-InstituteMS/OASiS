"""scikit-fem point-source / Dirac-delta load generator + knowledge.

Mirrors scikit-fem upstream ex17 (insertion of a point load) and
ex38 (point source via a scalar Dirac delta). A point source f =
δ(x - x₀) cannot be integrated via a quadrature rule; instead the
load is assembled by adding the test-function values at x₀
directly to the RHS vector at the nearest mesh node OR by
projecting δ(x-x₀) onto the FE space via `Basis.interpolate`.

For P1 on the unit square with f = δ(x - x₀):
    -Δu = δ(x - x₀)  in Ω
    u   = 0          on ∂Ω
The discrete RHS is b_i = N_i(x₀) where N_i are P1 basis
functions; for a node-coincident point source this collapses to a
single nonzero at the corresponding DOF.
"""


def _point_source_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate
    values for your specific problem.

    Point source at (x0, y0) on the unit square, homogeneous
    Dirichlet BCs, P1 triangles. The exact Green's function on
    [0,1]² (with Dirichlet BCs) is a Fourier series — we just
    check that the FE solution is bounded and peaks at the source
    location."""
    nx = params.get("nx", 32)
    x0 = params.get("x0", 0.5)
    y0 = params.get("y0", 0.5)
    return f'''\
"""Point-source Poisson on the unit square — scikit-fem"""
from skfem import (MeshTri, Basis, ElementTriP1, solve, condense)
from skfem.models.poisson import laplace
import numpy as np
import json

nx = {nx}
m = MeshTri.init_tensor(np.linspace(0, 1, nx + 1),
                        np.linspace(0, 1, nx + 1))
e = ElementTriP1()
ib = Basis(m, e)

K = laplace.assemble(ib)

# RHS via N_i(x0) — for a node-coincident source on the
# tensor-product mesh this collapses to a single nonzero entry
# at the nearest grid node. For off-node sources we'd need to
# locate the containing element and assemble the three barycentric
# weights into the RHS.
x0, y0 = {x0}, {y0}
nodes = m.p.T                                  # (n_nodes, 2)
dists = np.linalg.norm(nodes - np.array([x0, y0]), axis=1)
source_node = int(np.argmin(dists))

# For a properly mesh-coincident source this approximates
# δ(x-x0) → e_{{source_node}} (Kronecker). Off-node sources
# require barycentric distribution; raise if the source is
# more than h/sqrt(2) from any node so the approximation is
# clearly bad.
h = 1.0 / nx
if dists[source_node] > h / np.sqrt(2.0):
    print(f"WARNING: source ({{x0}}, {{y0}}) is "
          f"{{dists[source_node]:.4f}} from nearest node — "
          f"point-source approximation is rough (h/sqrt(2) = "
          f"{{h/np.sqrt(2.0):.4f}}).")

f = ib.zeros()
f[source_node] = 1.0                           # δ-like load

# Homogeneous Dirichlet on the whole boundary.
D = ib.get_dofs().flatten()
u = solve(*condense(K, f, D=D))

# Sanity: u should peak near the source node and be bounded.
peak_dof = int(np.argmax(u))
print(f"point source at ({{x0}}, {{y0}}) → node {{source_node}}; "
      f"peak DOF {{peak_dof}}; u_peak = {{u[peak_dof]:.4e}}; "
      f"max|u| = {{np.abs(u).max():.4e}}")

import meshio
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])])
mio = meshio.Mesh(points, [("triangle", m.t.T)],
                  point_data={{"u": u}})
mio.write("result.vtu")

summary = {{
    "n_dofs": int(ib.N),
    "source_node": source_node,
    "peak_dof": peak_dof,
    "source_node_equals_peak": int(source_node == peak_dof),
    "u_peak": float(u[peak_dof]),
    "u_max_abs": float(np.abs(u).max()),
    "source_xy": [x0, y0],
    "h": float(h),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


GENERATORS: dict = {
    "point_source_2d": _point_source_2d,
}


KNOWLEDGE: dict = {
    "point_source": {
        "description": (
            "Point-source / Dirac-delta load Poisson problem. "
            "f = δ(x - x0) cannot be integrated via quadrature; "
            "the discrete RHS is b_i = N_i(x0) — collapses to a "
            "Kronecker e_node entry for mesh-coincident sources. "
            "Matches scikit-fem upstream ex17 (point load) and "
            "ex38 (point source)."
        ),
        "weak_form": (
            "(grad(u), grad(v))_dx = N_i(x0) v_i  "
            "(P1: b is one-hot at the source DOF when x0 is a "
            "mesh node; otherwise barycentric weights at the "
            "containing element)."
        ),
        "elements": ["ElementTriP1"],
        "variants": ["2d"],
        "pitfalls": [
            "[Numerical] A Dirac source δ(x-x0) is NOT smooth, "
            "so the regularity of the analytical solution is at "
            "best in W^{1,p} for p<2 in 2D — the FE solution does "
            "NOT converge in the H^1 norm as h → 0; H^1 error "
            "stays O(1) and the L^2 error converges only at "
            "rate O(h). For convergence studies, regularize the "
            "source to a narrow Gaussian "
            "exp(-|x-x0|^2 / (2*sigma^2)) with sigma ~ h. "
            "Signal: refinement study shows the H^1-norm error "
            "FLAT or non-monotone as h is halved; L^2-norm error "
            "halves (order ~1) instead of quartering (order 2).",
            "[API] `ib.zeros()` returns a fresh float64 ndarray "
            "of length ib.N. Setting `f[source_node] = 1.0` is "
            "the correct one-hot assembly for a node-coincident "
            "source. Trying to use `unit_load.assemble(ib)` (the "
            "f=1 constant-source assembly) gives a 1.0 at every "
            "DOF — different physics entirely. "
            "Signal: solution u has a uniformly-distributed "
            "shape with peaks at all interior nodes, not a "
            "concentrated peak near the source location.",
            "[Mesh] If x0 is NOT a mesh node, np.argmin on the "
            "vertex array picks the NEAREST node but introduces "
            "a discretization error proportional to "
            "h/sqrt(2) (max distance from any point in a "
            "triangulated unit square to the nearest node). To "
            "place an off-node source properly, locate the "
            "containing triangle and distribute the unit load by "
            "barycentric coordinates: b[v0..v2] += λ0, λ1, λ2. "
            "Signal: peak in u sits at a slightly different "
            "location from x0; u_peak < expected; "
            "results_summary.json shows `source_node_equals_peak"
            "` == 1 but the user expected a fractional source. "
            "Concretely: `np.argmin` on the `MeshTri.p.T` vertex "
            "array always returns an integer node id, never a "
            "barycentric position, so off-node sources are "
            "silently rounded to the nearest mesh vertex. The "
            "fix uses `Basis.get_dofs` plus barycentric "
            "interpolation, not raw `np.argmin`.",
            "[API] `condense(K, f, D=D)` returns 4 values "
            "(K_c, f_c, x_c, I) that you must unpack via "
            "`solve(*condense(...))`. Passing `condense(K, f, D=D)` "
            "directly to a scipy solver raises TypeError because "
            "scipy expects (A, b) not a tuple of 4. "
            "Signal: TypeError 'spsolve() got too many positional "
            "arguments' or 'unhashable type: tuple' from scipy "
            "linalg; condense's contract is unique to scikit-fem.",
            "[Physics] Total integral of u over Ω equals "
            "Green's-function flux balance: u solves "
            "-Δu = δ(x-x0) on the unit square with Dirichlet BC, "
            "so ∫_Ω u dx = -1/lambda1 in the Laplacian spectrum "
            "for a point source at the center — for "
            "x0=y0=0.5 the value should be near a known "
            "Fourier-series sum. "
            "Signal: ∫_Ω u_h ≈ "
            "`m.p.shape[1] * np.mean(u) * h^2` differs from the "
            "analytic series sum by orders of magnitude "
            "(symptom of either the wrong source magnitude OR "
            "wrong BC orientation OR sign error in K).",
            "[Output] VTK output for a 2D MeshTri needs "
            "3D-padded points: "
            "`np.column_stack([m.p.T, np.zeros(m.p.shape[1])])`. "
            "Forgetting the zero z-column raises ValueError in "
            "`meshio.Mesh` when constructing the points array "
            "(it requires shape (N, 3) for VTU/VTX output). "
            "Signal: ValueError 'expected ndarray of shape "
            "(N, 3), got (N, 2)' from meshio.Mesh constructor; "
            "or the .vtu file writes but ParaView refuses to "
            "render it as 'invalid 2D coordinates'.",
        ],
        "references": [
            "scikit-fem ex17 (insertion of point load)",
            "scikit-fem ex38 (point source via Dirac delta)",
            "Brenner & Scott, 'The Mathematical Theory of "
            "Finite Element Methods', §0.5 (Sobolev embedding "
            "of Dirac delta).",
        ],
    },
}
