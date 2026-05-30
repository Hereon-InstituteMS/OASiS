"""scikit-fem linear elasticity generators and knowledge."""


def _elasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Linear elasticity on rectangular domain, fixed left edge, body force."""
    nx = params.get("nx", 40)
    ny = params.get("ny", 4)
    lx = params.get("lx", 10.0)
    ly = params.get("ly", 1.0)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    return f'''\
"""Linear elasticity: rectangular domain, fixed left — scikit-fem"""
from skfem import *
from skfem.models.elasticity import linear_elasticity, lame_parameters
import numpy as np
import json

_tol = 1e-10
m = (MeshQuad.init_tensor(np.linspace(0, {lx}, {nx+1}), np.linspace(0, {ly}, {ny+1}))
     .with_boundaries({{"left": lambda x: x[0] < _tol}}))
e = ElementVector(ElementQuad1())
ib = Basis(m, e)

lam, mu = lame_parameters({E}, {nu})
K = linear_elasticity(lam, mu).assemble(ib)

# Body force — set for your problem
@LinearForm
def body_force(v, w):
    return -1.0 * v[1]

f = body_force.assemble(ib)

# Fix left edge.  In scikit-fem >= 8 the boundary lookup requires the
# mesh to have been tagged via `with_boundaries({{...}})` (done above)
# so `get_dofs("left")` resolves to the tagged facets.  Without the
# tag the call raises because the mesh has no "left" facet group.
D = ib.get_dofs("left").flatten()
u = solve(*condense(K, f, D=D))

# Tip displacement.  `ElementVector(ElementQuad1())` stores DOFs
# *interleaved* (x0, y0, x1, y1, ...) — naively reshaping the flat
# solution as `(2, -1)` would scramble x and y across rows.  Recover
# the per-component arrays from the basis's `nodal_dofs` map, which
# records the global-DOF index of each (component, node) pair.
nodal_dofs = ib.nodal_dofs   # shape (n_components, n_nodes)
ux = u[nodal_dofs[0]]
uy = u[nodal_dofs[1]]
max_uy = uy.min()
print(f"Max tip displacement (uy.min): {{max_uy:.6f}}")

# Write a VTU so the sweep harness (and any other downstream consumer)
# can recover the displacement field, not just the scalar summary.
try:
    import meshio
    cells = [("quad", m.t.T)]
    points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])]) if m.p.shape[0] == 2 else m.p.T
    # node ordering of `points` matches `m.p.T`, which is also what
    # `nodal_dofs` is indexed by (one column per node), so the
    # per-node (ux, uy) pairs line up with the corresponding row of
    # `points` without any extra reordering.
    displacement = np.column_stack([ux, uy])           # (n_nodes, 2)
    displacement_3 = np.column_stack([displacement, np.zeros(displacement.shape[0])])
    meshio.Mesh(points, cells, point_data={{"displacement": displacement_3}}).write("result.vtu")
except Exception as _e:
    print(f"VTU write skipped: {{_e}}")

summary = {{"max_displacement_y": float(max_uy), "n_dofs": int(K.shape[0])}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "linear_elasticity": {
        "description": "Linear elasticity: plane strain/stress, 3D (examples 02, 03, 11, 21)",
        "solver": "Direct sparse (scipy.sparse.linalg.spsolve)",
        "elements": "ElementVector(ElementQuad1()) or ElementVector(ElementTriP1())",
        "built_in_forms": "linear_elasticity, linear_stress (from skfem.models.elasticity)",
        "pitfalls": [
            "Use ElementVector(ElementQuad1()) for vector problems, NOT ElementQuad1() alone",
            "lame_parameters(E, nu) computes lambda and mu from engineering constants",
            "linear_elasticity(lam, mu) returns a BilinearForm directly",
            "For eigenvalue problems (vibration): use eigsh(K, M=M, k=n, sigma=0)",
        ],
    },
}

GENERATORS = {
    "linear_elasticity_2d": _elasticity_2d,
}
