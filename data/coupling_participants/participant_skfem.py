"""scikit-fem participant for the OASiS `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
"""
import json
from pathlib import Path

import numpy as np
from skfem import (Basis, BilinearForm, ElementTriP1, FacetBasis, LinearForm,
                   MeshTri, condense, solve)
from skfem.helpers import dot, grad

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
#    this script gives the exact block for the RIGHT / Neumann side.
SIDE      = "dirichlet"   # "dirichlet" | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.6      # this subdomain
Y0, Y1    = 0.0, 0.4
IFACE_X   = 0.6           # shared interface (must be X0 or X1)
K         = 0.8           # conductivity
F_SRC     = 0.0           # volumetric source
T_OUTER   = 320.0         # Dirichlet value on the NON-interface x-boundary
NX, NY    = 24, 16        # this subdomain's own mesh
T_INIT    = 310.0          # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)   # interface is this side's x-max?
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0              # outward normal at interface = S * e_x
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None


def sample(imp, key, fallback, y):
    """Interpolate the partner's samples onto this participant's y-coordinates."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(y), float(fallback))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != ys.size:
        return np.full(len(y), float(fallback))
    o = np.argsort(ys)
    return np.interp(y, ys[o], vs[o])


imp = read_imports()

mesh = MeshTri.init_tensor(np.linspace(X0, X1, NX + 1),
                           np.linspace(Y0, Y1, NY + 1))
elem = ElementTriP1()
basis = Basis(mesh, elem)
n2d = basis.nodal_dofs[0]                  # node index -> dof (P1: identity)

px, py = mesh.p[0], mesh.p[1]
iface_n = np.where(np.abs(px - IFACE_X) < TOL)[0]
iface_n = iface_n[np.argsort(py[iface_n])]             # sorted by y
y_if = py[iface_n]
iface_dofs = n2d[iface_n]
outer_dofs = n2d[np.where(np.abs(px - OUTER_X) < TOL)[0]]


@BilinearForm
def stiffness(u, v, w):
    return K * dot(grad(u), grad(v))


@LinearForm
def source(v, w):
    return F_SRC * v


@LinearForm
def flux_load(v, w):
    return w["g"] * v


@BilinearForm
def mass(u, v, w):
    return u * v


@LinearForm
def proj_rhs(v, w):
    return (-K * S) * w["uh"].grad[0] * v


A = stiffness.assemble(basis)
b = source.assemble(basis)

sol = basis.zeros()
sol[outer_dofs] = T_OUTER
D = outer_dofs

if SIDE == "dirichlet":
    T_if = sample(imp, "values", T_INIT, y_if)
    sol[iface_dofs] = T_if
    D = np.concatenate([outer_dofs, iface_dofs])
else:
    q_if = sample(imp, "normal_fluxes", Q_INIT, y_if)
    gnod = basis.zeros()                   # P1 trace of the partner's samples
    gnod[iface_dofs] = q_if
    fbasis = FacetBasis(mesh, elem,
                        facets=mesh.facets_satisfying(
                            lambda p: np.abs(p[0] - IFACE_X) < TOL))
    b = b + flux_load.assemble(fbasis, g=fbasis.interpolate(gnod))
    # APPLY the partner's number unchanged (+ integral(g*v) ds_interface)

sol = solve(*condense(A, b, x=sol, D=D))

# outward normal flux density q_out = -k * S * dT/dx, L2-projected to P1
qh = solve(mass.assemble(basis),
           proj_rhs.assemble(basis, uh=basis.interpolate(sol)))

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
    "values": [float(t) for t in sol[iface_dofs]],
    "normal_fluxes": [float(q) for q in qh[iface_dofs]],
}, indent=2))
