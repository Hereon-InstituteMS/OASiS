"""DUNE-fem participant for the OASiS `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
"""
import json
from pathlib import Path

import numpy as np
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC, Constant
from ufl import (TrialFunction, TestFunction, SpatialCoordinate,
                 conditional, dot, ds, dx, grad, lt)

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
#    this script gives the exact block for the RIGHT / Neumann side.
SIDE      = "dirichlet"   # "dirichlet" | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.6
Y0, Y1    = 0.0, 0.4
IFACE_X   = 0.6
K         = 0.8
F_SRC     = 0.0           # volumetric source
T_OUTER   = 320.0
NX, NY    = 24, 16
T_INIT    = 310.0
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

OUTER_X = X0 if IFACE_X == X1 else X1
S = 1.0 if IFACE_X > OUTER_X else -1.0     # outward normal at interface = S * e_x
EPS = 1e-8                                 # boundary-indicator tolerance


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

gridView = structuredGrid([X0, Y0], [X1, Y1], [NX, NY])
space = lagrange(gridView, order=1)
x = SpatialCoordinate(space)

# dof -> coordinate map: nodal interpolation of the coordinate functions
xd = np.array(space.interpolate(x[0], name="xcoord").as_numpy)
yd = np.array(space.interpolate(x[1], name="ycoord").as_numpy)
iface_dofs = np.where(np.abs(xd - IFACE_X) < 1e-10)[0]
iface_dofs = iface_dofs[np.argsort(yd[iface_dofs])]
y_if = yd[iface_dofs]

u, v = TrialFunction(space), TestFunction(space)
a = K * dot(grad(u), grad(v)) * dx
# zero source must stay symbolic: a bare `0*v*dx` folds to a domainless Zero
b = Constant(F_SRC, name="f_src") * v * dx

bcs = [DirichletBC(space, T_OUTER, conditional(lt(abs(x[0] - OUTER_X), EPS), 1, 0))]

# coupling data carrier: a discrete function whose interface dofs hold the
# imported samples (the FORM never changes between iterations -> no re-JIT)
gfun = space.interpolate(0, name="iface_data")
gdofs = gfun.as_numpy
gdofs[:] = 0.0

if SIDE == "dirichlet":
    gdofs[iface_dofs] = sample(imp, "values", T_INIT, y_if)
    bcs.append(DirichletBC(space, gfun,
                           conditional(lt(abs(x[0] - IFACE_X), EPS), 1, 0)))
else:
    gdofs[iface_dofs] = sample(imp, "normal_fluxes", Q_INIT, y_if)
    # APPLY the partner's number unchanged: + int(g*v) ds on the interface
    b = b + conditional(lt(abs(x[0] - IFACE_X), EPS), gfun * v, 0.0) * ds

scheme = galerkin([a == b] + bcs, solver="cg")
uh = space.interpolate(0, name="temperature")
scheme.solve(target=uh)

# outward normal flux density q_out = -k * S * dT/dx, L2-projected to CG1
p, w = TrialFunction(space), TestFunction(space)
proj = galerkin([p * w * dx == -K * S * grad(uh)[0] * w * dx], solver="cg")
qh = space.interpolate(0, name="normal_flux")
proj.solve(target=qh)

T_dofs = np.array(uh.as_numpy)
q_dofs = np.array(qh.as_numpy)

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
    "values": [float(t) for t in T_dofs[iface_dofs]],
    "normal_fluxes": [float(q) for q in q_dofs[iface_dofs]],
}, indent=2))
