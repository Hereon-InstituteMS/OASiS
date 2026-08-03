"""NGSolve participant for the OASiS `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
"""
import json
from pathlib import Path

import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import (VERTEX, BilinearForm, CoefficientFunction, GridFunction,
                     H1, LinearForm, Mesh, NodeId, TaskManager, ds, dx, grad)

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
NX, NY    = 24, 16        # this subdomain's own mesh (netgen maxh derived below)
T_INIT    = 310.0          # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

MAXH  = min((X1 - X0) / NX, (Y1 - Y0) / NY)   # netgen's scalar mesh size
ORDER = 1                                     # H1 order (nodal == vertex dofs)

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

# ── mesh: SplineGeometry.AddRectangle edge order is bottom, right, top, left ──
geo = SplineGeometry()
geo.AddRectangle((X0, Y0), (X1, Y1),
                 bcs=(("bottom", "interface", "top", "outer") if ON_RIGHT else
                      ("bottom", "outer", "top", "interface")))
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

fes = H1(mesh, order=ORDER,
         dirichlet=("outer|interface" if SIDE == "dirichlet" else "outer"))
u, v = fes.TnT()

# vertex -> dof map and the interface / outer vertex sets (ORDER 1: dof per vertex)
vdof = np.array([fes.GetDofNrs(NodeId(VERTEX, i))[0] for i in range(mesh.nv)], int)
vxy = np.array([mesh.vertices[i].point for i in range(mesh.nv)], float)

iface_v = np.where(np.abs(vxy[:, 0] - IFACE_X) < TOL)[0]
iface_v = iface_v[np.argsort(vxy[iface_v, 1])]           # sorted by y
y_if = vxy[iface_v, 1]
iface_dofs = vdof[iface_v]
outer_dofs = vdof[np.where(np.abs(vxy[:, 0] - OUTER_X) < TOL)[0]]

a = BilinearForm(fes)
a += K * grad(u) * grad(v) * dx
f = LinearForm(fes)
f += CoefficientFunction(F_SRC) * v * dx

gfu = GridFunction(fes)                    # also carries the Dirichlet data
gfu.vec[:] = 0.0
for d in outer_dofs:
    gfu.vec[int(d)] = T_OUTER

if SIDE == "dirichlet":
    T_if = sample(imp, "values", T_INIT, y_if)
    for d, t in zip(iface_dofs, T_if):
        gfu.vec[int(d)] = float(t)
else:
    q_if = sample(imp, "normal_fluxes", Q_INIT, y_if)
    gfun = GridFunction(fes)               # P1 trace of the partner's samples
    gfun.vec[:] = 0.0
    for d, q in zip(iface_dofs, q_if):
        gfun.vec[int(d)] = float(q)
    f += gfun * v * ds("interface")        # APPLY the partner's number unchanged

with TaskManager():
    a.Assemble()
    f.Assemble()
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(fes.FreeDofs(),
                                  inverse="sparsecholesky") * res

    # outward normal flux density q_out = -k * S * dT/dx, L2-projected to H1
    fesq = H1(mesh, order=ORDER)
    p, w = fesq.TnT()
    m = BilinearForm(fesq)
    m += p * w * dx
    m.Assemble()
    fq = LinearForm(fesq)
    fq += (-K * S) * grad(gfu)[0] * w * dx
    fq.Assemble()
    qh = GridFunction(fesq)
    qh.vec.data = m.mat.Inverse(fesq.FreeDofs(),
                                inverse="sparsecholesky") * fq.vec

qdofs = np.array([fesq.GetDofNrs(NodeId(VERTEX, int(i)))[0] for i in iface_v], int)

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_v)),
    "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
    "values": [float(gfu.vec[int(d)]) for d in iface_dofs],
    "normal_fluxes": [float(qh.vec[int(d)]) for d in qdofs],
}, indent=2))
