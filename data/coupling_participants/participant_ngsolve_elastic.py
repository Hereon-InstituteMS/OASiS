"""NGSolve VECTOR participant for the OASiS `couple` driver.

Plane-strain linear elasticity  -div(sigma(u)) = 0  on ONE rectangular
subdomain of a domain split by a straight interface at x = IFACE_X. Unlike the
scalar (heat) participants, the exchanged interface state is a VECTOR on BOTH
channels:

    values        = displacement       u = (u_x, u_y)   at the interface nodes
    normal_fluxes = interface traction export            (SIGN CONVENTION below)

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

SIGN CONVENTION — the thing a vector coupling gets wrong silently.
`normal_fluxes` is exported as

    q_out = -(sigma . n_own)                       n_own = S * e_x

the SAME convention the shipped scalar participants use for heat
(q_out = -k dT/dn_own). The two sides' exports therefore CANCEL componentwise,
and the NEUMANN side applies the partner's numbers UNCHANGED
(`f += InnerProduct(g, v) * ds("interface")`), because the natural boundary
term of the elasticity weak form is +(sigma . n_own) . v = +q_out_partner . v.

NGSolve's VectorH1 is BLOCKED BY COMPONENT: GetDofNrs(NodeId(VERTEX, i))
returns (dof of u_x, dof of u_y) and those two indices are nv apart, not
adjacent. Writing nodal values with an adjacency assumption silently scatters
u_y into the u_x block.
"""
import json
from pathlib import Path

import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import (VERTEX, BilinearForm, CF, GridFunction, InnerProduct,
                     LinearForm, Mesh, NodeId, TaskManager, VectorH1, ds, dx,
                     grad)

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side.
SIDE      = "dirichlet"   # "dirichlet" (import u, export traction) | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.55     # this subdomain
Y0, Y1    = 0.0, 0.4
IFACE_X   = 0.55          # shared interface (must be X0 or X1)
E_MOD     = 1000.0        # Young's modulus
NU        = 0.3           # Poisson ratio (PLANE STRAIN)
# Prescribed displacement on this subdomain's WHOLE non-interface boundary
# (its outer x-face and both y-faces), as a polynomial in (x, y):
#     u_x = UDX[0] + UDX[1]*x + UDX[2]*y + UDX[3]*y*y
#     u_y = UDY[0] + UDY[1]*x + UDY[2]*y + UDY[3]*y*y
UDX = (0.0, 0.0, 0.0, 0.0)
UDY = (0.0, 0.0, 0.0, 0.0)
NX, NY    = 24, 16        # this subdomain's own mesh (netgen maxh derived below)
UI_X, UI_Y = 0.0, 0.0     # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # iteration-1 fallback interface traction export
# ─────────────────────────────────────────────────────────────────────────

MAXH = min((X1 - X0) / NX, (Y1 - Y0) / NY)    # netgen's scalar mesh size
ORDER = 1                                     # nodal == vertex dofs

LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))   # plane strain
MU = E_MOD / (2.0 * (1.0 + NU))

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
    """Map the partner's VECTOR samples onto this participant's y-coordinates,
    COMPONENT BY COMPONENT. One np.interp over a flattened (N, 2) array
    interleaves the components: right length, converges, every number wrong.

    Returns (len(y), ncomp)."""
    fb = np.asarray(fallback, float).ravel()
    if not imp or not imp.get("coordinates"):
        return np.tile(fb, (len(y), 1))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float)
    if vs.ndim == 1:
        vs = vs.reshape(-1, 1)
    if vs.shape[0] != ys.size or vs.shape[1] != fb.size:
        return np.tile(fb, (len(y), 1))
    o = np.argsort(ys)
    return np.column_stack([np.interp(y, ys[o], vs[o, c])
                            for c in range(vs.shape[1])])


imp = read_imports()

# ── mesh: SplineGeometry.AddRectangle edge order is bottom, right, top, left ──
# Everything that is not the interface carries the prescribed displacement, so
# it all gets the same boundary name.
geo = SplineGeometry()
geo.AddRectangle((X0, Y0), (X1, Y1),
                 bcs=(("outer", "interface", "outer", "outer") if ON_RIGHT else
                      ("outer", "outer", "outer", "interface")))
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

fes = VectorH1(mesh, order=ORDER,
               dirichlet=("outer|interface" if SIDE == "dirichlet" else "outer"))
u, v = fes.TnT()

# vertex -> (x-dof, y-dof). VectorH1 blocks by COMPONENT, so these are nv apart.
vdof = np.array([fes.GetDofNrs(NodeId(VERTEX, i))[:2] for i in range(mesh.nv)], int)
vxy = np.array([mesh.vertices[i].point for i in range(mesh.nv)], float)

iface_v = np.where(np.abs(vxy[:, 0] - IFACE_X) < TOL)[0]
iface_v = iface_v[np.argsort(vxy[iface_v, 1])]           # sorted by y
y_if = vxy[iface_v, 1]
outer_v = np.where((np.abs(vxy[:, 0] - OUTER_X) < TOL) |
                   (np.abs(vxy[:, 1] - Y0) < TOL) |
                   (np.abs(vxy[:, 1] - Y1) < TOL))[0]
# THE TWO INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES: they
# sit on a y-face, which carries a prescribed displacement in the un-split
# problem. Leaving them to the interface leaves them unconstrained on the
# Neumann side — the subproblem is still well posed, still converges, and lands
# a few percent off. They are still EXPORTED, just not interface-imposed.
corner = (np.abs(y_if - Y0) < TOL) | (np.abs(y_if - Y1) < TOL)


def eps_of(g):
    exx, eyy = g[0, 0], g[1, 1]
    return exx, eyy, 0.5 * (g[0, 1] + g[1, 0])


a = BilinearForm(fes)
gu, gv = grad(u), grad(v)
eu = eps_of(gu)
ev = eps_of(gv)
a += (2.0 * MU * (eu[0] * ev[0] + eu[1] * ev[1] + 2.0 * eu[2] * ev[2])
      + LAM * (eu[0] + eu[1]) * (ev[0] + ev[1])) * dx
f = LinearForm(fes)
f += CF((0.0, 0.0)) * v * dx

gfu = GridFunction(fes)                    # also carries the Dirichlet data
gfu.vec[:] = 0.0

if SIDE == "dirichlet":
    u_if = sample(imp, "values", (UI_X, UI_Y), y_if)
    for k, vtx in enumerate(iface_v):
        if corner[k]:
            continue
        gfu.vec[int(vdof[vtx, 0])] = float(u_if[k, 0])
        gfu.vec[int(vdof[vtx, 1])] = float(u_if[k, 1])
else:
    t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), y_if)
    gfun = GridFunction(fes)               # P1 trace of the partner's samples
    gfun.vec[:] = 0.0
    for k, vtx in enumerate(iface_v):
        gfun.vec[int(vdof[vtx, 0])] = float(t_if[k, 0])
        gfun.vec[int(vdof[vtx, 1])] = float(t_if[k, 1])
    # APPLY the partner's numbers UNCHANGED
    f += InnerProduct(gfun, v) * ds("interface")

# The outer boundary is written LAST so it wins at the two interface corners.
ox, oy = vxy[outer_v, 0], vxy[outer_v, 1]
oux = UDX[0] + UDX[1] * ox + UDX[2] * oy + UDX[3] * oy * oy
ouy = UDY[0] + UDY[1] * ox + UDY[2] * oy + UDY[3] * oy * oy
for k, vtx in enumerate(outer_v):
    gfu.vec[int(vdof[vtx, 0])] = float(oux[k])
    gfu.vec[int(vdof[vtx, 1])] = float(ouy[k])

with TaskManager():
    a.Assemble()
    f.Assemble()
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(fes.FreeDofs(),
                                  inverse="sparsecholesky") * res

    # q_out = -(sigma . n_own), L2-projected onto VectorH1 so its values live
    # on the same vertices as the interface DOFs.
    fesq = VectorH1(mesh, order=ORDER)
    p, w = fesq.TnT()
    m = BilinearForm(fesq)
    m += InnerProduct(p, w) * dx
    m.Assemble()
    exx, eyy, exy = eps_of(grad(gfu))
    sxx = 2.0 * MU * exx + LAM * (exx + eyy)
    sxy = 2.0 * MU * exy
    fq = LinearForm(fesq)
    fq += (-S) * (sxx * w[0] + sxy * w[1]) * dx
    fq.Assemble()
    qh = GridFunction(fesq)
    qh.vec.data = m.mat.Inverse(fesq.FreeDofs(),
                                inverse="sparsecholesky") * fq.vec

qdof = np.array([fesq.GetDofNrs(NodeId(VERTEX, int(i)))[:2] for i in iface_v], int)

Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": int(len(iface_v)),
    "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
    "values": [[float(gfu.vec[int(vdof[i, 0])]), float(gfu.vec[int(vdof[i, 1])])]
               for i in iface_v],
    "normal_fluxes": [[float(qh.vec[int(d0)]), float(qh.vec[int(d1)])]
                      for d0, d1 in qdof],
}, indent=2))
