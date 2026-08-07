"""FEniCSx (dolfinx) participant for the OASiS `couple` driver.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

Physics: steady conduction  -div(K grad T) = F_SRC  on one rectangular
subdomain of a domain split by a straight interface at x = IFACE_X.
Top and bottom edges are natural (zero-flux). The non-interface x-boundary
carries a Dirichlet value T_OUTER.
"""
import json
import sys
from pathlib import Path

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
#    this script gives the exact block for the RIGHT / Neumann side.
SIDE      = "dirichlet"   # "dirichlet" (import T, export flux) | "neumann"
PARTNER   = "right"       # the partner's `name` in your couple(...) call
X0, X1    = 0.0, 0.6      # this subdomain's x-extent
Y0, Y1    = 0.0, 0.4      # this subdomain's y-extent
IFACE_X   = 0.6           # the shared interface; must equal X0 or X1
K         = 0.8           # conductivity
F_SRC     = 0.0           # volumetric source
T_OUTER   = 320.0         # Dirichlet value on the NON-interface x-boundary
NX, NY    = 24, 16        # this subdomain's OWN mesh; need not match the partner
T_INIT    = 310.0         # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

OUTER_X = X0 if IFACE_X == X1 else X1
S = 1.0 if IFACE_X > OUTER_X else -1.0     # outward normal at interface = S*e_x


def read_imports():
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1,
    so the caller must fall back to an initial guess."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, y):
    """Map the partner's samples onto THIS participant's interface points.
    The driver does no interpolation — non-matching meshes are handled here."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(y), float(fallback))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != ys.size:
        return np.full(len(y), float(fallback))
    o = np.argsort(ys)
    return np.interp(y, ys[o], vs[o])


imp = read_imports()

domain = dmesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]],
                                [NX, NY], dmesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

xy = V.tabulate_dof_coordinates()
iface_dofs = np.where(np.abs(xy[:, 0] - IFACE_X) < 1e-10)[0]
iface_dofs = iface_dofs[np.argsort(xy[iface_dofs, 1])]   # constant order, always
y_if = xy[iface_dofs, 1]
if len(iface_dofs) == 0:
    sys.exit(f"no interface DOFs at x={IFACE_X}: this subdomain spans "
             f"[{X0},{X1}], so nothing is shared with the partner")

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = fem.Constant(domain, default_scalar_type(K)) * \
    ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = fem.Constant(domain, default_scalar_type(F_SRC)) * v * ufl.dx

outer = dmesh.locate_entities_boundary(domain, fdim,
                                       lambda x: np.isclose(x[0], OUTER_X))
bcs = [fem.dirichletbc(default_scalar_type(T_OUTER),
                       fem.locate_dofs_topological(V, fdim, outer), V)]

if SIDE == "dirichlet":
    g = fem.Function(V)
    g.x.array[iface_dofs] = sample(imp, "values", T_INIT, y_if)
    bcs.append(fem.dirichletbc(g, iface_dofs))
else:
    g = fem.Function(V)
    g.x.array[iface_dofs] = sample(imp, "normal_fluxes", Q_INIT, y_if)
    facets = dmesh.locate_entities_boundary(domain, fdim,
                                            lambda x: np.isclose(x[0], IFACE_X))
    tags = dmesh.meshtags(domain, fdim, np.sort(facets),
                          np.full(len(facets), 7, dtype=np.int32))
    ds_if = ufl.Measure("ds", domain=domain, subdomain_data=tags)(7)
    L += g * v * ds_if          # APPLY the partner's number UNCHANGED

uh = LinearProblem(a, L, bcs=bcs, petsc_options_prefix="cpl",
                   petsc_options={"ksp_type": "preonly",
                                  "pc_type": "lu"}).solve()

# Outward normal flux density q = -K * dT/dn, L2-projected onto CG1 so its
# values live on the same nodes as the interface DOFs.
p_, w_ = ufl.TrialFunction(V), ufl.TestFunction(V)
qh = LinearProblem(p_ * w_ * ufl.dx, -K * S * uh.dx(0) * w_ * ufl.dx,
                   petsc_options_prefix="flx",
                   petsc_options={"ksp_type": "preonly",
                                  "pc_type": "lu"}).solve()

T = uh.x.array[iface_dofs]
Q = qh.x.array[iface_dofs]
print(f"[fenics {SIDE}] interface n={len(T)} "
      f"T=[{T.min():.6g},{T.max():.6g}] q=[{Q.min():.6g},{Q.max():.6g}]")

# exports.json LAST: the driver takes its existence as proof of success.
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
    "values": [float(t) for t in T],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))
