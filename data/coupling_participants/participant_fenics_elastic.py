"""FEniCSx (dolfinx) VECTOR participant for the OASiS `couple` driver.

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
(q_out = -k dT/dn_own). Two consequences, both load-bearing:

  * the two sides' exports CANCEL componentwise, because n_own is anti-parallel
    across the interface — that is what makes the interface balance check a
    conservation statement rather than an accident;
  * the NEUMANN side applies the partner's numbers UNCHANGED, as
    `L += inner(g, v) * ds`, because the natural boundary term of the
    elasticity weak form is +(sigma . n_own) . v = +q_out_partner . v.

Exporting the raw traction (sigma . n_own) instead flips the sign the Neumann
side applies; the iteration still converges, to the wrong answer.
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
#    As shipped this is the LEFT / Dirichlet side.
SIDE      = "dirichlet"   # "dirichlet" (import u, export traction) | "neumann"
PARTNER   = "right"       # the partner's `name` in your couple(...) call
X0, X1    = 0.0, 0.55     # this subdomain's x-extent
Y0, Y1    = 0.0, 0.4      # this subdomain's y-extent
IFACE_X   = 0.55          # the shared interface; must equal X0 or X1
E_MOD     = 1000.0        # Young's modulus
NU        = 0.3           # Poisson ratio (PLANE STRAIN)
# Prescribed displacement on this subdomain's WHOLE non-interface boundary
# (its outer x-face and both y-faces), as a polynomial in (x, y):
#     u_x = UDX[0] + UDX[1]*x + UDX[2]*y + UDX[3]*y*y
#     u_y = UDY[0] + UDY[1]*x + UDY[2]*y + UDY[3]*y*y
# The two subdomains must agree at the two interface corners, or the coupled
# problem is not the un-split one.
UDX = (0.0, 0.0, 0.0, 0.0)
UDY = (0.0, 0.0, 0.0, 0.0)
NX, NY    = 24, 16        # this subdomain's OWN mesh; need not match the partner
UI_X, UI_Y = 0.0, 0.0     # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # iteration-1 fallback interface traction export
# ─────────────────────────────────────────────────────────────────────────

LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))   # plane strain
MU = E_MOD / (2.0 * (1.0 + NU))

OUTER_X = X0 if abs(IFACE_X - X1) < abs(IFACE_X - X0) else X1
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
    """Map the partner's VECTOR samples onto THIS participant's interface
    points, COMPONENT BY COMPONENT.

    The driver does no interpolation — non-matching interface meshes are
    handled here, and for a vector field that has to be done per component. One
    np.interp over a flattened (N, 2) array interleaves the two components: the
    result still has the right length, the coupling still converges, and every
    number is wrong.

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

domain = dmesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]],
                                [NX, NY], dmesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1, (2,)))
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

# tabulate_dof_coordinates() has ONE ROW PER NODE (dof block); the scalar
# array index of component c at node n is n*2 + c.
xy = V.tabulate_dof_coordinates()
iface_n = np.where(np.abs(xy[:, 0] - IFACE_X) < 1e-10)[0]
iface_n = iface_n[np.argsort(xy[iface_n, 1])]            # constant order, always
y_if = xy[iface_n, 1]
if len(iface_n) == 0:
    sys.exit(f"no interface DOFs at x={IFACE_X}: this subdomain spans "
             f"[{X0},{X1}], so nothing is shared with the partner")
# THE TWO INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES.
# (IFACE_X, Y0) and (IFACE_X, Y1) sit on a y-face, which carries a prescribed
# displacement in the un-split problem, so they stay Dirichlet in BOTH
# subproblems. Handing them to the interface leaves them unconstrained on the
# Neumann side: that subproblem is still well posed, still converges, and lands
# a few percent off — measured, 4.7% in the interface displacement and 28% in
# the interface traction, on a coupling whose residual reached 1e-10 and whose
# flux balanced. They are still EXPORTED; they are just not interface-imposed.
corner = (np.abs(y_if - Y0) < 1e-10) | (np.abs(y_if - Y1) < 1e-10)
iface_bc_n = iface_n[~corner]


def eps(w):
    return ufl.sym(ufl.grad(w))


def sigma(w):
    return 2.0 * MU * eps(w) + LAM * ufl.tr(eps(w)) * ufl.Identity(2)


u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(sigma(u), eps(v)) * ufl.dx
L = ufl.inner(fem.Constant(domain, np.zeros(2, dtype=default_scalar_type)),
              v) * ufl.dx

# ── Dirichlet on the WHOLE non-interface boundary ─────────────────────────
g_out = fem.Function(V)
g_out.x.array[:] = 0.0
outer_n = np.where((np.abs(xy[:, 0] - OUTER_X) < 1e-10) |
                   (np.abs(xy[:, 1] - Y0) < 1e-10) |
                   (np.abs(xy[:, 1] - Y1) < 1e-10))[0]
ox, oy = xy[outer_n, 0], xy[outer_n, 1]
g_out.x.array[2 * outer_n] = (UDX[0] + UDX[1] * ox + UDX[2] * oy
                              + UDX[3] * oy * oy)
g_out.x.array[2 * outer_n + 1] = (UDY[0] + UDY[1] * ox + UDY[2] * oy
                                  + UDY[3] * oy * oy)
bcs = [fem.dirichletbc(g_out, outer_n.astype(np.int32))]

if SIDE == "dirichlet":
    g = fem.Function(V)
    g.x.array[:] = 0.0
    u_if = sample(imp, "values", (UI_X, UI_Y), y_if)
    g.x.array[2 * iface_n] = u_if[:, 0]
    g.x.array[2 * iface_n + 1] = u_if[:, 1]
    bcs.append(fem.dirichletbc(g, iface_bc_n.astype(np.int32)))
else:
    facets = dmesh.locate_entities_boundary(
        domain, fdim, lambda x: np.isclose(x[0], IFACE_X))
    tags = dmesh.meshtags(domain, fdim, np.sort(facets),
                          np.full(len(facets), 7, dtype=np.int32))
    ds_if = ufl.Measure("ds", domain=domain, subdomain_data=tags)(7)
    g = fem.Function(V)
    g.x.array[:] = 0.0
    t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), y_if)
    g.x.array[2 * iface_n] = t_if[:, 0]
    g.x.array[2 * iface_n + 1] = t_if[:, 1]
    L += ufl.inner(g, v) * ds_if     # APPLY the partner's numbers UNCHANGED

uh = LinearProblem(a, L, bcs=bcs, petsc_options_prefix="cpl",
                   petsc_options={"ksp_type": "preonly",
                                  "pc_type": "lu"}).solve()

# q_out = -(sigma . n_own), L2-projected onto the same CG1 vector space so its
# values live on the same nodes as the interface DOFs.
p_, w_ = ufl.TrialFunction(V), ufl.TestFunction(V)
n_own = ufl.as_vector([default_scalar_type(S), default_scalar_type(0.0)])
qh = LinearProblem(ufl.inner(p_, w_) * ufl.dx,
                   ufl.inner(-ufl.dot(sigma(uh), n_own), w_) * ufl.dx,
                   petsc_options_prefix="trc",
                   petsc_options={"ksp_type": "preonly",
                                  "pc_type": "lu"}).solve()

U = np.column_stack([uh.x.array[2 * iface_n], uh.x.array[2 * iface_n + 1]])
Q = np.column_stack([qh.x.array[2 * iface_n], qh.x.array[2 * iface_n + 1]])
print(f"[fenics {SIDE}] interface n={len(U)} "
      f"ux=[{U[:,0].min():.6g},{U[:,0].max():.6g}] "
      f"uy=[{U[:,1].min():.6g},{U[:,1].max():.6g}] "
      f"tx=[{Q[:,0].min():.6g},{Q[:,0].max():.6g}] "
      f"ty=[{Q[:,1].min():.6g},{Q[:,1].max():.6g}]")

# exports.json LAST: the driver takes its existence as proof of success.
Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": int(len(iface_n)),
    "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
    "values": [[float(a_), float(b_)] for a_, b_ in U],
    "normal_fluxes": [[float(a_), float(b_)] for a_, b_ in Q],
}, indent=2))
