"""scikit-fem VECTOR participant for the OASiS `couple` driver.

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
(q_out = -k dT/dn_own) and the one participant_febio.py uses for its 1-D
elastic analogue. Two consequences, both load-bearing:

  * the two sides' exports CANCEL componentwise, because n_own is anti-parallel
    across the interface — that is what makes the interface balance check a
    conservation statement rather than an accident;
  * the NEUMANN side applies the partner's numbers UNCHANGED, as
    `L += dot(g, v) ds`, because the natural boundary term of the elasticity
    weak form is +(sigma . n_own) . v = +q_out_partner . v.

Exporting the raw traction (sigma . n_own) instead flips the sign the Neumann
side applies; the iteration still converges, to the wrong answer.

RELAXATION IS NOT PER COMPONENT. The driver applies ONE theta to the whole
interface state, and the optimal theta is 1/(1+rho) with rho the ratio of the
two subdomains' interface stiffnesses. For a VECTOR interface rho is a matrix,
so u_x and u_y generally want DIFFERENT thetas and the single theta must be
chosen for the WORST component: (1-theta)^2 + rho_c*theta^2 < 1 has to hold for
every component c, so theta < 2/(1+max_c rho_c). Measured on this problem with
traction-free y-faces: rho_x ~ 0.4 while rho_y ~ 1.8, and theta = 1/(1+rho_x)
diverges on the y component while the x component converges — a
half-converging coupling that a single global residual reports only as "did
not converge". Two subdomains of the SAME length and Poisson ratio have
Steklov-Poincare operators that are proportional (S_left = (E_l/E_r) S_right),
so rho collapses to the scalar E_l/E_r and one theta is optimal for both
components; that is why the shipped placeholder geometry splits the strip in
half.
"""
import json
from pathlib import Path

import numpy as np
from skfem import (Basis, BilinearForm, ElementTriP1, ElementVector,
                   FacetBasis, LinearForm, MeshTri, asm, condense, solve)
from skfem.helpers import ddot, sym_grad, trace

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
# The two subdomains must agree at the two interface corners, or the coupled
# problem is not the un-split one.
UDX = (0.0, 0.0, 0.0, 0.0)
UDY = (0.0, 0.0, 0.0, 0.0)
NX, NY    = 24, 16        # this subdomain's own mesh (need not match the partner)
UI_X, UI_Y = 0.0, 0.0     # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # iteration-1 fallback interface traction export
# ─────────────────────────────────────────────────────────────────────────

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
    COMPONENT BY COMPONENT.

    The driver does no interpolation — non-matching interface meshes are
    handled here, and for a vector field that has to be done per component. One
    np.interp over a flattened (N, 2) array interleaves the two components: the
    result still has the right length, the coupling still converges, and every
    number is wrong.

    Returns (len(y), ncomp). `fallback` is the per-component constant used on
    iteration 1, when imports.json is `{}`.
    """
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


def u_dirichlet(x, y):
    """The prescribed displacement on the non-interface boundary."""
    return (UDX[0] + UDX[1] * x + UDX[2] * y + UDX[3] * y * y,
            UDY[0] + UDY[1] * x + UDY[2] * y + UDY[3] * y * y)


imp = read_imports()

mesh = MeshTri.init_tensor(np.linspace(X0, X1, NX + 1),
                           np.linspace(Y0, Y1, NY + 1))
elem = ElementVector(ElementTriP1())
basis = Basis(mesh, elem)
nd = basis.nodal_dofs                      # (2, nnodes): node -> (x, y) dof

px, py = mesh.p[0], mesh.p[1]
iface_n = np.where(np.abs(px - IFACE_X) < TOL)[0]
iface_n = iface_n[np.argsort(py[iface_n])]             # sorted by y
y_if = py[iface_n]
outer_n = np.where((np.abs(px - OUTER_X) < TOL) |
                   (np.abs(py - Y0) < TOL) | (np.abs(py - Y1) < TOL))[0]
# THE TWO INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES.
# (IFACE_X, Y0) and (IFACE_X, Y1) sit on a y-face, which carries a prescribed
# displacement in the un-split problem, so they are Dirichlet nodes there and
# must stay Dirichlet in BOTH subproblems. Handing them to the interface
# instead leaves them unconstrained on the Neumann side: that subproblem is
# still well posed, still converges, and lands a few percent off — measured
# here, 4.7% in the interface displacement and 28% in the interface traction,
# on a coupling whose residual reached 1e-10 and whose flux balanced. So the
# interface Dirichlet set EXCLUDES them; they are still exported, because they
# are still points of the interface.
iface_bc_n = iface_n[(np.abs(py[iface_n] - Y0) > TOL) &
                     (np.abs(py[iface_n] - Y1) > TOL)]
iface_bc_dofs = np.concatenate([nd[0, iface_bc_n], nd[1, iface_bc_n]])
outer_dofs = np.concatenate([nd[0, outer_n], nd[1, outer_n]])


@BilinearForm
def stiffness(u, v, w):
    eu, ev = sym_grad(u), sym_grad(v)
    return 2.0 * MU * ddot(eu, ev) + LAM * trace(eu) * trace(ev)


@BilinearForm
def mass(u, v, w):
    return u[0] * v[0] + u[1] * v[1]


@LinearForm
def traction(v, w):
    return w["t"][0] * v[0] + w["t"][1] * v[1]


@LinearForm
def proj_rhs(v, w):
    """L2 projection of q_out = -(sigma . n_own) onto the vector P1 space."""
    g = w["uh"].grad                       # g[i][j] = du_i/dx_j
    exx, eyy = g[0][0], g[1][1]
    exy = 0.5 * (g[0][1] + g[1][0])
    sxx = 2.0 * MU * exx + LAM * (exx + eyy)
    sxy = 2.0 * MU * exy
    return (-S) * (sxx * v[0] + sxy * v[1])


A = stiffness.assemble(basis)
b = basis.zeros()

sol = basis.zeros()
ux_d, uy_d = u_dirichlet(px[outer_n], py[outer_n])
sol[nd[0, outer_n]] = ux_d
sol[nd[1, outer_n]] = uy_d
D = outer_dofs

if SIDE == "dirichlet":
    u_if = sample(imp, "values", (UI_X, UI_Y), y_if)
    keep = (np.abs(y_if - Y0) > TOL) & (np.abs(y_if - Y1) > TOL)
    sol[nd[0, iface_bc_n]] = u_if[keep, 0]
    sol[nd[1, iface_bc_n]] = u_if[keep, 1]
    D = np.unique(np.concatenate([outer_dofs, iface_bc_dofs]))
else:
    t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), y_if)
    gnod = basis.zeros()                   # P1 trace of the partner's samples
    gnod[nd[0, iface_n]] = t_if[:, 0]
    gnod[nd[1, iface_n]] = t_if[:, 1]
    fbi = FacetBasis(mesh, elem,
                     facets=mesh.facets_satisfying(
                         lambda p: np.abs(p[0] - IFACE_X) < TOL))
    # APPLY the partner's numbers UNCHANGED (+ integral(g . v) ds_interface)
    b = b + asm(traction, fbi, t=fbi.interpolate(gnod))

sol = solve(*condense(A, b, x=sol, D=D))

qh = solve(mass.assemble(basis),
           proj_rhs.assemble(basis, uh=basis.interpolate(sol)))

Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": int(len(iface_n)),
    "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
    "values": [[float(sol[nd[0, i]]), float(sol[nd[1, i]])] for i in iface_n],
    "normal_fluxes": [[float(qh[nd[0, i]]), float(qh[nd[1, i]])] for i in iface_n],
}, indent=2))
