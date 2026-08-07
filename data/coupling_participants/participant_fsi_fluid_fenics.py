"""FEniCSx (dolfinx) FLUID participant for the OASiS `couple` driver — FSI.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

Physics: incompressible Navier-Stokes (Taylor-Hood P2/P1) on a rectangular
channel whose ONE moving boundary is the FSI interface.  The interface
displacement handed in by the structure participant is extended into the fluid
domain by a harmonic (Laplace) ALE lift and the mesh is moved by it before the
flow is solved.  What is exported back is the traction the fluid exerts ON THE
STRUCTURE.

SIGN CONVENTION — the single most common FSI wiring error, so it is written
down here and nowhere else is allowed to re-derive it:

    exported traction  t = sigma_f . n_s        with n_s = -n_f

    n_f  outward unit normal of the FLUID domain on the interface
    n_s  outward unit normal of the STRUCTURE on the same surface = -n_f

  Cauchy's t(n) = sigma.n is the traction exerted BY the material n points INTO,
  ON the material n points OUT OF.  So sigma_f . n_f is what the STRUCTURE does
  to the FLUID; the load on the structure is its negative.  Check: static fluid
  at pressure p>0 under a wall gives sigma_f = -p I, n_f = +e_y, and the export
  t = -sigma_f . n_f = +p e_y  — the fluid pushes the wall AWAY from itself.

  The structure participant applies the imported t DIRECTLY as a Neumann load
  (no further sign change).  Flipping this sign is a physics mutation: it turns
  a bulging wall into a collapsing one.

The traction is not evaluated pointwise (sigma.n of a Taylor-Hood solution is
discontinuous and inaccurate at nodes).  It is the VARIATIONALLY CONSISTENT
traction: the L2(Gamma) projection of sigma.n onto the interface trace of the
continuous P1 vector space, i.e. the consistent nodal forces divided through by
the interface mass matrix.  That is the field whose integral over Gamma is the
exact discrete interface force, which is what the equilibrium check needs.

Interface parametrisation is LAGRANGIAN: exported/imported coordinates are the
REFERENCE (undeformed) positions of the interface nodes, on both sides.  The
interface is a material surface, so this is the stable parametrisation; using
deformed coordinates makes the exchange chase its own tail.
"""
import json
import sys
from pathlib import Path

import numpy as np
import ufl
import basix.ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER    = "solid"     # the structure participant's `name` in your couple(...) call
LX         = 1.0         # channel length
HY         = 0.2         # channel height (undeformed)
IFACE_SIDE = "top"       # which fluid boundary is the FSI interface: "top" | "bottom"
NX, NY     = 48, 10      # fluid mesh
MU         = 1.0         # dynamic viscosity
RHO_F      = 1.0         # fluid density
U_MEAN     = 1.0         # mean inflow speed (parabolic profile)
ALE_STIFF  = 1.0         # ALE Laplace stiffening exponent (1.0 = plain harmonic)
D_INIT     = 0.0         # iteration-1 fallback interface displacement (both comps)
MOVE_MESH  = True        # SET False ONLY to suppress the structure->fluid direction
                         # (the one-way control). A real FSI run keeps this True.
# ─────────────────────────────────────────────────────────────────────────

IFACE_Y = HY if IFACE_SIDE == "top" else 0.0
# outward normal of the FLUID on the interface, as a sign on e_y
NF_SIGN = 1.0 if IFACE_SIDE == "top" else -1.0


def read_imports():
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def sample_vec(imp, x_targets, fallback, ncomp=2):
    """Map the partner's samples onto THIS participant's interface points.

    The driver does no interpolation — non-matching interface meshes are handled
    here, component by component, by 1-D interpolation along the interface
    parameter (the x coordinate).  np.interp clamps outside the partner's range,
    which is what we want at a clamped end.
    """
    if not imp or not imp.get("coordinates"):
        return np.full((len(x_targets), ncomp), float(fallback))
    xs = np.asarray(imp["coordinates"], float)[:, 0]
    vals = np.asarray(imp["values"], float).reshape(len(xs), -1)
    order = np.argsort(xs)
    out = np.zeros((len(x_targets), ncomp))
    for c in range(min(ncomp, vals.shape[1])):
        out[:, c] = np.interp(x_targets, xs[order], vals[order, c])
    return out


def main():
    comm = MPI.COMM_SELF
    msh = dmesh.create_rectangle(
        comm, [np.array([0.0, 0.0]), np.array([LX, HY])], [NX, NY],
        dmesh.CellType.triangle)
    gdim = msh.geometry.dim

    # ── P1 vector space used for BOTH the ALE lift and the traction projection.
    #    Its dof ordering must coincide with the geometry node ordering for the
    #    in-place mesh move below; assert it rather than trust it.
    V1 = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
    dofc = V1.tabulate_dof_coordinates()[:, :gdim]
    if dofc.shape[0] != msh.geometry.x.shape[0] or not np.allclose(
            dofc, msh.geometry.x[:, :gdim], atol=1e-12):
        raise RuntimeError(
            "P1 dof ordering does not match the geometry node ordering; the "
            "in-place mesh move would scramble the mesh")

    # REFERENCE interface node coordinates (Lagrangian parametrisation)
    tol = 1e-9
    iface_nodes = np.where(np.abs(dofc[:, 1] - IFACE_Y) < tol)[0]
    x_iface = dofc[iface_nodes, 0]
    order = np.argsort(x_iface)
    iface_nodes = iface_nodes[order]
    x_iface = x_iface[order]
    ref_coords = np.column_stack([x_iface, np.full_like(x_iface, IFACE_Y)])

    # ── imported interface displacement ────────────────────────────────────
    imp = read_imports()
    d_iface = sample_vec(imp, x_iface, D_INIT, ncomp=2)

    # ── facet tags: 1 inflow, 2 outflow, 3 fixed wall, 4 FSI interface ─────
    def _inflow(x):
        return np.isclose(x[0], 0.0)

    def _outflow(x):
        return np.isclose(x[0], LX)

    def _iface(x):
        return np.isclose(x[1], IFACE_Y)

    def _wall(x):
        return np.isclose(x[1], HY - IFACE_Y)   # the other horizontal boundary

    fdim = msh.topology.dim - 1
    marks, facets = [], []
    for tag, fn in ((1, _inflow), (2, _outflow), (3, _wall), (4, _iface)):
        f = dmesh.locate_entities_boundary(msh, fdim, fn)
        facets.append(f)
        marks.append(np.full(len(f), tag, dtype=np.int32))
    facets = np.concatenate(facets)
    marks = np.concatenate(marks)
    srt = np.argsort(facets)
    ft = dmesh.meshtags(msh, fdim, facets[srt], marks[srt])

    # ── ALE lift: harmonic extension of the interface displacement ─────────
    d_ale = fem.Function(V1, name="ale_displacement")
    if MOVE_MESH and np.any(np.abs(d_iface) > 0):
        u_, v_ = ufl.TrialFunction(V1), ufl.TestFunction(V1)
        # Jacobian-based stiffening keeps small cells near the interface from
        # inverting; ALE_STIFF = 1.0 is the plain harmonic extension.
        a = ALE_STIFF * ufl.inner(ufl.grad(u_), ufl.grad(v_)) * ufl.dx
        L = ufl.inner(fem.Constant(msh, np.zeros(gdim)), v_) * ufl.dx
        g_iface = fem.Function(V1)
        g_iface.x.array.reshape(-1, gdim)[iface_nodes] = d_iface
        zero = fem.Function(V1)
        bcs = [fem.dirichletbc(g_iface, fem.locate_dofs_topological(
                   V1, fdim, ft.find(4))),
               fem.dirichletbc(zero, fem.locate_dofs_topological(
                   V1, fdim, np.concatenate([ft.find(1), ft.find(2), ft.find(3)])))]
        pr = LinearProblem(a, L, bcs=bcs, u=d_ale,
                           petsc_options_prefix="ale_",
                           petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        pr.solve()
        msh.geometry.x[:, :gdim] += d_ale.x.array.reshape(-1, gdim)

    # ── Navier-Stokes, Taylor-Hood P2/P1, monolithic Newton ────────────────
    Ve = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(gdim,))
    Qe = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = fem.functionspace(msh, basix.ufl.mixed_element([Ve, Qe]))
    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    n = ufl.FacetNormal(msh)

    F = (RHO_F * ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
         + MU * ufl.inner(ufl.grad(u) + ufl.grad(u).T, ufl.grad(v)) * ufl.dx
         - ufl.inner(p, ufl.div(v)) * ufl.dx
         + ufl.inner(ufl.div(u), q) * ufl.dx)

    W0 = W.sub(0)
    Vsub, _ = W0.collapse()

    def _parabolic(x):
        vals = np.zeros((gdim, x.shape[1]))
        vals[0] = 6.0 * U_MEAN * x[1] * (HY - x[1]) / HY**2
        return vals

    u_in = fem.Function(Vsub)
    u_in.interpolate(_parabolic)
    u_zero = fem.Function(Vsub)
    bcs = [
        fem.dirichletbc(u_in, fem.locate_dofs_topological(
            (W0, Vsub), fdim, ft.find(1)), W0),
        fem.dirichletbc(u_zero, fem.locate_dofs_topological(
            (W0, Vsub), fdim, ft.find(3)), W0),
        # steady state: the interface is a stationary no-slip wall in its
        # DEFORMED position (the mesh already carries the displacement)
        fem.dirichletbc(u_zero, fem.locate_dofs_topological(
            (W0, Vsub), fdim, ft.find(4)), W0),
    ]

    problem = NonlinearProblem(
        F, w, bcs=bcs, petsc_options_prefix="ns_",
        petsc_options={"snes_type": "newtonls", "snes_rtol": 1e-11,
                       "snes_atol": 1e-12, "snes_max_it": 40,
                       "ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps",
                       "snes_error_if_not_converged": True,
                       "ksp_error_if_not_converged": True})
    problem.solve()
    reason = problem.solver.getConvergedReason()
    nit = problem.solver.getIterationNumber()
    if reason <= 0:
        raise RuntimeError(f"fluid Newton did not converge (reason={reason}, "
                           f"{nit} iterations)")

    # ── variationally consistent traction on the interface ─────────────────
    #   t = sigma_f . n_s = -sigma_f . n_f    (see the module docstring)
    uh, ph = w.sub(0), w.sub(1)
    sigma = -ph * ufl.Identity(gdim) + MU * (ufl.grad(uh) + ufl.grad(uh).T)
    tt, vv = ufl.TrialFunction(V1), ufl.TestFunction(V1)
    a_m = ufl.inner(tt, vv) * ds(4)
    L_t = ufl.inner(-ufl.dot(sigma, n), vv) * ds(4)

    from dolfinx.fem.petsc import assemble_matrix, assemble_vector
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    A = assemble_matrix(fem.form(a_m))
    A.assemble()
    ai, aj, av = A.getValuesCSR()
    M = sp.csr_matrix((av, aj, ai), shape=A.getSize())
    b = assemble_vector(fem.form(L_t))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bvec = b.array.copy()
    # the mass matrix is singular off the interface: put 1 on those diagonals
    diag = M.diagonal().copy()
    dead = np.where(np.abs(diag) < 1e-14)[0]
    Mfix = M.tolil()
    for i in dead:
        Mfix[i, i] = 1.0
    bvec[dead] = 0.0
    t_all = spla.spsolve(Mfix.tocsc(), bvec).reshape(-1, gdim)
    traction = t_all[iface_nodes]

    # net interface force, for the equilibrium check on the other side
    fx = fem.assemble_scalar(fem.form(-ufl.dot(sigma, n)[0] * ds(4)))
    fy = fem.assemble_scalar(fem.form(-ufl.dot(sigma, n)[1] * ds(4)))

    out = {
        "field_name": "traction_on_structure",
        "n_points": int(len(x_iface)),
        "coordinates": ref_coords.tolist(),
        "values": traction.tolist(),
        "meta": {
            "sign_convention": "sigma_f . n_s (load ON the structure)",
            "net_force": [float(fx), float(fy)],
            "mesh_moved": bool(MOVE_MESH),
            "newton_iterations": int(nit),
            "p_inlet_mean": float(
                fem.assemble_scalar(fem.form(ph * ds(1)))
                / fem.assemble_scalar(fem.form(fem.Constant(msh, 1.0) * ds(1)))),
        },
    }
    Path("exports.json").write_text(json.dumps(out, indent=2))
    print(f"[fluid] newton={nit} net_force=({fx:.6e},{fy:.6e}) "
          f"|d_iface|max={np.max(np.abs(d_iface)):.4e}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
