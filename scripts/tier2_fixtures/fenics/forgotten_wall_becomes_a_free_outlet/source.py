"""Tier-2 for fenics stokes_darcy#5: every boundary facet that carries neither a
Dirichlet condition nor a ds term silently gets the do-nothing condition
sigma.n = 0, i.e. it becomes a free outlet.

Wrong variant: a 3D box in which the two z faces are located but never given a
no-slip Dirichlet condition and never appear in the right-hand side. Right variant:
the same run with those facets in the no-slip set.

3D Brinkman box, 6x6x6 tetrahedral unit cube, porous lower half with K = 1e-4,
unit pressure imposed weakly at the inlet x = 0, zero at the outlet x = 1, no-slip
on the two y faces, MUMPS LU, 6934 dofs.

Observed on dolfinx 0.10.0: nothing is raised, getConvergedReason() returns 4 and
the fields look plausible (max |u| 1.140e-01, max |p| 1.090e+00, all finite) -- but
the inlet/outlet flux balance is off by -8.93e-01 of the inlet flux, i.e. order 1.
Integrating over the two forgotten faces recovers exactly the missing flux, so the
imbalance over inlet plus outlet plus z faces is -1.1e-16: the forgotten faces
really did become free outlets. Sealing them brings the inlet/outlet balance to
4.7e-16 and cuts the throughflow by a factor of four. The 99% quoted in the claim
is configuration dependent; the mechanism and the order of magnitude are not.

Mutation control: T2_MUTATE=1 puts the z faces into the no-slip set of the run
under test, so the imbalance token goes False.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import basix.ufl  # noqa: E402
import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dolfinx import fem, mesh  # noqa: E402
from petsc4py import PETSc  # noqa: E402

MU, KPERM, PIN, N = 1.0, 1e-4, 1.0, 6
TOL = 1e-12


def run(seal_z_faces: bool) -> dict:
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_cube(comm, N, N, N)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    porous = mesh.locate_entities(msh, tdim, lambda x: x[1] <= 0.5 + TOL)
    marks = np.full(msh.topology.index_map(tdim).size_local, 1, dtype=np.int32)
    marks[porous] = 2
    ct = mesh.meshtags(msh, tdim, np.arange(len(marks), dtype=np.int32), marks)
    inlet = mesh.locate_entities_boundary(msh, fdim,
                                         lambda x: np.isclose(x[0], 0.0))
    outlet = mesh.locate_entities_boundary(msh, fdim,
                                          lambda x: np.isclose(x[0], 1.0))
    wall_y = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    face_z = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[2], 0.0) | np.isclose(x[2], 1.0))
    fi = np.concatenate([inlet, outlet, face_z])
    fv = np.concatenate([np.full(len(inlet), 1, np.int32),
                         np.full(len(outlet), 2, np.int32),
                         np.full(len(face_z), 3, np.int32)])
    o = np.argsort(fi)
    ft = mesh.meshtags(msh, fdim, fi[o], fv[o])

    Ve = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(tdim,))
    Pe = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = fem.functionspace(msh, basix.ufl.mixed_element([Ve, Pe]))
    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    dx = ufl.Measure("dx", domain=msh, subdomain_data=ct)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    nv = ufl.FacetNormal(msh)
    a = (MU * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
         + (MU / KPERM) * ufl.inner(u, v) * dx(2)
         - p * ufl.div(v) * dx - q * ufl.div(u) * dx)
    # NOTE: the z faces appear in NO ds term of L, so if they carry no Dirichlet
    # condition either they are pure do-nothing boundaries.
    L = -PIN * ufl.dot(v, nv) * ds(1)
    V0, _ = W.sub(0).collapse()
    zero = fem.Function(V0)
    walls = np.union1d(wall_y, face_z) if seal_z_faces else wall_y
    bcs = [fem.dirichletbc(
        zero, fem.locate_dofs_topological((W.sub(0), V0), fdim, walls), W.sub(0))]
    af, Lf = fem.form(a), fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(af, bcs=bcs)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(Lf)
    dolfinx.fem.petsc.apply_lifting(b, [af], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    wh = fem.Function(W)
    raised = ""
    try:
        ksp.solve(b, wh.x.petsc_vec)
        wh.x.scatter_forward()
    except Exception as exc:  # noqa: BLE001
        raised = f"{type(exc).__name__}: {exc}"
    uh, ph = wh.sub(0).collapse(), wh.sub(1).collapse()
    f_in = float(fem.assemble_scalar(fem.form(ufl.dot(uh, nv) * ds(1))))
    f_out = float(fem.assemble_scalar(fem.form(ufl.dot(uh, nv) * ds(2))))
    f_z = float(fem.assemble_scalar(fem.form(ufl.dot(uh, nv) * ds(3))))
    return dict(dofs=W.dofmap.index_map.size_global * W.dofmap.index_map_bs,
                reason=int(ksp.getConvergedReason()), raised=raised,
                flux_in=f_in, flux_out=f_out, leak=f_z,
                inout=(f_in + f_out) / abs(f_in),
                closed=(f_in + f_out + f_z) / abs(f_in),
                umax=float(np.abs(uh.x.array).max()),
                pmax=float(np.abs(ph.x.array).max()),
                finite=bool(np.all(np.isfinite(ph.x.array))
                            and np.all(np.isfinite(uh.x.array))))


def show(tag: str, r: dict) -> None:
    print(f"{tag}: dofs={r['dofs']} ksp_reason={r['reason']} "
          f"raised={r['raised'][:40]!r} max_abs_u={r['umax']:.3e} "
          f"max_abs_p={r['pmax']:.3e} all_finite={r['finite']}")
    print(f"{tag}: flux_in={r['flux_in']:.6e} flux_out={r['flux_out']:.6e} "
          f"flux_through_the_z_faces={r['leak']:.6e}")
    print(f"{tag}: relative_inlet_outlet_balance={r['inout']:.4e} "
          f"relative_balance_including_the_z_faces={r['closed']:.3e}")


def main() -> int:
    tested = run(seal_z_faces=MUTATE)
    show("under_test", tested)
    if MUTATE:
        print("mutation=z_faces_are_in_the_no_slip_set")
    sealed = run(seal_z_faces=True)
    show("sealed    ", sealed)

    quiet = tested["raised"] == "" and tested["reason"] == 4 and tested["finite"]
    order_one = abs(tested["inout"]) > 0.5
    leak_explains = abs(tested["closed"]) < 1e-12
    sealed_ok = abs(sealed["inout"]) < 1e-12
    print(f"forgotten_faces_run_raised_nothing_and_looks_plausible={quiet}")
    print(f"inlet_outlet_balance_is_off_by_order_one={order_one}")
    print(f"the_flux_through_the_forgotten_faces_explains_the_imbalance="
          f"{leak_explains}")
    print(f"sealing_the_faces_restores_a_machine_zero_balance={sealed_ok}")
    if quiet and order_one and leak_explains and sealed_ok:
        print("VERDICT=an_unconstrained_face_is_a_free_outlet_and_only_the_flux_"
              "balance_shows_it")
        return 0
    print("VERDICT=the_unconstrained_faces_behaved_like_walls")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
