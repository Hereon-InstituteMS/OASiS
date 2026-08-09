"""Tier-2 for fenics stokes_darcy#4: facet and cell predicates for the region
touching the porous interface must use tolerant inequalities. dolfinx marks an
entity only when ALL of its vertices satisfy the predicate, so splitting an outlet
with x[1] > 0.5 and x[1] < 0.5 leaves the two facets that touch y = 0.5 in neither
set.

Wrong variant: strict `x[1] > 0.5` and `x[1] < 0.5`. Right variant:
`x[1] >= 0.5 - 1e-12` and `x[1] <= 0.5 + 1e-12`.

A Brinkman channel on a 16x16 unit square (porous lower quarter, K = 1e-4,
no-slip top and bottom, unit inlet pressure at x = 0) has its outlet at x = 1 split
into an upper and a lower tag, and the fixture integrates 1*ds over both tags and
the flux through both tags.

Observed on dolfinx 0.10.0: with the strict pair nothing is raised and no warning
is emitted, but the two tags hold 7 facets each instead of 8, the two tagged pieces
of a boundary of length 1.0 measure 0.437500 + 0.437500 = 0.875000, and the flux
balance over inlet plus the two tagged outlet pieces comes out at -21.96% of the
inlet flux instead of machine zero -- the flux through the two dropped facets is
simply never counted. With the tolerant pair the two tags hold 8 facets each, they
do NOT overlap (each seam facet has only one vertex exactly at y = 0.5, so it
satisfies only one of the two tolerant predicates), the measures sum to 1.000000
and the same balance reads 5.6e-15. The claim's "a few percent" is configuration
dependent: splitting where the channel runs fastest costs 22% here.

Mutation control: T2_MUTATE=1 uses the tolerant predicates in the slot under test,
so the dropped-facet and imbalance tokens go False.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import basix.ufl  # noqa: E402
import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dolfinx import fem, mesh  # noqa: E402
from petsc4py import PETSc  # noqa: E402

MU, KPERM, PIN, N = 1.0, 1e-4, 1.0, 16
TOL = 1e-12
Y_SPLIT = 0.5


def run(tolerant: bool) -> dict:
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    porous = mesh.locate_entities(msh, tdim, lambda x: x[1] <= 0.25 + TOL)
    marks = np.full(msh.topology.index_map(tdim).size_local, 1, dtype=np.int32)
    marks[porous] = 2
    ct = mesh.meshtags(msh, tdim, np.arange(len(marks), dtype=np.int32), marks)
    inlet = mesh.locate_entities_boundary(msh, fdim,
                                         lambda x: np.isclose(x[0], 0.0))
    wall = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
    whole_outlet = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 1.0))
    if tolerant:
        up = mesh.locate_entities_boundary(
            msh, fdim,
            lambda x: np.isclose(x[0], 1.0) & (x[1] >= Y_SPLIT - TOL))
        lo = mesh.locate_entities_boundary(
            msh, fdim,
            lambda x: np.isclose(x[0], 1.0) & (x[1] <= Y_SPLIT + TOL))
    else:
        up = mesh.locate_entities_boundary(
            msh, fdim, lambda x: np.isclose(x[0], 1.0) & (x[1] > Y_SPLIT))
        lo = mesh.locate_entities_boundary(
            msh, fdim, lambda x: np.isclose(x[0], 1.0) & (x[1] < Y_SPLIT))
    overlap = int(np.intersect1d(up, lo).size)
    dropped = int(np.setdiff1d(whole_outlet, np.union1d(up, lo)).size)

    fi = np.concatenate([inlet, up, lo])
    fv = np.concatenate([np.full(len(inlet), 1, np.int32),
                         np.full(len(up), 2, np.int32),
                         np.full(len(lo), 3, np.int32)])
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
    L = -PIN * ufl.dot(v, nv) * ds(1)
    V0, _ = W.sub(0).collapse()
    zero = fem.Function(V0)
    bcs = [fem.dirichletbc(
        zero, fem.locate_dofs_topological((W.sub(0), V0), fdim, wall), W.sub(0))]
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
    raised, warned = "", 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            ksp.solve(b, wh.x.petsc_vec)
            wh.x.scatter_forward()
        except Exception as exc:  # noqa: BLE001
            raised = f"{type(exc).__name__}: {exc}"
        warned = len(caught)
    uh = wh.sub(0).collapse()
    one = fem.Constant(msh, dolfinx.default_scalar_type(1.0))
    l_up = float(fem.assemble_scalar(fem.form(one * ds(2))))
    l_lo = float(fem.assemble_scalar(fem.form(one * ds(3))))
    f_in = float(fem.assemble_scalar(fem.form(ufl.dot(uh, nv) * ds(1))))
    f_up = float(fem.assemble_scalar(fem.form(ufl.dot(uh, nv) * ds(2))))
    f_lo = float(fem.assemble_scalar(fem.form(ufl.dot(uh, nv) * ds(3))))
    return dict(n_up=len(up), n_lo=len(lo), n_outlet=len(whole_outlet),
                overlap=overlap, dropped=dropped, l_up=l_up, l_lo=l_lo,
                total_len=l_up + l_lo, reason=int(ksp.getConvergedReason()),
                raised=raised, warned=warned,
                rel_balance=(f_in + f_up + f_lo) / abs(f_in))


def show(tag: str, r: dict) -> None:
    print(f"{tag}: facets_upper={r['n_up']} facets_lower={r['n_lo']} "
          f"facets_on_the_whole_outlet={r['n_outlet']} "
          f"facets_in_neither_set={r['dropped']} sets_overlap_by={r['overlap']}")
    print(f"{tag}: tagged_length_upper={r['l_up']:.6f} "
          f"tagged_length_lower={r['l_lo']:.6f} "
          f"tagged_length_total={r['total_len']:.6f} "
          f"true_boundary_length=1.000000")
    print(f"{tag}: ksp_reason={r['reason']} raised={r['raised'][:40]!r} "
          f"python_warnings={r['warned']} "
          f"relative_flux_balance={r['rel_balance']:.4e}")


def main() -> int:
    tested = run(tolerant=MUTATE)
    show("under_test", tested)
    if MUTATE:
        print("mutation=slot_under_test_uses_tolerant_inequalities")
    ref = run(tolerant=True)
    show("tolerant  ", ref)

    silent = tested["raised"] == "" and tested["warned"] == 0 \
        and tested["reason"] == 4
    dropped_two = tested["dropped"] == 2
    short = abs(tested["total_len"] - 1.0) > 1e-9
    imbalanced = abs(tested["rel_balance"]) > 0.01
    covered = abs(ref["total_len"] - 1.0) < 1e-12 and ref["dropped"] == 0
    disjoint = ref["overlap"] == 0
    ref_balanced = abs(ref["rel_balance"]) < 1e-12
    print(f"strict_split_raised_nothing_and_warned_nothing={silent}")
    print(f"strict_split_left_two_facets_in_neither_set={dropped_two}")
    print(f"strict_tagged_length_is_short_of_the_true_boundary={short}")
    print(f"strict_flux_balance_is_off_by_more_than_one_percent={imbalanced}")
    print(f"tolerant_split_covers_the_whole_boundary={covered}")
    print(f"tolerant_predicate_sets_do_not_overlap={disjoint}")
    print(f"tolerant_flux_balance_is_machine_zero={ref_balanced}")
    if silent and dropped_two and short and imbalanced and covered \
            and disjoint and ref_balanced:
        print("VERDICT=strict_inequalities_drop_the_facets_that_touch_the_seam")
        return 0
    print("VERDICT=strict_inequalities_tagged_the_whole_boundary")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
