"""Tier-2 for fenics dg_methods#6: the inflow value of a DG problem must be
imposed WEAKLY through the boundary integral. A strong fem.dirichletbc on a DG
space is a silent no-op.

Wrong variant: locate the inflow facets, call fem.locate_dofs_topological on
the DG1 space, build a fem.dirichletbc with value 1 and hand it to the
assembler. Nothing complains at any stage.

Observed on an 8x8 unit square with the x = 0 facets: locate_dofs_topological
returns ZERO dofs on the DG1 space (the same call on a Lagrange1 space returns
9), fem.dirichletbc accepts that empty index array without raising, and the
assembled system is completely unaware of it — the solution is identically zero
whether the boundary value is 0 or 1. Imposing the same value weakly, through
-(b.n - |b.n|)/2 * u_D * v * ds, changes the answer to max|u| = 1.

Mutation control: T2_MUTATE=1 imposes the inflow value weakly in the slot where
the strong condition was, so the boundary value does reach the solution.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

from dolfinx import fem, mesh  # noqa: E402
import dolfinx.fem.petsc as dfp  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

FCO = {"quadrature_degree": 4}
N = 8


def main() -> int:
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    inflow = mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))

    V = fem.functionspace(msh, ("DG", 1))
    W = fem.functionspace(msh, ("Lagrange", 1))
    dg_dofs = fem.locate_dofs_topological(V, fdim, inflow)
    cg_dofs = fem.locate_dofs_topological(W, fdim, inflow)
    print(f"inflow_facets={len(inflow)} dg1_dofs_found={len(dg_dofs)} "
          f"lagrange1_dofs_found={len(cg_dofs)}")
    print(f"dg1_locate_dofs_returns_empty={len(dg_dofs) == 0}")
    print(f"lagrange1_locate_dofs_returns_dofs={len(cg_dofs) > 0}")

    u_D = fem.Constant(msh, 1.0)
    bc_raised = ""
    try:
        bc = fem.dirichletbc(u_D, dg_dofs, V)
        n_bc = int(bc.dof_indices()[0].size)
    except Exception as exc:                            # pragma: no cover
        bc_raised = f"{type(exc).__name__}: {exc}"
        bc, n_bc = None, -1
    print(f"dirichletbc_on_dg_constructed_without_error={bc_raised == ''} "
          f"constrained_dofs={n_bc}")

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    b = ufl.as_vector([1.0, 0.5])
    n = ufl.FacetNormal(msh)
    bn = ufl.dot(b, n)
    up = ((bn("+") + abs(bn("+"))) / 2.0 * u("+")
          + (bn("+") - abs(bn("+"))) / 2.0 * u("-"))
    bn_out = (bn + abs(bn)) / 2.0
    bn_in = (bn - abs(bn)) / 2.0
    weak = fem.Constant(msh, 0.0)
    a = (-ufl.inner(u * b, ufl.grad(v)) * ufl.dx
         + up * ufl.jump(v) * ufl.dS
         + bn_out * u * v * ufl.ds)
    L = -weak * bn_in * u_D * v * ufl.ds
    a_form = fem.form(a, form_compiler_options=FCO)
    L_form = fem.form(L, form_compiler_options=FCO)

    def solve(weakly: bool, value: float):
        weak.value = 1.0 if weakly else 0.0
        u_D.value = value
        bcs = [] if weakly else [bc]
        A = dfp.assemble_matrix(a_form, bcs=bcs)
        A.assemble()
        rhs = dfp.assemble_vector(L_form)
        dfp.apply_lifting(rhs, [a_form], bcs=[bcs])
        rhs.ghostUpdate(addv=PETSc.InsertMode.ADD,
                        mode=PETSc.ScatterMode.REVERSE)
        dfp.set_bc(rhs, bcs)
        ksp = PETSc.KSP().create(msh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        uh = fem.Function(V)
        ksp.solve(rhs, uh.x.petsc_vec)
        uh.x.scatter_forward()
        return ksp.getConvergedReason(), uh.x.array.copy()

    r0, s0 = solve(MUTATE, 0.0)
    r1, s1 = solve(MUTATE, 1.0)
    same = bool(np.array_equal(s0, s1))
    print(f"slot_reasons={r0}/{r1} slot_max_abs_u_value0={np.abs(s0).max():.6e} "
          f"slot_max_abs_u_value1={np.abs(s1).max():.6e}")
    print(f"slot_boundary_value_has_no_effect={same}")

    rw0, w0 = solve(True, 0.0)
    rw1, w1 = solve(True, 1.0)
    weak_imposes = bool(not np.array_equal(w0, w1)
                        and abs(np.abs(w1).max() - 1.0) < 1.0e-9)
    print(f"weak_reference_max_abs_u_value1={np.abs(w1).max():.6e}")
    print(f"weak_imposition_does_impose_the_value={weak_imposes}")

    if (len(dg_dofs) == 0 and len(cg_dofs) > 0 and bc_raised == ""
            and same and float(np.abs(s1).max()) == 0.0 and weak_imposes
            and r0 > 0 and r1 > 0):
        print("VERDICT=strong_dirichletbc_on_dg_is_a_silent_noop")
        return 0
    print("VERDICT=strong_dirichletbc_on_dg_did_something")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
