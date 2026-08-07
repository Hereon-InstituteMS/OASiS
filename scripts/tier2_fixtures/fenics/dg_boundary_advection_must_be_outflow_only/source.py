"""Tier-2 for fenics dg_methods#0: the boundary advection term must be
restricted to the OUTFLOW part of the boundary, (b.n + |b.n|)/2. Writing
dot(b, n)*u*v*ds over the whole boundary subtracts on the inflow facets,
destroys coercivity and makes the assembled operator numerically singular.

Scope, exactly as the claim states it: the outflow restriction is the
STRUCTURAL fix, so the fixture uses the case where nothing else can hide it —
pure advection (eps = 0, no diffusion, no ds Nitsche block) on the unit square
with b = (1, 0.5). The two variants differ only in two fem.Constant
coefficients in front of b.n and |b.n| on ds, so the compiled form, the mesh
and the right-hand side are bit-for-bit the same and the boundary term is the
only difference.

Observed with the raw dot(b, n): the smallest singular value of the assembled
matrix is ~3e-19 against a largest of ~1e-1, PETSc's LU aborts on a zero pivot
("Zero pivot in LU factorization"), KSPConvergedReason is -11
(KSP_DIVERGED_PC_FAILED), ksp.solve() raises nothing, and the unchecked print
of the solution reads "u: min=inf, max=inf". The same matrix with the outflow
restriction has a condition number of order 1e1-1e2 and solves cleanly.

Mutation control: T2_MUTATE=1 puts the outflow restriction in the slot where
the raw dot(b, n) was, so the operator is invertible and every pathological
expectation disappears.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

from dolfinx import fem, mesh  # noqa: E402
from dolfinx.fem.petsc import assemble_matrix, assemble_vector  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N = 8
RAW = (1.0, 0.0)        # dot(b, n) over the whole boundary  -> the pitfall
OUTFLOW = (0.5, 0.5)    # (b.n + |b.n|)/2                    -> the fix


def main() -> int:
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    V = fem.functionspace(msh, ("DG", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

    b = ufl.as_vector([1.0, 0.5])
    n = ufl.FacetNormal(msh)
    bn = ufl.dot(b, n)
    upwind = ((bn("+") + abs(bn("+"))) / 2.0 * u("+")
              + (bn("+") - abs(bn("+"))) / 2.0 * u("-"))
    c_bn = fem.Constant(msh, 1.0)
    c_abs = fem.Constant(msh, 0.0)
    a = (-ufl.inner(u * b, ufl.grad(v)) * ufl.dx
         + upwind * ufl.jump(v) * ufl.dS
         + (c_bn * bn + c_abs * abs(bn)) * u * v * ufl.ds)
    f = fem.Constant(msh, 1.0)
    a_form = fem.form(a)
    L_form = fem.form(f * v * ufl.dx)
    rhs = assemble_vector(L_form)
    rhs.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    def assemble(coeffs):
        c_bn.value, c_abs.value = coeffs
        A = assemble_matrix(a_form)
        A.assemble()
        return A

    def singular_values(A):
        ai, aj, av = A.getValuesCSR()
        dense = sp.csr_matrix((av, aj, ai), shape=A.getSize()).toarray()
        s = sla.svdvals(dense)
        return float(s.min()), float(s.max())

    def lu_solve(A):
        ksp = PETSc.KSP().create(msh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        uh = fem.Function(V)
        raised = ""
        try:
            ksp.solve(rhs, uh.x.petsc_vec)
        except Exception as exc:                       # pragma: no cover
            raised = f"{type(exc).__name__}: {exc}"
        uh.x.scatter_forward()
        return ksp.getConvergedReason(), uh.x.array.copy(), raised

    # ---- the slot under test -------------------------------------------
    A_bad = assemble(OUTFLOW if MUTATE else RAW)
    smin, smax = singular_values(A_bad)
    reason, arr, raised = lu_solve(A_bad)
    singular = smin / smax < 1.0e-14
    all_bad = bool(np.all(~np.isfinite(arr)))
    print(f"bad_slot_smin_over_smax={smin / smax:.4e} cond={smax / smin:.4e}")
    print(f"bad_slot_matrix_numerically_singular={singular}")
    print(f"bad_slot_ksp_converged_reason={reason} "
          f"is_diverged_pc_failed={reason == -11}")
    print(f"bad_slot_solve_raised_nothing={raised == ''}")
    print(f"bad_slot_solution_all_nonfinite={all_bad}")
    # exactly the line an unchecked script prints
    print(f"u: min={arr.min():.6e}, max={arr.max():.6e}")

    # What PETSc says when it is asked to complain at all.
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A_bad)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.setErrorIfNotConverged(True)
    uh2 = fem.Function(V)
    try:
        ksp.solve(rhs, uh2.x.petsc_vec)
        print("petsc_error_if_not_converged=no_exception")
    except Exception as exc:
        print(f"petsc_error_if_not_converged={type(exc).__name__}\n{exc}")

    # ---- the outflow-restricted reference, always run -------------------
    A_ref = assemble(OUTFLOW)
    rmin, rmax = singular_values(A_ref)
    r_reason, r_arr, _ = lu_solve(A_ref)
    ref_cond = rmax / rmin
    print(f"outflow_reference_cond={ref_cond:.4e}")
    print(f"outflow_reference_cond_below_1e4={ref_cond < 1.0e4}")
    print(f"outflow_reference_reason_is_converged={r_reason > 0}")
    print(f"outflow_reference_solution_finite="
          f"{bool(np.all(np.isfinite(r_arr)))}")

    if (singular and reason == -11 and all_bad and raised == ""
            and ref_cond < 1.0e4 and r_reason > 0):
        print("VERDICT=raw_dot_b_n_on_ds_destroys_coercivity")
        return 0
    print("VERDICT=raw_dot_b_n_on_ds_is_harmless")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
