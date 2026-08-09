"""Tier-2 for fenics helmholtz#1: the Helmholtz operator is indefinite, so CG
does not merely converge slowly — PETSc stops it with DIVERGED_INDEFINITE_MAT
(-10) after a handful of iterations, and problem.solve() does not raise.

The fixture solves k = 20 on a 32x32 square with CG under four preconditioners
and reports the converged reason of each, then shows LU converging on the same
problem.

Mutation control: T2_MUTATE=1 uses a direct LU solve, which converges.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix  # noqa: E402
import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"



def solve(ksp: str, pc: str) -> tuple[int, int]:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    k = dolfinx.fem.Constant(msh, 20.0)
    a = (ufl.dot(ufl.grad(u), ufl.grad(v)) - k ** 2 * u * v) * ufl.dx
    L = dolfinx.fem.Constant(msh, 1.0) * v * ufl.dx
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[], petsc_options_prefix="t2_hz_",
        petsc_options={"ksp_type": ksp, "pc_type": pc, "ksp_max_it": 200})
    prob.solve()
    return prob.solver.getConvergedReason(), prob.solver.getIterationNumber()


def main() -> int:
    if MUTATE:
        pairs = [("preonly", "lu")]
    else:
        pairs = [("cg", "icc"), ("cg", "jacobi"), ("cg", "none"), ("cg", "ilu")]
    reasons = []
    for ksp, pc in pairs:
        r, it = solve(ksp, pc)
        reasons.append(r)
        print(f"ksp={ksp} pc={pc} converged_reason={r} iterations={it}")
    lu_reason, _ = solve("preonly", "lu")
    print(f"lu_converged_reason={lu_reason}")
    print(f"solve_raised=False")
    # FINDING: the claim says icc, jacobi, none AND ilu all give -10
    # (DIVERGED_INDEFINITE_MAT). Measured here, ilu gives -8
    # (DIVERGED_INDEFINITE_PC) — a different code, and the claim itself names
    # -8 as "a different code" while listing ilu under -10.
    all_negative = all(r < 0 for r in reasons)
    n_minus_10 = sum(1 for r in reasons if r == -10)
    ilu_reason = reasons[-1] if len(reasons) == 4 else None
    print(f"reason_codes={reasons}")
    print(f"every_cg_variant_diverges={all_negative}")
    print(f"count_reporting_minus_10={n_minus_10}")
    print(f"ilu_reports_minus_10={ilu_reason == -10}")
    print(f"lu_converges={lu_reason > 0}")
    if all_negative and n_minus_10 >= 3 and lu_reason > 0:
        print("VERDICT=cg_diverges_on_the_indefinite_operator_lu_works")
        return 0
    print("VERDICT=cg_did_not_report_indefinite")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
