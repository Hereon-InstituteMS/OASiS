"""Tier-2 for fenics poisson#3: pure-Neumann Poisson does not fail loudly, it
returns a solution carrying an arbitrary constant.

Wrong variant: no Dirichlet condition anywhere, source f = 1 on the unit
square. The constant is in the kernel of the operator, so the solve reports
success and hands back a field whose spread is negligible next to its own
offset.

The fixture measures the shape of the answer instead of quoting a number: it
compares peak-to-peak spread against |mean|. A legitimate solution of this
problem has spread comparable to its mean; a null-space artefact has an offset
orders of magnitude larger than anything physical in the field.

FINDING against the claim as written. The claim says "CG with pc_type='none'
even converges without raising". It does NOT converge: the KSP reports
converged_reason = -4 (DIVERGED_DTOL). What the claim gets right, and what
actually matters, is the silence — problem.solve() raises nothing, returns a
Function, and an unchecked script would carry that field forward as a result.
So this fixture pins the harsher signal: no exception, a NEGATIVE converged
reason nobody looked at, and an offset that dwarfs the field.

Mutation control: T2_MUTATE=1 pins one dof with a Dirichlet condition, which
removes the kernel; the offset collapses and the verdict flips.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = dolfinx.fem.Constant(msh, 1.0) * v * ufl.dx

    bcs = []
    if MUTATE:
        # Pin exactly one dof: the documented cure.
        one_dof = np.array([0], dtype=np.int32)
        bcs = [dolfinx.fem.dirichletbc(
            dolfinx.fem.Constant(msh, 0.0), one_dof, V)]

    problem = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix="t2_pn_",
        petsc_options={"ksp_type": "cg", "pc_type": "none",
                       "ksp_rtol": 1e-10, "ksp_max_it": 2000})
    uh = problem.solve()
    if isinstance(uh, tuple):
        uh = uh[0]
    reason = problem.solver.getConvergedReason()
    arr = uh.x.array
    spread = float(np.max(arr) - np.min(arr))
    offset = float(abs(np.mean(arr)))
    print(f"solver_raised=False converged_reason={reason}")
    print(f"reason_is_negative={reason < 0}")
    print(f"spread={spread:.6e} offset={offset:.6e}")
    ratio = offset / spread if spread > 0 else float("inf")
    print(f"offset_over_spread={ratio:.3e}")
    print(f"offset_dominates={ratio > 1.0e3}")
    if reason < 0 and ratio > 1.0e3:
        print("VERDICT=no_raise_negative_reason_offset_dominates")
        return 0
    if reason > 0 and ratio > 1.0e3:
        print("VERDICT=converged_with_offset")
        return 0
    print("VERDICT=no_nullspace_offset")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
