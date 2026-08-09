"""Tier-2 for fenics navier_stokes#2: problem.solve() does not raise when the
nonlinear solve fails. It returns a Function whatever happened.

The fixture drives a nonlinear problem into failure with a one-iteration budget,
then shows that the only way to find out is
problem.solver.getConvergedReason().

Mutation control: T2_MUTATE=1 gives the solver a real iteration budget, so the
reason turns positive.
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
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    # A stiff nonlinearity, started far from the solution.
    u.x.array[:] = 5.0
    F = (ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + ufl.exp(10.0 * u) * v * ufl.dx
         - dolfinx.fem.Constant(msh, 1.0) * v * ufl.dx)
    opts = {"snes_type": "newtonls", "ksp_type": "preonly", "pc_type": "lu",
            "snes_max_it": 200 if MUTATE else 1,
            "snes_linesearch_type": "basic"}
    problem = dolfinx.fem.petsc.NonlinearProblem(
        F, u, bcs=[], petsc_options_prefix="t2_ns2_", petsc_options=opts)

    raised = ""
    try:
        out = problem.solve()
        print("solve_raised=False")
    except Exception as exc:
        raised = f"{type(exc).__name__}: {exc}"
        print(f"solve_raised=True {raised}")
        out = None
    reason = problem.solver.getConvergedReason()
    print(f"snes_max_it={opts['snes_max_it']}")
    print(f"converged_reason={reason}")
    print(f"reason_is_negative_or_zero={reason <= 0}")
    finite = out is not None and bool(np.all(np.isfinite(u.x.array)))
    print(f"returned_a_finite_field={finite}")
    if not raised and reason <= 0 and finite:
        print("VERDICT=failed_solve_returns_quietly_check_the_reason")
        return 0
    print("VERDICT=failure_was_announced")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
