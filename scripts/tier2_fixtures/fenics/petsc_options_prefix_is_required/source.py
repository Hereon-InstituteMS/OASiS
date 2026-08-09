"""Tier-2 for fenics heat#11: LinearProblem and NonlinearProblem take
petsc_options_prefix as a REQUIRED keyword-only argument in dolfinx 0.10, and
there is no dolfinx.nls.petsc.NewtonSolver path around it.

"Traceback" is deliberately NOT in this fixture's forbid_in_output. The
TypeError leaves a half-constructed LinearProblem / NonlinearProblem whose
__del__ then raises AttributeError on `self._solver` / `self._snes`, and Python
prints that as an ignored-exception traceback on stderr. That noise is dolfinx's,
not the fixture's, and forbidding it would fail a fixture that is working.

Mutation control: T2_MUTATE=1 passes the prefix; construction succeeds.
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
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = dolfinx.fem.Constant(msh, 1.0) * v * ufl.dx

    lin_err = ""
    try:
        if MUTATE:
            dolfinx.fem.petsc.LinearProblem(
                a, L, bcs=[], petsc_options_prefix="t2_ok_")
        else:
            dolfinx.fem.petsc.LinearProblem(a, L, bcs=[])
        print("LinearProblem_without_prefix_raised=False")
    except TypeError as exc:
        lin_err = str(exc)
        print(f"LinearProblem_without_prefix_raised=True TypeError: {lin_err}")

    w = dolfinx.fem.Function(V)
    F = ufl.dot(ufl.grad(w), ufl.grad(v)) * ufl.dx - L
    non_err = ""
    try:
        if MUTATE:
            dolfinx.fem.petsc.NonlinearProblem(
                F, w, bcs=[], petsc_options_prefix="t2_ok2_")
        else:
            dolfinx.fem.petsc.NonlinearProblem(F, w, bcs=[])
        print("NonlinearProblem_without_prefix_raised=False")
    except TypeError as exc:
        non_err = str(exc)
        print(f"NonlinearProblem_without_prefix_raised=True "
              f"TypeError: {non_err}")

    if lin_err and non_err:
        print("VERDICT=petsc_options_prefix_is_required_on_both")
        return 0
    print("VERDICT=prefix_optional")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
