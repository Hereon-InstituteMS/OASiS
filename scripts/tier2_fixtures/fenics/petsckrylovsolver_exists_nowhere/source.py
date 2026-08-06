"""Tier-2 for fenics navier_stokes#1: PETScKrylovSolver is a legacy DOLFIN name
that exists in no dolfinx 0.10 module.

The fixture searches every plausible module for the name and reports what it
found, then shows the supported route: LinearProblem(...).solver, which IS a
petsc4py KSP.

Mutation control: T2_MUTATE=1 searches for a name that DOES exist
(LinearProblem), so the "found nowhere" observation disappears.
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

import importlib  # noqa: E402

MODULES = ["dolfinx", "dolfinx.fem", "dolfinx.fem.petsc", "dolfinx.la",
           "dolfinx.nls", "dolfinx.nls.petsc", "dolfinx.cpp"]


def main() -> int:
    name = "LinearProblem" if MUTATE else "PETScKrylovSolver"
    print(f"searched_name={name}")
    found = []
    for m in MODULES:
        try:
            mod = importlib.import_module(m)
        except Exception:
            continue
        if hasattr(mod, name):
            found.append(m)
    print(f"modules_searched={len(MODULES)}")
    print(f"found_in={found}")
    print(f"found_anywhere={bool(found)}")

    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    prob = dolfinx.fem.petsc.LinearProblem(
        ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        dolfinx.fem.Constant(msh, 1.0) * v * ufl.dx,
        bcs=[], petsc_options_prefix="t2_ns1_")
    ksp_type = type(prob.solver).__name__
    print(f"linearproblem_solver_type={ksp_type}")
    print(f"linearproblem_solver_is_ksp={ksp_type == 'KSP'}")
    if not found and ksp_type == "KSP":
        print("VERDICT=petsckrylovsolver_absent_use_problem_solver")
        return 0
    print("VERDICT=name_resolved")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
