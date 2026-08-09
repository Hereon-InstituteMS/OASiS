"""Tier-2 for fenics navier_stokes#0: dolfinx.nls.petsc.NewtonSolver cannot wrap
a 0.10 NonlinearProblem. That pairing is from <= 0.8.

The fixture builds a 0.10 NonlinearProblem, tries the legacy pairing, and prints
what came back. It then shows the supported path: problem.solve() plus
problem.solver, which is a petsc4py SNES.

Mutation control: T2_MUTATE=1 skips the legacy pairing and only exercises the
supported path.
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
import dolfinx.nls.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"



def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    w_ = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    F = (ufl.dot(ufl.grad(w_), ufl.grad(v)) * ufl.dx
         + w_ ** 2 * v * ufl.dx
         - dolfinx.fem.Constant(msh, 1.0) * v * ufl.dx)
    # A Dirichlet condition on the whole boundary: without it this residual is
    # a pure-Neumann problem, the Jacobian is singular and the SUPPORTED path
    # fails too (reason -3), which would say nothing about the legacy pairing.
    msh.topology.create_connectivity(1, 2)
    bfacets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(msh, 0.0),
        dolfinx.fem.locate_dofs_topological(V, 1, bfacets), V)
    problem = dolfinx.fem.petsc.NonlinearProblem(
        F, w_, bcs=[bc], petsc_options_prefix="t2_ns0_",
        petsc_options={"snes_type": "newtonls", "ksp_type": "preonly",
                       "pc_type": "lu"})

    legacy = ""
    if not MUTATE:
        try:
            dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
            print("legacy_pairing_raised=False")
        except Exception as exc:
            legacy = f"{type(exc).__name__}: {exc}"
            print(f"legacy_pairing_raised=True {legacy}")

    problem.solve()
    reason = problem.solver.getConvergedReason()
    kind = type(problem.solver).__name__
    print(f"supported_path_solver_type={kind}")
    print(f"supported_path_converged_reason={reason}")
    print(f"supported_path_converged={reason > 0}")
    if legacy and reason > 0:
        print("VERDICT=legacy_newtonsolver_pairing_is_broken_on_0_10")
        return 0
    print("VERDICT=legacy_pairing_worked")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
