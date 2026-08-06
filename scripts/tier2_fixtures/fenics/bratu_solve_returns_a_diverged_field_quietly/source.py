"""Tier-2 for fenics nonlinear_pde#1: problem.solve() hands back the solution
Function whether or not SNES converged, so the caller must assert
problem.solver.getConvergedReason() > 0 itself.

Wrong variant: solve the Bratu problem -div(grad u) = lambda*exp(u) past its
turning point (lambda = 20 on the unit square, whose fold sits near lambda = 7)
with the undamped 'basic' line search and an 8-iteration budget, then trust the
returned field because it looks finite.

Observed on dolfinx 0.10.0: solve() raises nothing, the SNES converged reason is
-5 (DIVERGED_MAX_IT), and the field that comes back is completely finite with a
maximum of order 1e-4 -- small and tidy enough to pass any "is it finite / is it
O(1)" check. Turning on "snes_error_if_not_converged" makes the identical run
raise petsc4py.PETSc.Error with the text 'Error: error code 91' /
'SNESSolve has not converged'.

Mutation control: T2_MUTATE=1 solves the SAME problem at lambda = 1, below the
fold; the reason turns positive, nothing is raised even with
snes_error_if_not_converged, and every expectation about the silent failure is
lost.
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

N = 16
LMBDA = 1.0 if MUTATE else 20.0
MAX_IT = 8


def solve_bratu(tag: str, error_if_not_converged: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), dofs, V)

    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    lam = dolfinx.fem.Constant(msh, LMBDA)
    F = (ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - lam * ufl.exp(u) * v * ufl.dx)
    opts = {"snes_type": "newtonls", "snes_linesearch_type": "basic",
            "ksp_type": "preonly", "pc_type": "lu", "snes_max_it": MAX_IT}
    if error_if_not_converged:
        opts["snes_error_if_not_converged"] = True
    problem = dolfinx.fem.petsc.NonlinearProblem(
        F, u, bcs=[bc], petsc_options_prefix=f"t2_np1_{tag}_",
        petsc_options=opts)

    raised, returned = "", None
    try:
        returned = problem.solve()
    except Exception as exc:  # petsc4py.PETSc.Error
        raised = f"{type(exc).__name__}: {exc}"
    reason = problem.solver.getConvergedReason()
    its = problem.solver.getIterationNumber()
    return raised, returned, reason, its, u.x.array.copy()


def main() -> int:
    raised, returned, reason, its, arr = solve_bratu("quiet", False)
    print(f"lambda={LMBDA:g} snes_max_it={MAX_IT}")
    print(f"solve_raised_without_the_option={bool(raised)}")
    print(f"converged_reason={reason} iterations={its}")
    print(f"reason_is_negative={reason < 0}")
    print(f"a_function_came_back={returned is not None}")
    finite = bool(np.all(np.isfinite(arr)))
    tidy = finite and float(np.abs(arr).max()) < 1.0
    print(f"field_max={float(arr.max()):.6e} "
          f"any_nan={bool(np.isnan(arr).any())}")
    print(f"returned_field_is_finite_and_tidy={tidy}")

    raised2, _, reason2, its2, _ = solve_bratu("loud", True)
    print(f"with_snes_error_if_not_converged_reason={reason2} iterations={its2}")
    print("raised_text_first_line="
          + (raised2.splitlines()[0] if raised2 else "<nothing raised>"))
    print("raised_text_last_line="
          + (raised2.splitlines()[-1] if raised2 else "<nothing raised>"))
    print(f"snes_error_if_not_converged_raises={bool(raised2)}")

    if (not raised and returned is not None and reason < 0 and tidy
            and raised2 and "not converged" in raised2):
        print("VERDICT=solve_returns_a_diverged_field_check_the_reason")
        return 0
    print("VERDICT=the_failure_announced_itself")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
