"""Tier-2 for fenics nonlinear_pde#7: the PETSc line search changes WHICH failure
you get, not whether you get one, and every one of the failed runs hands back a
finite, innocent-looking field.

Wrong variant: shopping for a line search to make Bratu converge past its turning
point (lambda = 20 on the unit square, 30-iteration budget) and then trusting
whichever run "finished".

Observed on dolfinx 0.10.0 / PETSc 3.24, same problem, only
snes_linesearch_type changed:
  basic -> reason -5 (DIVERGED_MAX_IT) after the full 30 iterations, max(u) of
           order 1e-4
  bt    -> reason -6 (DIVERGED_LINE_SEARCH) after 23 iterations, max(u) exactly
           0.0
  l2    -> reason -9 (DIVERGED_DTOL) after 2 iterations, max(u) of order 1e+01
  cp    -> same as basic
All four fields are finite and free of NaNs, and they disagree with each other by
orders of magnitude, so "it ran and the numbers are finite" is no evidence at
all.

Mutation control: T2_MUTATE=1 drops the load below the fold (lambda = 1), where
all four line searches converge to the same field and the divergence spread
disappears.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N = 16
LMBDA = 1.0 if MUTATE else 20.0
MAX_IT = 30
SEARCHES = ("basic", "bt", "l2", "cp")


def reason_name(r: int) -> str:
    for nm in dir(PETSc.SNES.ConvergedReason):
        if nm.startswith(("CONVERGED", "DIVERGED")) and \
                int(getattr(PETSc.SNES.ConvergedReason, nm)) == r:
            return nm
    return "<absent>"


def run(ls: str):
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
    problem = dolfinx.fem.petsc.NonlinearProblem(
        F, u, bcs=[bc], petsc_options_prefix=f"t2_np7_{ls}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_max_it": MAX_IT, "snes_linesearch_type": ls})
    problem.solve()
    arr = u.x.array.copy()
    return (problem.solver.getConvergedReason(),
            problem.solver.getIterationNumber(), arr)


def main() -> int:
    print(f"lambda={LMBDA:g} snes_max_it={MAX_IT}")
    reasons, its, maxima, finite_all = {}, {}, {}, True
    for ls in SEARCHES:
        r, it, arr = run(ls)
        reasons[ls], its[ls] = r, it
        maxima[ls] = float(arr.max())
        finite_all = finite_all and bool(np.all(np.isfinite(arr)))
        print(f"linesearch={ls} reason={r} name={reason_name(r)} "
              f"iterations={it} max_u={maxima[ls]:.6e} "
              f"any_nan={bool(np.isnan(arr).any())}")

    distinct = len(set(reasons.values()))
    none_converged = all(r < 0 for r in reasons.values())
    print(f"reasons={reasons}")
    print(f"distinct_reasons={distinct}")
    print(f"at_least_three_distinct_reasons={distinct >= 3}")
    print(f"no_line_search_converged={none_converged}")
    print(f"every_field_is_finite_and_nan_free={finite_all}")

    hi = max(abs(v) for v in maxima.values())
    lo = min(abs(v) for v in maxima.values())
    spread = hi > 1.0e3 * max(lo, 1.0e-300)
    print(f"max_u_hi={hi:.3e} max_u_lo={lo:.3e}")
    print(f"fields_disagree_by_orders_of_magnitude={spread}")

    if distinct >= 3 and none_converged and finite_all and spread:
        print("VERDICT=line_search_changes_which_failure_not_whether")
        return 0
    print("VERDICT=the_line_search_decided_convergence")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
