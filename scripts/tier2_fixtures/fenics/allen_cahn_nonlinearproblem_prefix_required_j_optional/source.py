"""Tier-2 for fenics multiphase#9: `dolfinx.fem.petsc.NonlinearProblem` takes
`petsc_options_prefix` as a REQUIRED keyword-only argument and `J` as an
OPTIONAL one, and the deprecated `dolfinx.nls.petsc.NewtonSolver(comm, problem)`
cannot wrap the resulting object.

Wrong variant 1: `NonlinearProblem(F, phi)` on the Allen-Cahn residual ->
`TypeError: NonlinearProblem.__init__() missing 1 required keyword-only
argument: 'petsc_options_prefix'`.
Wrong variant 2: build it correctly, then wrap it ->
`AttributeError: 'NonlinearProblem' object has no attribute 'a'`.
Right variant: pass the prefix, leave J out (UFL differentiates the residual
for you) and drive the solve through `problem.solve()` / `problem.solver`.

`sys.unraisablehook` is silenced because the half-built objects raise a second
time from their own `__del__` at garbage-collection time, which would otherwise
put an unrelated traceback on stderr.

Mutation control: T2_MUTATE=1 uses the correct API only - the prefix is passed
and the solve goes through problem.solver - so neither error text appears.
"""
from __future__ import annotations

import os
import sys
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))
sys.unraisablehook = lambda unraisable: None  # noqa: E731

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402
import dolfinx.nls.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N, EPS_OVER_H, R, DT = 16, 3.0, 0.25, 1e-3
OPTS = {"snes_type": "newtonls", "ksp_type": "preonly", "pc_type": "lu"}


def residual():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    eps = EPS_OVER_H / N
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    phi, phi_n = dolfinx.fem.Function(V), dolfinx.fem.Function(V)

    def ic(x):
        d = R - np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)
        return np.tanh(d / (eps * np.sqrt(2.0)))

    phi.interpolate(ic)
    phi_n.interpolate(ic)
    v = ufl.TestFunction(V)
    dt_c = dolfinx.fem.Constant(msh, DT)
    eps_c = dolfinx.fem.Constant(msh, eps)
    F = ((phi - phi_n) / dt_c * v * ufl.dx
         + eps_c * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
         + (1.0 / eps_c) * (phi ** 3 - phi) * v * ufl.dx)
    return F, phi, phi_n


def main() -> int:
    F, phi, phi_n = residual()

    prefix_err = "none"
    if not MUTATE:
        try:
            dolfinx.fem.petsc.NonlinearProblem(F, phi)
        except Exception as exc:  # noqa: BLE001
            prefix_err = f"{type(exc).__name__}: {exc}"
    print(f"prefix_omitted_error={prefix_err}")

    # J left out entirely: the optional keyword.
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, phi, petsc_options_prefix="t2_mp9_", petsc_options=OPTS)
    print(f"j_omitted_construction_succeeded={isinstance(prob, dolfinx.fem.petsc.NonlinearProblem)}")

    wrap_err = "none"
    if not MUTATE:
        try:
            dolfinx.nls.petsc.NewtonSolver(MPI.COMM_WORLD, prob)
        except Exception as exc:  # noqa: BLE001
            wrap_err = f"{type(exc).__name__}: {exc}"
    print(f"newtonsolver_wrap_error={wrap_err}")

    before = phi.x.array.copy()
    prob.solve()
    reason = prob.solver.getConvergedReason()
    moved = float(np.max(np.abs(phi.x.array - before)))
    print(f"snes_converged_reason={reason} field_moved={moved:.3e}")
    print(f"problem_solver_route_works={reason > 0 and moved > 0.0}")

    ok = (reason > 0 and moved > 0.0
          and prefix_err.startswith("TypeError")
          and wrap_err.startswith("AttributeError"))
    if ok:
        print("VERDICT=prefix_is_required_j_is_optional_newtonsolver_cannot_wrap")
        return 0
    print("VERDICT=api_accepted_the_wrong_calls")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
