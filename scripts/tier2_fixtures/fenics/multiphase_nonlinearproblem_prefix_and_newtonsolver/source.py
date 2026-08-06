"""Tier-2 for fenics multiphase#9: `dolfinx.fem.petsc.NonlinearProblem` takes
`petsc_options_prefix` as a REQUIRED keyword-only argument and `J` as an OPTIONAL
one, and the deprecated `dolfinx.nls.petsc.NewtonSolver` cannot wrap the 0.10
NonlinearProblem.

The problem is the Allen-Cahn step this physics actually uses (32x32 unit square,
eps = 3h, droplet r = 0.25, one backward Euler step of dt = 1e-3). Observed:
omitting the prefix raises TypeError "NonlinearProblem.__init__() missing 1
required keyword-only argument: 'petsc_options_prefix'"; passing the constructed
problem to dolfinx.nls.petsc.NewtonSolver(comm, problem) raises AttributeError
"'NonlinearProblem' object has no attribute 'a'"; and passing J=ufl.derivative(
F, phi) as a keyword is accepted and solves.

Mutation control: T2_MUTATE=1 passes the prefix and never touches NewtonSolver,
so neither error text is produced.
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

N, DT, EPS_OVER_H, R = 32, 1e-3, 3.0, 0.25

# A half-built object prints "Exception ignored in __del__" when collected;
# class-level defaults keep each failure to the one exception under test.
dolfinx.nls.petsc.NewtonSolver._A = None
dolfinx.nls.petsc.NewtonSolver._b = None
for _attr in ("_snes", "_A", "_b", "_x", "_P_mat"):
    setattr(dolfinx.fem.petsc.NonlinearProblem, _attr, None)


def main() -> int:
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

    prefix_msg = ""
    try:
        if MUTATE:
            dolfinx.fem.petsc.NonlinearProblem(
                F, phi, petsc_options_prefix="t2_mp9_probe_")
        else:
            dolfinx.fem.petsc.NonlinearProblem(F, phi)
        prefix_accepted = True
    except TypeError as exc:
        prefix_accepted = False
        prefix_msg = f"{type(exc).__name__}: {exc}"

    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, phi, J=ufl.derivative(F, phi),
        petsc_options_prefix="t2_mp9_",
        petsc_options={"snes_type": "newtonls", "ksp_type": "preonly",
                       "pc_type": "lu"})

    legacy_msg = ""
    legacy_accepted = True
    if not MUTATE:
        try:
            dolfinx.nls.petsc.NewtonSolver(msh.comm, prob)
        except AttributeError as exc:
            legacy_accepted = False
            legacy_msg = f"{type(exc).__name__}: {exc}"

    if prefix_msg:
        print(f"no_prefix_error: {prefix_msg}")
    print(f"petsc_options_prefix_is_required={not prefix_accepted}")
    if legacy_msg:
        print(f"newtonsolver_error: {legacy_msg}")
    print(f"legacy_newtonsolver_cannot_wrap_it={not legacy_accepted}")

    prob.solve()
    reason = prob.solver.getConvergedReason()
    its = prob.solver.getIterationNumber()
    print(f"optional_J_keyword_accepted=True converged_reason={reason} "
          f"iterations={its}")
    print(f"solve_with_optional_J_converged={reason > 0}")
    print(f"solution_is_finite={bool(np.all(np.isfinite(phi.x.array)))}")

    if (not prefix_accepted and not legacy_accepted and reason > 0
            and bool(np.all(np.isfinite(phi.x.array)))):
        print("VERDICT=prefix_is_required_and_newtonsolver_cannot_wrap_it")
        return 0
    print("VERDICT=prefix_or_newtonsolver_was_accepted")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
