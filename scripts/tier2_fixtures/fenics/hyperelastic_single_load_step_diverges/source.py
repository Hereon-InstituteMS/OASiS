"""Tier-2 for fenics hyperelasticity#1: putting the whole load on in one step
makes the hyperelastic Newton solve diverge, and incremental load stepping is
what recovers it.

A compressible Neo-Hookean cantilever (mu = 1, lambda = 10, P = dPsi/dF through
ufl.variable/ufl.diff) is pulled down by a body-force fem.Constant. The same
total load is applied twice: all at once, and ramped over 8 increments with the
previous solution kept as the initial guess.

Observed on dolfinx 0.10.0 / petsc4py 3.24.4: the single step ends at
problem.solver.getConvergedReason() = -5, SNES_DIVERGED_MAX_IT, with the residual
still at 2.1e-1 after the 30-iteration budget, while every one of the 8 increments
converges (reason 3) and ends at a residual around 1e-10. Nothing is raised
either way, and the failed single step still returns a displacement field -- a
different, smaller one (max|u| 2.41 against the correct 3.13), so the answer
looks plausible unless the reason is checked.

Mutation control: T2_MUTATE=1 applies the same total load in 8 increments for
the run under test, and the divergence disappears.
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

TOTAL_LOAD = 5.0
REASON_NAMES = {v: k for k, v in PETSc.SNES.ConvergedReason.__dict__.items()
                if isinstance(v, int)}


def run(n_steps: int, prefix: str):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    d = msh.geometry.dim
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (d,)))
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    f_var = ufl.variable(ufl.Identity(d) + ufl.grad(u))
    jac = ufl.det(f_var)
    mu, lmbda = 1.0, 10.0
    psi = (mu / 2) * (ufl.tr(f_var.T * f_var) - d) - mu * ufl.ln(jac) \
        + (lmbda / 2) * ufl.ln(jac) ** 2
    piola = ufl.diff(psi, f_var)
    body = dolfinx.fem.Constant(msh, (0.0, 0.0))
    res = ufl.inner(piola, ufl.grad(v)) * ufl.dx - ufl.inner(body, v) * ufl.dx

    msh.topology.create_connectivity(d - 1, d)
    left = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 0.0))
    bc = dolfinx.fem.dirichletbc(
        np.zeros(d), dolfinx.fem.locate_dofs_topological(V, d - 1, left), V)
    problem = dolfinx.fem.petsc.NonlinearProblem(
        res, u, bcs=[bc], petsc_options_prefix=prefix,
        petsc_options={"snes_type": "newtonls", "ksp_type": "preonly",
                       "pc_type": "lu", "snes_max_it": 30, "snes_rtol": 1e-8})
    reasons, residuals = [], []
    raised = ""
    for i in range(n_steps):
        body.value[1] = -TOTAL_LOAD * (i + 1) / n_steps
        try:
            problem.solve()
        except Exception as exc:  # noqa: BLE001
            raised = f"{type(exc).__name__}: {exc}"
        reasons.append(problem.solver.getConvergedReason())
        residuals.append(float(problem.solver.getFunctionNorm()))
    return reasons, residuals, float(np.max(np.abs(u.x.array))), raised


def main() -> int:
    steps = 8 if MUTATE else 1
    r_test, res_test, u_test, raised = run(steps, "t2_hy1a_")
    r_ref, res_ref, u_ref, _ = run(8, "t2_hy1b_")
    names = [REASON_NAMES.get(r, str(r)) for r in r_test]
    print(f"increments_under_test={steps} "
          f"load_per_increment={TOTAL_LOAD / steps:.3f}")
    print(f"under_test_reasons={r_test} names={names}")
    print(f"under_test_final_residual={res_test[-1]:.3e} "
          f"max_abs_u={u_test:.4f}")
    print(f"eight_increment_reference_reasons={r_ref}")
    print(f"eight_increment_final_residual={res_ref[-1]:.3e} "
          f"max_abs_u={u_ref:.4f}")
    print(f"solve_raised={raised!r}")

    diverged = any(r < 0 for r in r_test)
    print(f"single_step_diverged={diverged}")
    print(f"single_step_reason_name={names[0]}")
    stuck = res_test[-1] > 1e-2 and res_test[-1] > 1e4 * res_ref[-1]
    print(f"residual_at_failure_stays_far_above_the_converged_one={stuck}")
    stepped_ok = all(r > 0 for r in r_ref)
    print(f"stepped_run_converges_at_every_increment={stepped_ok}")
    wrong = abs(u_test / u_ref - 1.0) > 0.1
    print(f"failed_solve_returned_a_different_displacement={wrong}")
    print(f"nothing_was_raised={raised == ''}")

    if diverged and stuck and stepped_ok and wrong and raised == "":
        print("VERDICT=one_big_load_step_diverges_incremental_stepping_recovers")
        return 0
    print("VERDICT=single_load_step_was_fine")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
