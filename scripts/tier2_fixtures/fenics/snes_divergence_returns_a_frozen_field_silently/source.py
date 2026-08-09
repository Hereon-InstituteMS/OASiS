"""Tier-2 for fenics multiphase#0: `problem.solve()` on a dolfinx 0.10
NonlinearProblem returns the Function whether or not SNES converged - it does
not raise. In a time loop a solver failure becomes a frozen field that the
script keeps reporting as progress.

Allen-Cahn on a 32x32 unit square, eps = 2.56h, circular droplet, 5 backward
Euler steps. The wrong variant uses an oversized dt = 1.0: SNES returns
DIVERGED_LINE_SEARCH at every step, the iteration count collapses to 0 after
the first step, phi is bit-identical from step 2 onwards, and the loop still
runs to completion and would exit 0. The remedy the claim names is checked in
the same run: with "snes_error_if_not_converged": True the same solve raises
petsc4py.PETSc.Error carrying "SNESSolve has not converged due to
DIVERGED_LINE_SEARCH".

Mutation control: T2_MUTATE=1 selects dt = 1e-2; SNES converges every step, the
field advances, and the remedy raises nothing.
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

from petsc4py import PETSc  # noqa: E402

N, NSTEP, EPS_OVER_H, R = 32, 5, 2.56, 0.25
BAD_DT, GOOD_DT = 1.0, 1e-2
REASONS = {v: k for k, v in PETSc.SNES.ConvergedReason.__dict__.items()
           if isinstance(v, int)}


def setup(dt: float, prefix: str, extra: dict | None = None):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    eps = EPS_OVER_H / N
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    phi = dolfinx.fem.Function(V)
    phi_n = dolfinx.fem.Function(V)

    def ic(x):
        d = R - np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)
        return np.tanh(d / (eps * np.sqrt(2.0)))

    phi.interpolate(ic)
    phi_n.interpolate(ic)
    v = ufl.TestFunction(V)
    dt_c = dolfinx.fem.Constant(msh, dt)
    eps_c = dolfinx.fem.Constant(msh, eps)
    F = ((phi - phi_n) / dt_c * v * ufl.dx
         + eps_c * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
         + (1.0 / eps_c) * (phi ** 3 - phi) * v * ufl.dx)
    opts = {"snes_type": "newtonls", "ksp_type": "preonly", "pc_type": "lu"}
    opts.update(extra or {})
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, phi, petsc_options_prefix=prefix, petsc_options=opts)
    return phi, phi_n, prob


def run(dt: float, prefix: str):
    phi, phi_n, prob = setup(dt, prefix)
    reasons, its, states = [], [], []
    raised = ""
    for _ in range(NSTEP):
        try:
            out = prob.solve()
        except Exception as exc:  # noqa: BLE001
            raised = f"{type(exc).__name__}: {exc}"
            break
        if isinstance(out, tuple):
            out = out[0]
        reasons.append(prob.solver.getConvergedReason())
        its.append(prob.solver.getIterationNumber())
        states.append(phi.x.array.copy())
        phi_n.x.array[:] = phi.x.array
    return reasons, its, states, raised


def remedy_raises(dt: float, prefix: str) -> str:
    _, _, prob = setup(dt, prefix, {"snes_error_if_not_converged": True})
    try:
        prob.solve()
    except Exception as exc:  # noqa: BLE001
        return " ".join(str(exc).split())
    return ""


def main() -> int:
    dt = GOOD_DT if MUTATE else BAD_DT
    reasons, its, states, raised = run(dt, "t2_mp0_sel_")
    ref_reasons, _, ref_states, _ = run(GOOD_DT, "t2_mp0_ref_")

    print(f"selected_dt={dt} reasons={reasons} iterations={its}")
    print(f"reference_dt={GOOD_DT} reasons={ref_reasons}")
    completed = len(reasons) == NSTEP and raised == ""
    print(f"solve_raised_an_exception={raised != ''}")
    print(f"loop_completed_all_{NSTEP}_steps={completed}")
    all_bad = bool(reasons) and all(r < 0 for r in reasons)
    print(f"every_step_reason_negative={all_bad}")
    if reasons:
        print(f"reason_name={REASONS.get(reasons[-1])}")
    frozen = (len(states) == NSTEP
              and all(np.array_equal(states[1], s) for s in states[1:]))
    print(f"field_frozen_from_step_2={frozen}")
    print(f"final_phi_range=[{states[-1].min():.6e}, {states[-1].max():.6e}]")
    ref_moves = (len(ref_states) == NSTEP
                 and not np.array_equal(ref_states[1], ref_states[-1]))
    print(f"reference_run_keeps_moving={ref_moves}")

    msg = remedy_raises(dt, "t2_mp0_err_")
    print(f"snes_error_if_not_converged_message: {msg}")
    print(f"remedy_raises={msg != ''}")

    if completed and all_bad and frozen and ref_moves and msg:
        print("VERDICT=snes_divergence_is_silent_and_freezes_the_field")
        return 0
    print("VERDICT=solver_failure_was_reported")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
