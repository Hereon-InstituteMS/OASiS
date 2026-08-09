"""Tier-2 for fenics reaction_diffusion#7: stiff kinetics need an implicit
treatment. An explicit (theta = 0) reaction term diverges at a step size that
backward Euler (theta = 1) handles without effort -- and the SNES keeps
reporting CONVERGED while it happens, because each step really is solved
correctly; it is the SCHEME that is unstable.

Two-species 2A <-> B on a 24x24 unit square, D = 0.01, forward rate 100, dt =
0.05, at most 12 steps, blow-up declared when max|c| exceeds 1e6. The reaction
term is evaluated as theta*r(w) + (1 - theta)*r(w_n); diffusion stays implicit
in both runs, so theta alone is under test.

Observed on dolfinx 0.10.0: theta = 0 passes 1e6 at step 3 with the SNES
returning a positive converged reason on every one of those steps, while
theta = 1 completes all 12 steps with max|c| below 1.0.

Mutation control: T2_MUTATE=1 runs theta = 1 in the slot under test.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N, D, KF, KR, DT, NSTEP, BLOWUP = 24, 0.01, 100.0, 1.0, 0.05, 12, 1.0e6


def run(theta: float):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
    w = dolfinx.fem.Function(W)
    w_n = dolfinx.fem.Function(W)
    w_n.sub(0).interpolate(lambda x: 1.0 + 0.5 * np.sin(2 * np.pi * x[0]))
    w_n.sub(1).interpolate(lambda x: np.full_like(x[0], 0.2))
    w_n.x.scatter_forward()
    w.x.array[:] = w_n.x.array
    A, B = ufl.split(w)
    An, Bn = ufl.split(w_n)
    va, vb = ufl.TestFunctions(W)
    r = theta * (KF * A * A - KR * B) + (1.0 - theta) * (KF * An * An - KR * Bn)
    F = (((A - An) / DT) * va * ufl.dx
         + D * ufl.dot(ufl.grad(A), ufl.grad(va)) * ufl.dx
         + 2 * r * va * ufl.dx
         + ((B - Bn) / DT) * vb * ufl.dx
         + D * ufl.dot(ufl.grad(B), ufl.grad(vb)) * ufl.dx
         - r * vb * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, w, petsc_options_prefix=f"t2_rd7_{int(theta * 10)}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    reasons, peak, blew_at = [], 0.0, None
    for step in range(1, NSTEP + 1):
        try:
            prob.solve()
        except Exception:  # noqa: BLE001 - divergence is the point
            blew_at = step
            break
        w.x.scatter_forward()
        reasons.append(int(prob.solver.getConvergedReason()))
        w_n.x.array[:] = w.x.array
        peak = float(np.max(np.abs(w.x.array)))
        if not np.all(np.isfinite(w.x.array)) or peak > BLOWUP:
            blew_at = step
            break
    return blew_at, peak, reasons


def main() -> int:
    theta_t = 1.0 if MUTATE else 0.0
    blew_t, peak_t, reasons_t = run(theta_t)
    blew_r, peak_r, reasons_r = run(1.0)

    print(f"theta_under_test={theta_t} blew_up_at_step={blew_t} "
          f"max_abs_c={peak_t:.3e} snes_reasons={reasons_t}")
    print(f"reference_theta=1.0 blew_up_at_step={blew_r} "
          f"max_abs_c={peak_r:.3e} steps_completed={len(reasons_r)}")
    print(f"backward_euler_completes_every_step="
          f"{blew_r is None and len(reasons_r) == NSTEP}")
    print(f"backward_euler_stays_below_one={peak_r < 1.0}")
    blew = blew_t is not None
    early = blew and blew_t <= 5
    all_converged = bool(reasons_t) and all(r > 0 for r in reasons_t)
    print(f"explicit_reaction_blows_up={blew}")
    print(f"explicit_reaction_blows_up_within_five_steps={early}")
    print(f"snes_reported_converged_on_every_diverging_step={all_converged}")

    if (blew_r is None and peak_r < 1.0 and blew and early and all_converged):
        print("VERDICT=explicit_reaction_is_unstable_while_snes_says_converged")
        return 0
    print("VERDICT=explicit_reaction_was_stable")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
