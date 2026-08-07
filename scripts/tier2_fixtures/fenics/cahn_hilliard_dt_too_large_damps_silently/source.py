"""Tier-2 for fenics cahn_hilliard#0: the Cahn-Hilliard system is stiff and dt
must be small, but a wrong dt does NOT announce itself as DIVERGED_FNORM_NAN.

Unit square 24x24, P1 x P1 mixed, lmbda = 1e-2, M = 1, theta = 0.5, initial
concentration 0.63 +/- 0.01 uniform noise (seeded).

Wrong variant: a comfortable-looking dt = 1.0. Observed on dolfinx 0.10.0 it
converges in 2-4 Newton iterations per step with no error of any kind, and the
concentration is flattened onto the initial mean -- min and max agree with the
initial mean to four significant figures after two steps -- so no phase
separation can ever occur. An intermediate dt = 1e-4 instead makes Newton fail
outright (DIVERGED_MAX_IT / DIVERGED_LINE_SEARCH). Only dt = 5e-6 both
converges and keeps the perturbation alive. DIVERGED_FNORM_NAN is seen at no dt.

Mutation control: T2_MUTATE=1 replaces both pathological dt values with the
working 5e-6, so the flattening and the Newton failure both disappear.
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

N, LMBDA, MOB, THETA = 24, 1.0e-2, 1.0, 0.5
DT_SMALL = 5.0e-6
DT_BIG = DT_SMALL if MUTATE else 1.0
DT_MID = DT_SMALL if MUTATE else 1.0e-4


def build(tag: str, dt_val: float):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    ME = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
    u, u0 = dolfinx.fem.Function(ME), dolfinx.fem.Function(ME)
    rng = np.random.default_rng(7)
    u.sub(0).interpolate(lambda x: 0.63 + 0.02 * (0.5 - rng.random(x.shape[1])))
    u.sub(1).interpolate(lambda x: np.zeros(x.shape[1]))
    u.x.scatter_forward()
    u0.x.array[:] = u.x.array
    q, v = ufl.TestFunctions(ME)
    c, mu = ufl.split(u)
    c0, mu0 = ufl.split(u0)
    cv = ufl.variable(c)
    dfdc = ufl.diff(100.0 * cv**2 * (1 - cv) ** 2, cv)
    mu_mid = (1.0 - THETA) * mu0 + THETA * mu
    dt = dolfinx.fem.Constant(msh, dt_val)
    F = ((c - c0) * q * ufl.dx
         + dt * MOB * ufl.dot(ufl.grad(mu_mid), ufl.grad(q)) * ufl.dx
         + mu * v * ufl.dx - dfdc * v * ufl.dx
         - LMBDA * ufl.dot(ufl.grad(c), ufl.grad(v)) * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, u, petsc_options_prefix=f"t2_ch0_{tag}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_max_it": 30})
    _, cdofs = ME.sub(0).collapse()
    return u, u0, prob, np.asarray(cdofs, dtype=np.int32)


def march(tag: str, dt_val: float, steps: int):
    u, u0, prob, cdofs = build(tag, dt_val)
    c_start = u.x.array[cdofs].copy()
    reasons = []
    for _ in range(steps):
        u0.x.array[:] = u.x.array
        prob.solve()
        u.x.scatter_forward()
        reasons.append(prob.solver.getConvergedReason())
        if reasons[-1] <= 0:
            break
    c = u.x.array[cdofs]
    return dict(reasons=reasons, mean0=float(c_start.mean()),
                spread0=float(c_start.max() - c_start.min()),
                cmin=float(c.min()), cmax=float(c.max()),
                spread=float(c.max() - c.min()),
                nan=bool(np.isnan(u.x.array).any()))


def show(tag, dt_val, r):
    print(f"{tag}: dt={dt_val:g} reasons={r['reasons']} "
          f"c_range=[{r['cmin']:.6f}, {r['cmax']:.6f}] "
          f"spread0={r['spread0']:.3e} spread={r['spread']:.3e} "
          f"nan_in_field={r['nan']}")


def main() -> int:
    big = march("big", DT_BIG, 2)
    mid = march("mid", DT_MID, 2)
    small = march("small", DT_SMALL, 3)
    show("too_large_dt", DT_BIG, big)
    show("intermediate_dt", DT_MID, mid)
    show("small_dt", DT_SMALL, small)

    big_ok = all(r > 0 for r in big["reasons"])
    flat = (f"{big['cmin']:.4g}" == f"{big['mean0']:.4g}"
            and f"{big['cmax']:.4g}" == f"{big['mean0']:.4g}")
    print(f"too_large_dt_converged_without_error={big_ok}")
    print(f"too_large_dt_field_is_the_initial_mean_to_4_digits={flat}")
    print(f"big_dt_converges_but_flattens={big_ok and flat}")

    mid_bad = [r for r in mid["reasons"] if r <= 0]
    print(f"intermediate_dt_newton_fails={bool(mid_bad)} "
          f"failing_reason={mid_bad[0] if mid_bad else None}")
    print("intermediate_dt_failure_is_max_it_or_line_search="
          f"{bool(mid_bad) and mid_bad[0] in (-5, -6)}")

    small_ok = (all(r > 0 for r in small["reasons"])
                and small["spread"] > 0.5 * small["spread0"])
    print(f"small_dt_converges_and_keeps_the_perturbation={small_ok}")

    all_reasons = big["reasons"] + mid["reasons"] + small["reasons"]
    no_nan = (-4 not in all_reasons
              and not (big["nan"] or mid["nan"] or small["nan"]))
    print(f"all_reasons={all_reasons}")
    print(f"diverged_fnorm_nan_never_seen={no_nan}")

    if (big_ok and flat and mid_bad and mid_bad[0] in (-5, -6)
            and small_ok and no_nan):
        print("VERDICT=too_large_dt_is_silent_watch_the_spread_not_the_reason")
        return 0
    print("VERDICT=dt_was_fine")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
