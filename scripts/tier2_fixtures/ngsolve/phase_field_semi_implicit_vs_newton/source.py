"""Tier-2: lagging W'(c) keeps every Allen-Cahn step linear; the fully
implicit form needs a Newton solve per step and costs far more - but it is
the one that survives a large dt.

Claim: ngsolve phase_field#2 - evaluate W'(c) at c^n and solve linearly;
a fully implicit c^{n+1} inside W'(c) requires Newton at each step, which
dominates wall-clock.  Trade-off: the semi-implicit stability constant
shrinks, so use a smaller dt if you see ringing.

Wrong variant: put the trial function inside W'(c), i.e.
  a += ((1/dt)*(c - c_old)*w + eps^2*M*grad(c)*grad(w) + M*W'(c)*w)*dx
  and drive it with ngsolve.solvers.Newton every step.
Right variant : W'(gfc) with gfc the previous state - one linear solve.

Observed on NGSolve 6.2.2604, unit_square maxh=0.06 (ne=634), P1, droplet
radius 0.3, eps=0.06, M=20 (2026-08-06):
  dt=2e-4, 60 steps
    semi-implicit  60 linear solves,   0.02 s
    fully implicit 180 Newton iters (3.0/step, flag 0 every step), 0.81 s
    final fields agree to 3.18e-05 in L2
  dt=0.2, 5 steps
    semi-implicit  c in (29.204, 130.371) - blown up
    fully implicit c stays inside [0, 1]

Correction to the claim: the Newton count here is ~3 inner iterations per
step, not the quoted 5-10, and the wall-clock penalty is ~40x, not 5-10x.

Mutation control.  T2_MUTATE=1 lags W' inside the fully implicit form --
`MOB * w_prime(c) * w` becomes `MOB * w_prime(cold) * w`, the documented fix of
evaluating W' at the previous state.  The residual is then linear, so Newton no
longer needs several inner iterations per step and the big-dt run loses its
stability advantage: `newton_iters_per_step_gt_2`,
`newton_needs_at_least_3x_the_linear_solves` and `bigdt_newton_stayed_bounded`
print False and the fixture goes red.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time

from netgen.geom2d import unit_square
from ngsolve import (BilinearForm, GridFunction, H1, Integrate, LinearForm,
                     Mesh, dx, exp, grad, sqrt, x, y)
from ngsolve.solvers import Newton

EPS, MOB, RAD = 0.06, 20.0, 0.30
DT_SMALL, N_SMALL = 2e-4, 60
DT_BIG, N_BIG = 0.2, 5

# Mutation control: lag W' in the fully implicit form -- the documented fix --
# so the nonlinearity whose cost and stability this fixture measures is gone.
MUTATE = os.environ.get("T2_MUTATE") == "1"


def w_prime(c):
    return 2 * c * (1 - c) * (1 - 2 * c)


def droplet():
    r = sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    e2 = exp(-2 * (r - RAD) / (2 * EPS))
    return 0.5 + 0.5 * (e2 - 1) / (e2 + 1)


def capture_fds(fn):
    with tempfile.TemporaryFile(mode="w+") as tmp:
        so, se = os.dup(1), os.dup(2)
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(tmp.fileno(), 1)
            os.dup2(tmp.fileno(), 2)
            value = fn()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(so, 1)
            os.dup2(se, 2)
            os.close(so)
            os.close(se)
        tmp.seek(0)
        return value, tmp.read()


def main() -> int:
    ok = True
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.06))
    fes = H1(mesh, order=1)
    c, w = fes.TnT()
    print(f"mesh_ne={mesh.ne} ndof={fes.ndof}")

    def semi(dt, nsteps):
        lhs = BilinearForm(((1 / dt) * c * w + EPS ** 2 * MOB
                            * grad(c) * grad(w)) * dx).Assemble()
        inv = lhs.mat.Inverse(fes.FreeDofs())
        gfc = GridFunction(fes)
        gfc.Set(droplet())
        t0 = time.perf_counter()
        for _ in range(nsteps):
            rhs = LinearForm(((1 / dt) * gfc - MOB * w_prime(gfc)) * w * dx)
            rhs.Assemble()
            gfc.vec.data = inv * rhs.vec
        return gfc, time.perf_counter() - t0, nsteps

    def implicit(dt, nsteps, printing=False):
        gfn = GridFunction(fes)
        gfn.Set(droplet())
        cold = GridFunction(fes)
        a = BilinearForm(fes)
        a += ((1 / dt) * (c - cold) * w
              + EPS ** 2 * MOB * grad(c) * grad(w)
              + MOB * w_prime(cold if MUTATE else c) * w) * dx
        t0 = time.perf_counter()
        total, flags = 0, []
        for _ in range(nsteps):
            cold.vec.data = gfn.vec
            flag, nit = Newton(a, gfn, printing=printing, maxit=50)
            flags.append(flag)
            total += nit
        return gfn, time.perf_counter() - t0, total, flags

    # ---- small dt: both stable, compare cost and answer -----------------
    g_semi, t_semi, n_lin = semi(DT_SMALL, N_SMALL)
    (g_imp, t_imp, n_newton, flags), chatter = capture_fds(
        lambda: implicit(DT_SMALL, N_SMALL, printing=True))
    per_step = n_newton / N_SMALL
    print(f"semi_implicit_linear_solves={n_lin}")
    print("semi_implicit_nonlinear_iterations=0")
    print(f"newton_total_iterations={n_newton}")
    print(f"newton_iters_per_step_gt_2={per_step > 2.0}")
    print(f"newton_converged_every_step={all(f == 0 for f in flags)}")
    print(f"newton_printed_iteration_banner={'Newton iteration' in chatter}")
    print(f"newton_chatter_head={chatter.splitlines()[:2]}")
    if not per_step > 2.0:
        print(f"FAIL: the fully implicit form needed only {per_step:.2f} "
              f"Newton iterations per step; it is not behaving nonlinearly",
              file=sys.stderr)
        ok = False
    if not all(f == 0 for f in flags):
        print(f"FAIL: Newton failed on the small-dt problem: {flags}",
              file=sys.stderr)
        ok = False
    if "Newton iteration" not in chatter:
        print("FAIL: NGSolve's Newton printed no iteration banner",
              file=sys.stderr)
        ok = False

    diff = math.sqrt(Integrate((g_semi - g_imp) ** 2, mesh))
    norm = math.sqrt(Integrate(g_semi * g_semi, mesh))
    print(f"semi_and_newton_agree_within_1pct={diff / norm < 0.01}")
    if not diff / norm < 0.01:
        print(f"FAIL: lagging W' changed the answer by {diff / norm:.3e} "
              f"relative", file=sys.stderr)
        ok = False

    # Timing is measured and reported, never asserted: a busy host would
    # otherwise turn this fixture red for reasons unrelated to the claim.  The
    # cost difference is asserted as a count of linear solves instead -- the
    # semi-implicit scheme does exactly one per step, Newton does one per
    # Newton iteration -- which is what the wall-clock was standing in for and
    # holds on any machine.
    print(f"newton_over_semi_time_observed={t_imp / t_semi:.2f}")
    print(f"semi_implicit_solves_per_step={n_lin / N_SMALL:.2f}")
    print(f"newton_solves_per_step={n_newton / N_SMALL:.2f}")
    print(f"newton_needs_at_least_3x_the_linear_solves="
          f"{n_newton >= 3 * n_lin}")
    if not n_newton >= 3 * n_lin:
        print(f"FAIL: the Newton path needed only {n_newton} linear solves "
              f"against {n_lin} for the semi-implicit path", file=sys.stderr)
        ok = False

    # ---- large dt: the stability price of lagging -----------------------
    g_semi_big, _, _ = semi(DT_BIG, N_BIG)
    g_imp_big, _, _, flags_big = implicit(DT_BIG, N_BIG)
    lo_s, hi_s = min(g_semi_big.vec), max(g_semi_big.vec)
    lo_i, hi_i = min(g_imp_big.vec), max(g_imp_big.vec)
    print(f"bigdt_semi_range=({lo_s:.4f},{hi_s:.4f})")
    print(f"bigdt_newton_range=({lo_i:.4f},{hi_i:.4f})")
    print(f"bigdt_semi_implicit_blew_up={hi_s > 2.0 or lo_s < -1.0}")
    print("bigdt_newton_stayed_bounded="
          f"{-0.05 < lo_i and hi_i < 1.05}")
    if not (hi_s > 2.0 or lo_s < -1.0):
        print(f"FAIL: the semi-implicit scheme survived dt={DT_BIG}; the "
              f"stability trade-off is absent", file=sys.stderr)
        ok = False
    if not (-0.05 < lo_i and hi_i < 1.05):
        print(f"FAIL: the fully implicit scheme also left [0,1] at "
              f"dt={DT_BIG} ({lo_i:.4f}, {hi_i:.4f})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
