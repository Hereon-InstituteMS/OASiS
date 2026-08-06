"""Tier-2: dropping C(u) leaves a Stokes Jacobian and costs the quadratic rate.

Claim: skfem navier_stokes#1 -- the block system is [[A_visc + C(u), B^T],
[B, 0]] with C the linearised convection.  "Signal: omitting C(u) from the
BilinearForm gives a Stokes Jacobian and Newton converges linearly (not
quadratically) on Navier-Stokes -- residual ratio ~0.5 per iteration instead
of decreasing geometrically across asm + condense + spsolve."

Measured on skfem 12.0.1, lid-driven cavity at Re = 100, Taylor-Hood
ElementVector(TriP2)/TriP1 on MeshTri().refined(3), one pressure DOF pinned.
The residual reported is always that of the FULL Navier-Stokes equations, so
the two runs are compared on the same measure and only the Jacobian differs.

  * THE ORDER CLAIM IS CONFIRMED.  With the full Jacobian the residual falls
    quadratically and reaches machine zero in a handful of iterations; with
    the convection block omitted the ratio settles to a constant, which is
    linear convergence.
  * THE QUOTED RATE IS WRONG, and wrong in the safe-looking direction.  The
    measured ratio is nowhere near 0.5 -- it is several times smaller, so the
    Stokes-Jacobian iteration converges considerably FASTER than the entry
    leads you to expect.  A gate that watches for "ratio about 0.5" would not
    fire on this bug at all.
  * IT STILL CONVERGES.  The inexact Jacobian reaches the same solution to
    round-off, silently, in more iterations.  The defect costs iterations,
    not correctness, so the only reliable signal is the ORDER -- and a
    single-tolerance run cannot see it.
"""
from __future__ import annotations

import sys

import numpy as np

from _harness import converged, iterate

RE = 100.0


def ratios(hist, lo, hi):
    return [hist[i + 1] / hist[i] for i in range(lo, min(hi, len(hist) - 1))
            if hist[i] > 0]


def main() -> int:
    ok = True
    x_full, h_full = iterate(RE, "newton")
    x_stokes, h_stokes = iterate(RE, "stokes", nit=16)
    print(f"reynolds={RE:g}")
    print(f"full_jacobian_iters={len(h_full)}")
    print(f"stokes_jacobian_iters={len(h_stokes)}")
    print(f"full_history={[f'{v:.3e}' for v in h_full[:5]]}")
    print(f"stokes_history={[f'{v:.3e}' for v in h_stokes[:6]]}")
    print(f"full_jacobian_converged={converged(h_full)}")
    print(f"stokes_jacobian_converged={converged(h_stokes)}")
    print(f"stokes_needs_more_iterations={len(h_stokes) > len(h_full)}")
    if not converged(h_full):
        print("FAIL: the full Jacobian did not converge", file=sys.stderr)
        ok = False
    if len(h_stokes) <= len(h_full):
        print("FAIL: omitting C(u) cost no iterations", file=sys.stderr)
        ok = False

    r_full = ratios(h_full, 0, 3)
    r_stk = ratios(h_stokes, 2, 9)
    print(f"full_ratios={[f'{v:.3e}' for v in r_full]}")
    print(f"stokes_ratios={[f'{v:.4f}' for v in r_stk]}")
    quad = all(h_full[i + 1] < h_full[i] ** 1.5
               for i in range(len(h_full) - 2))
    print(f"full_jacobian_is_quadratic={quad}")
    spread = (max(r_stk) - min(r_stk)) if r_stk else 1.0
    print(f"stokes_ratio_spread={spread:.4f}")
    print(f"stokes_jacobian_rate_is_constant_linear={spread < 0.15}")
    mean = float(np.mean(r_stk)) if r_stk else 0.0
    print(f"stokes_mean_ratio={mean:.4f}")
    print(f"stokes_ratio_is_about_one_half={0.4 < mean < 0.6}")
    print(f"stokes_ratio_is_far_below_one_half={mean < 0.35}")
    if not quad:
        print("FAIL: the full Jacobian was not quadratic", file=sys.stderr)
        ok = False
    if spread >= 0.15:
        print("FAIL: the Stokes-Jacobian rate was not a constant linear "
              "factor", file=sys.stderr)
        ok = False
    if 0.4 < mean < 0.6:
        print("FAIL: the measured ratio really was about 0.5, so the entry's "
              "number is usable", file=sys.stderr)
        ok = False

    rel = float(np.linalg.norm(x_stokes - x_full)
                / np.linalg.norm(x_full))
    print(f"solution_relative_difference={rel:.3e}")
    print(f"same_solution_to_round_off={rel < 1e-8}")
    print(f"defect_costs_iterations_not_correctness="
          f"{converged(h_stokes) and rel < 1e-8}")
    if rel >= 1e-8:
        print("FAIL: the Stokes Jacobian reached a different solution",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
