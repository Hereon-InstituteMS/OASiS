"""Tier-2: dropping the second convection term is Picard, and it is linear.

Claim: skfem navier_stokes#5 -- the convection linearisation is
(u_prev.grad)delta_u + (delta_u.grad)u_prev.  "Signal: dropping the second
term in the BilinearForm (Picard linearization instead of Newton) gives linear
convergence on the asm + condense + spsolve pipeline -- useful as a starter
but switch to full Newton for quadratic."

This is the neighbouring claim to navier_stokes#1 and must not be confused
with it: #1 drops the convection block ENTIRELY, leaving a Stokes Jacobian;
#5 keeps the transport term and drops only the reaction term.  Both cost the
quadratic rate, and this fixture runs all three side by side to show they are
three distinct iterations.

Measured on skfem 12.0.1, lid-driven cavity at Re = 100, Taylor-Hood on
MeshTri().refined(3), residual always that of the full Navier-Stokes system:

  * CONFIRMED.  Newton is quadratic; Picard converges at a constant ratio,
    i.e. linearly, and needs several times as many iterations.
  * it reaches the SAME solution to round-off, silently, so once again only
    the rate distinguishes them.
  * a caution the entry does not give: Picard is NOT simply "Newton minus a
    bit".  On this problem the Picard iteration is measurably SLOWER than
    the Stokes-Jacobian iteration of navier_stokes#1 -- keeping the transport
    term and dropping the reaction term converges at a larger ratio than
    dropping both.  So "closer to the true Jacobian" does not imply "faster
    here", and the two inexact variants cannot be ranked by inspection.
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
    runs = {}
    for variant in ("newton", "picard", "stokes"):
        x, h = iterate(RE, variant, nit=45)
        runs[variant] = (x, h)
        print(f"{variant}_iters={len(h)}")
        print(f"{variant}_history={[f'{v:.3e}' for v in h[:6]]}")
        print(f"{variant}_converged={converged(h)}")

    x_n, h_n = runs["newton"]
    x_p, h_p = runs["picard"]
    x_s, h_s = runs["stokes"]

    quad = all(h_n[i + 1] < h_n[i] ** 1.5 for i in range(len(h_n) - 2))
    print(f"newton_is_quadratic={quad}")
    r_p = ratios(h_p, 2, 9)
    print(f"picard_ratios={[f'{v:.4f}' for v in r_p]}")
    spread_p = (max(r_p) - min(r_p)) if r_p else 1.0
    print(f"picard_ratio_spread={spread_p:.4f}")
    print(f"picard_rate_is_constant_linear={spread_p < 0.15}")
    print(f"picard_needs_more_iterations_than_newton={len(h_p) > len(h_n)}")
    print(f"picard_converged={converged(h_p)}")
    if not quad:
        print("FAIL: full Newton was not quadratic", file=sys.stderr)
        ok = False
    if spread_p >= 0.15:
        print("FAIL: Picard's rate was not a constant linear factor",
              file=sys.stderr)
        ok = False
    if len(h_p) <= len(h_n):
        print("FAIL: Picard cost no iterations", file=sys.stderr)
        ok = False

    rel = float(np.linalg.norm(x_p - x_n) / np.linalg.norm(x_n))
    print(f"picard_vs_newton_solution_difference={rel:.3e}")
    print(f"picard_reaches_the_same_solution={rel < 1e-8}")
    if rel >= 1e-8:
        print("FAIL: Picard reached a different solution", file=sys.stderr)
        ok = False

    # --- the three variants are genuinely three different iterations ------
    r_s = ratios(h_s, 2, 9)
    mean_p = float(np.mean(r_p)) if r_p else 0.0
    mean_s = float(np.mean(r_s)) if r_s else 0.0
    print(f"picard_mean_ratio={mean_p:.4f}")
    print(f"stokes_mean_ratio={mean_s:.4f}")
    print(f"picard_and_stokes_rates_differ={abs(mean_p - mean_s) > 0.05}")
    print(f"picard_is_slower_than_stokes_jacobian={mean_p > mean_s}")
    print(f"three_distinct_iterations="
          f"{len({len(h_n), len(h_p), len(h_s)}) >= 2
             and abs(mean_p - mean_s) > 0.05}")
    if abs(mean_p - mean_s) <= 0.05:
        print("FAIL: Picard and the Stokes Jacobian converged at the same "
              "rate, so they are not distinguishable here", file=sys.stderr)
        ok = False
    if mean_p <= mean_s:
        print("FAIL: Picard was faster than the Stokes Jacobian, so the "
              "ranking caution does not apply", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
