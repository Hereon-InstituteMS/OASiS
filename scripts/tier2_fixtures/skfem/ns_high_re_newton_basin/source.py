"""Tier-2: the Reynolds threshold, and a remedy that makes things worse.

Claim: skfem navier_stokes#4 -- "High Re: consider Picard (fixed-point) for
first few iterations, then Newton.  Signal: pure Newton at Re > ~200 diverges
from an at-rest initial guess (residual ... explodes within 2-3 iterations);
Picard for 5 iterations brings the state into Newton's convergence basin and
switching restores quadratic convergence."

Measured on skfem 12.0.1, lid-driven cavity, Taylor-Hood, one pressure DOF
pinned, always starting from rest.  Three separate things in the entry are
wrong.

  * "Re > ~200 DIVERGES" IS WRONG.  On MeshTri().refined(3), pure Newton from
    rest converges at Re = 100, 200 AND 400, in a handful of iterations each.
    The threshold is well above the quoted one.
  * WHERE IT DOES FAIL, THE CAUSE IS THE MESH, NOT THE REYNOLDS NUMBER.
    Pure Newton from rest fails at Re = 800 on refined(3) -- and converges at
    the SAME Re, from the SAME at-rest guess, on refined(4).  Attributing the
    failure to Newton's basin misreads an under-resolved discretisation as a
    nonlinear-solver problem.
  * "EXPLODES WITHIN 2-3 ITERATIONS" IS WRONG.  In the failing case the
    residual first STAGNATES for several iterations near its initial value
    before it blows up, so a watchdog armed for an early explosion sees a
    plausible-looking plateau instead.
  * THE PRESCRIBED REMEDY MAKES IT WORSE.  On refined(4) at Re = 800, where
    pure Newton converges, running five Picard steps first and then switching
    to Newton DIVERGES.  The advice takes a working run and breaks it.
  * A REMEDY THAT DOES WORK is continuation in Re: on the coarse refined(3)
    mesh, ramping through Re = 100, 200, 400, 600, 800 and starting each
    solve from the previous solution converges at every step, including the
    Re = 800 that fails from rest.

Mutation control: T2_MUTATE=1 applies this fixture's own documented fix at the
pathology site -- the failing Re = 800 run is given refine=4 instead of the
under-resolved refine=3, the one-token edit the middle bullet above says makes
the same Re from the same at-rest guess converge.  Nothing about Newton
changes.  That removes 're800_refine3_converged=False',
'same_Re_converges_on_a_finer_mesh=True', 're800_stagnates_before_blowing_up=True',
're800_explodes_within_three_iterations=False' and
'continuation_reaches_re800_where_from_rest_failed=True' from the output.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

from _harness import converged, iterate

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    ok = True

    # --- the quoted threshold ---------------------------------------------
    low = {}
    for re in (100.0, 200.0, 400.0):
        _, h = iterate(re, "newton", nit=15, refine=3)
        low[re] = converged(h)
        print(f"re{int(re)}_refine3_from_rest_converged={converged(h)} "
              f"iters={len(h)}")
    print(f"newton_converges_above_re_200="
          f"{low[200.0] and low[400.0]}")
    print(f"quoted_threshold_of_200_holds={not low[400.0]}")
    if not (low[100.0] and low[200.0] and low[400.0]):
        print("FAIL: pure Newton failed at or below Re = 400, so the quoted "
              "threshold may hold", file=sys.stderr)
        ok = False

    # --- where it fails, and why -------------------------------------------
    # THE PATHOLOGY: the under-resolved refined(3) discretisation at Re = 800.
    # T2_MUTATE applies the documented fix -- refine the mesh -- at this site.
    _, h3 = iterate(800.0, "newton", nit=20, refine=4 if MUTATE else 3)
    _, h4 = iterate(800.0, "newton", nit=20, refine=4)
    print(f"re800_refine3_converged={converged(h3)} iters={len(h3)}")
    print(f"re800_refine4_converged={converged(h4)} iters={len(h4)}")
    print(f"re800_history_refine3={[f'{v:.2e}' for v in h3[:6]]}")
    print(f"same_Re_converges_on_a_finer_mesh="
          f"{not converged(h3) and converged(h4)}")
    if converged(h3):
        print("FAIL: Re = 800 converged on the coarse mesh too",
              file=sys.stderr)
        ok = False
    if not converged(h4):
        print("FAIL: Re = 800 did not converge even on the finer mesh, so "
              "the failure is not a resolution artefact", file=sys.stderr)
        ok = False

    # --- does it explode within 2-3 iterations? ---------------------------
    finite = [v for v in h3 if np.isfinite(v)]
    early = finite[:4]
    plateau = max(early) / min(early) if early else 1e9
    print(f"re800_first_four_residuals={[f'{v:.3e}' for v in early]}")
    print(f"re800_early_spread_factor={plateau:.3f}")
    print(f"re800_stagnates_before_blowing_up={plateau < 3.0}")
    print(f"re800_explodes_within_three_iterations={plateau > 10.0}")
    if plateau > 10.0:
        print("FAIL: it really did explode within three iterations",
              file=sys.stderr)
        ok = False

    # --- the prescribed remedy ---------------------------------------------
    _, h4p = iterate(800.0, "newton", nit=20, refine=4, picard_first=5)
    print(f"re800_refine4_picard_first_5_converged={converged(h4p)}")
    print(f"re800_refine4_pure_newton_converged={converged(h4)}")
    print(f"picard_first_breaks_a_converging_run="
          f"{converged(h4) and not converged(h4p)}")
    if converged(h4p):
        print("FAIL: the Picard-first remedy converged, so it does not break "
              "the run", file=sys.stderr)
        ok = False

    # --- a remedy that does work -------------------------------------------
    x = None
    ramp = []
    for re in (100.0, 200.0, 400.0, 600.0, 800.0):
        x, h = iterate(re, "newton", nit=20, refine=3, u0=x)
        ramp.append(converged(h))
        print(f"continuation_re{int(re)}_converged={converged(h)} "
              f"iters={len(h)}")
    print(f"continuation_converges_at_every_step={all(ramp)}")
    print(f"continuation_reaches_re800_where_from_rest_failed="
          f"{ramp[-1] and not converged(h3)}")
    if not all(ramp):
        print("FAIL: continuation in Re did not converge at every step",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
