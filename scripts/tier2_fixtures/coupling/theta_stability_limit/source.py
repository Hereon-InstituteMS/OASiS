"""What a constant theta actually does, at three conductance ratios — settle,
oscillate, or run away — measured against the formula the knowledge predicts it
with.

THE CLAIM UNDER TEST is the one quantitative rule in the whole coupling
knowledge, and therefore the most exposed:

  "the driver's Jacobi+relaxation loop has amplification factor
   sqrt((1-theta)^2 + rho*theta^2)"
  "theta = 1.0 NEVER converges at rho = 1 and DIVERGES for rho > 1. It is not a
   'no relaxation, exact for linear problems' setting on this driver"
  "at rho = 1, theta = 0.5 converges and theta = 1.0 oscillates forever WITHOUT
   blowing up — the interface value simply never settles"
  "at rho = 4, theta = 0.5 with a CONSTANT accelerator DIVERGES — the interface
   values run away by many orders of magnitude ... while theta = 0.2 converges.
   Nothing warns you in advance: a diverging coupling looks like a converging
   one for the first few iterations."

An amplification formula is a prediction, so this fixture uses it as one. For
each ratio and each theta it evaluates sqrt((1-theta)^2 + rho*theta^2) BEFORE
running anything, and then asserts the real coupling did what the formula said:
below 1 must settle, above 1 must run away, and at exactly 1 must do neither.

That is much stronger than checking a list of thetas somebody once observed,
because it fails if the formula stops describing the driver — which is what
would happen if the relaxation were changed from Jacobi to Gauss-Seidel, or
applied once per cycle instead of once per participant. Either change would
leave every pair fixture green.

WHAT IS MEASURED, and why not the obvious thing. Not the residual: it is
normalised by the export magnitude, so a diverging run saturates it near a
constant of order one instead of sending it to infinity — measured here, the
residual after a run that had blown up by twelve orders of magnitude was 0.70.
Not the raw interface value either: it sits near 310 K, so an error amplified
fiftyfold still leaves it within a factor of two of the answer. What separates
the three behaviours cleanly is the DEVIATION from the exact interface
temperature, which grows or shrinks as the amplification factor to the power of
the iteration count with nothing to hide behind.

rho is set by moving ONE constant, the left conductivity, so the three ratios
are the same problem at three material contrasts.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _couplinglib as L                                    # noqa: E402

RHOS = [1.0, 4.0, 9.0]
# Four thetas cover all three behaviours at every ratio, including BOTH
# marginal cases (rho=1 at theta=1.0 and rho=9 at theta=0.2, where the
# amplification factor is exactly 1). A fifth value bought nothing and
# each one costs sixty iterations of two solvers.
GRID = [0.1, 0.2, 0.5, 1.0]
MAX_ITER = 60
TOL = 1e-8

SETTLED = 1e-2          # deviation below this: the iteration is settling
RUNAWAY = 1.0           # deviation above this: it has run away
# Between the two lies the marginal case the knowledge describes as
# "oscillates forever WITHOUT blowing up".


def classify(dev: float) -> str:
    if not math.isfinite(dev) or dev > RUNAWAY:
        return "ran_away"
    if dev < SETTLED:
        return "settled"
    return "neither"


def body() -> None:
    L.require_available("skfem")
    print(f"initial_guess_T={L.SWEEP_T_INIT}")
    print(f"max_iter={MAX_ITER}")
    for rho in RHOS:
        p = L.problem_with_rho(rho)
        limit = 2.0 / (1.0 + rho)
        print(f"=== rho={rho:g} k_left={p.kl:.4f} T_exact={p.t_iface:.6f} "
              f"theta_opt={p.theta_opt('left'):.6f} "
              f"predicted_stability_limit={limit:.6f}")
        outcome = {}
        for th in GRID:
            amp = p.amplification("left", th)
            r = L.probe_theta(f"s{rho:g}_{th}", rho, th, MAX_ITER, tol=TOL,
                              quiet=True)
            dev = r["deviation"]
            what = classify(dev)
            outcome[th] = what
            tag = f"rho{rho:g}_theta{th}"
            shown = "inf" if not math.isfinite(dev) else f"{dev:.3e}"
            print(f"{tag}_amplification={amp:.4f}")
            print(f"{tag}_deviation_from_exact={shown}")
            print(f"{tag}_outcome={what}")

            # The formula's prediction, checked against the run.
            if amp > 1.001:
                L.check(what == "ran_away",
                        f"{tag}_predicted_divergence_did_not_happen",
                        f"amplification {amp:.4f} > 1 predicts the interface "
                        f"values run away; deviation ended at {shown}")
            elif amp < 0.999:
                L.check(what == "settled",
                        f"{tag}_predicted_settling_but_it_did_not",
                        f"amplification {amp:.4f} < 1 predicts a settling "
                        f"iteration; deviation ended at {shown}")
            else:
                # amplification exactly 1: neither settles nor blows up.
                L.check(what == "neither",
                        f"{tag}_marginal_case_did_not_hover",
                        f"amplification is exactly 1, so the interface value "
                        f"should never settle and never blow up; deviation "
                        f"ended at {shown}")
                L.check(not r["converged"], f"{tag}_converged_at_amplification_1",
                        "an iteration with unit amplification must not reach "
                        "the tolerance")

        settled = [t for t in GRID if outcome[t] == "settled"]
        away = [t for t in GRID if outcome[t] == "ran_away"]
        print(f"rho{rho:g}_settled={','.join(str(t) for t in settled) or 'none'}")
        print(f"rho{rho:g}_ran_away={','.join(str(t) for t in away) or 'none'}")
        print(f"rho{rho:g}_settled_count={len(settled)}")

        # theta = 1.0 gets its own assertion because the knowledge singles it
        # out: not a "no relaxation, exact for linear problems" setting.
        print(f"rho{rho:g}_theta1_outcome={outcome[1.0]}")
        if abs(rho - 1.0) < 1e-9:
            L.check(outcome[1.0] == "neither", "rho1_theta1_not_marginal",
                    "at rho=1 theta=1.0 must oscillate forever without blowing "
                    "up, which is the whole point of the warning")
        else:
            L.check(outcome[1.0] == "ran_away",
                    f"rho{rho:g}_theta1_did_not_diverge",
                    "the knowledge says theta=1.0 DIVERGES for rho > 1")

        # 1/(1+rho) must at least be among the settling ones.
        nearest = min(GRID, key=lambda t: abs(t - p.theta_opt("left")))
        print(f"rho{rho:g}_theta_opt_on_grid={nearest}")
        L.check(outcome[nearest] == "settled",
                f"rho{rho:g}_theta_opt_did_not_settle",
                f"1/(1+rho) = {p.theta_opt('left'):.4f} must settle")

    print(f"ratios_tested={len(RHOS)}")
    print(f"thetas_per_ratio={len(GRID)}")
    print(f"runs={len(RHOS) * len(GRID)}")


L.main(body)
