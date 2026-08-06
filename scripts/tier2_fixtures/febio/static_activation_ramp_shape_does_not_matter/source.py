"""Tier-2: under STATIC, the shape of the activation ramp does not change
the converged answer.

Verifies febio::active_contraction#3, which itself FALSIFIED an older
claim that a step change in T0 makes the first step fail and that SMOOTH
interpolation is required. The fixture executes the comparison:

  * a near-step activation — full amplitude reached in the first
    fiftieth of the run,
  * a linear ramp over the whole run,

both under <analysis>STATIC</analysis>. Both must complete every step and
reach a final configuration equal to well within five decimal places. If
a future FEBio ever does need the ramp smoothed, this fixture goes red
and the claim needs rewriting — which is the correct outcome.

MUTATION CONTROL — AND IT DOES NOT DISCRIMINATE (2026-08-07).
T2_MUTATE=1 neutralises the L.swap edits, so both runs use the SAME
deck. The fixture still passes, with max_abs_difference_in_final_
position=0.000e+00 and agree_to_five_decimals=1. That is structural,
not a defect in the mutation: this fixture asserts an EQUALITY — that
two activation ramps give the same converged answer — and an equality
is satisfied a fortiori when the two decks are identical. There is no
"pathology" here to remove; removing the edits removes the comparison.

What the fixture is missing is a POSITIVE CONTROL: a third deck that
this same measurement DOES separate — say the same STATIC run with a
different T0 — proving that max_abs_difference_in_final_position can
be large and that agree_to_five_decimals is therefore informative.
Without it, `diff < 1e-5` cannot tell "the ramp shape does not matter"
from "the two runs were the same run". The hook is left in place so the
negative result is re-runnable.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    # MUTATION CONTROL. The pathology this fixture reproduces is the
    # EDIT that turns a correct deck into the broken one, so removing
    # the pathology means not making that edit. Neutralising L.swap /
    # L.drop does exactly that: every deck built below is the correct
    # one, the pitfall is never triggered, the diagnostic must not
    # appear and the verdict token flips to not_reproduced.
    print("mutation=the_deck_edits_that_introduce_the_pitfall_are_"
          "neutralised")
    L.swap = lambda deck, old, new, **kw: deck
    L.drop = lambda deck, fragment: deck


SHIPPED_CURVE = ("<interpolate>SMOOTH</interpolate><extend>CONSTANT</extend>\n"
                 "      <points><pt>0,0</pt><pt>0.5,1</pt><pt>1,1</pt>"
                 "</points>")
NEAR_STEP = ("<interpolate>LINEAR</interpolate><extend>CONSTANT</extend>\n"
             "      <points><pt>0,0</pt><pt>0.02,1</pt><pt>1,1</pt>"
             "</points>")
LINEAR_RAMP = ("<interpolate>LINEAR</interpolate><extend>CONSTANT</extend>\n"
               "      <points><pt>0,0</pt><pt>1,1</pt></points>")


def positions(deck):
    r = L.run(L.add_logfile(deck, ("node_data", "z", "p.csv")),
              collect=("p.csv",), timeout=1200)
    blocks = L.parse_log_csv(r.files.get("p.csv") or "")
    last = blocks[-1][1] if blocks else {}
    return r, [v for k in sorted(last) for v in last[k]]


def main() -> int:
    base = L.template("active_contraction_3d_fiber")
    static = L.swap(base, "<analysis>DYNAMIC</analysis>",
                    "<analysis>STATIC</analysis>")
    r_step, z_step = positions(L.swap(static, SHIPPED_CURVE, NEAR_STEP))
    r_ramp, z_ramp = positions(L.swap(static, SHIPPED_CURVE, LINEAR_RAMP))
    if not z_step or len(z_step) != len(z_ramp):
        print("FAIL: a run logged no node positions")
        return L.report(False, "static_ramp_shape", "reproduced",
                        "not_reproduced")
    diff = max(abs(a - b) for a, b in zip(z_step, z_ramp))
    print(f"near_step_activation: rc={r_step.rc} "
          f"normal={int(r_step.normal_termination)} "
          f"steps={r_step.steps_completed} "
          f"failed_to_converge="
          f"{int('------- failed to converge at time' in r_step.text)}")
    print(f"linear_ramp: rc={r_ramp.rc} "
          f"normal={int(r_ramp.normal_termination)} "
          f"steps={r_ramp.steps_completed}")
    print(f"max_abs_difference_in_final_position={diff:.3e} "
          f"agree_to_five_decimals={int(diff < 1e-5)}")
    both_converged = (r_step.rc == 0 and r_ramp.rc == 0
                      and r_step.normal_termination
                      and r_ramp.normal_termination
                      and r_step.steps_completed
                      == r_ramp.steps_completed > 0
                      and "------- failed to converge at time"
                      not in r_step.text)
    good = both_converged and diff < 1e-5
    return L.report(good, "static_ramp_shape", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
