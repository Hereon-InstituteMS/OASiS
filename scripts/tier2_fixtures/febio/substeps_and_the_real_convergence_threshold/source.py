"""Tier-2: a single load step carries neo-Hookean far past 10% strain, and
the threshold depends on the material.

Verifies febio::hyperelasticity#1. The pitfall's operative content is
that the common rule of thumb — "use 10 substeps past 10% strain" — is
wrong by orders of magnitude for a coupled material, and that where a
single step DOES fail, more substeps fix it.

Four runs on the shipped cube, all at <time_steps> x <step_size> = 1:

  * neo-Hookean, ONE step, 50% compression — converges,
  * neo-Hookean, ONE step, 300% — still converges,
  * Mooney-Rivlin, ONE step, 100% — fails with
    `N negative jacobians detected.`,
  * the same Mooney-Rivlin case in TWO steps — converges.

The last pair is the claim in one comparison: the failure is real and
substeps are the fix, but the threshold is nowhere near 10% strain.
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


import re

NEO = ('    <material id="1" name="Material1" type="neo-Hookean">\n'
       "      <density>1.0</density><E>1000.0</E><v>0.3</v>\n"
       "    </material>")
MOONEY = ('    <material id="1" name="Material1" type="Mooney-Rivlin">\n'
          "      <density>1.0</density><c1>100.0</c1><c2>50.0</c2>"
          "<k>10000.0</k>\n"
          "    </material>")


def run_case(material: str, steps: int, stretch: float):
    base = L.template("hyperelasticity_3d_cube")
    deck = L.swap(base, NEO, material)
    deck = L.swap(deck, "<time_steps>10</time_steps>",
                  f"<time_steps>{steps}</time_steps>")
    deck = L.swap(deck, "<step_size>0.1</step_size>",
                  f"<step_size>{1.0 / steps}</step_size>")
    patched = re.sub(r'<value lc="1">[-\d.]+</value>',
                     f'<value lc="1">{stretch}</value>', deck)
    if patched == deck:
        L.die("the prescribed-displacement value was not found; the "
              "template changed and nothing was triggered")
    return L.run(patched, timeout=1200)


def main() -> int:
    cases = {
        "neo_1_step_50pct": (NEO, 1, -0.5),
        "neo_1_step_300pct": (NEO, 1, 3.0),
        "mooney_1_step_100pct": (MOONEY, 1, 1.0),
        "mooney_2_steps_100pct": (MOONEY, 2, 1.0),
    }
    got = {}
    for tag, (mat, steps, stretch) in cases.items():
        r = run_case(mat, steps, stretch)
        got[tag] = r
        print(f"{tag}: rc={r.rc} normal={int(r.normal_termination)} "
              f"steps_completed={r.steps_completed} "
              f"negative_jacobians="
              f"{int('negative jacobians detected.' in r.text)} "
              f"failed_to_converge="
              f"{int('------- failed to converge at time' in r.text)}")
    far_past_ten_percent = (got["neo_1_step_50pct"].rc == 0
                            and got["neo_1_step_50pct"].normal_termination
                            and got["neo_1_step_300pct"].rc == 0
                            and got["neo_1_step_300pct"].normal_termination)
    mooney_fails = (got["mooney_1_step_100pct"].rc != 0
                    and "negative jacobians detected."
                    in got["mooney_1_step_100pct"].text)
    substeps_fix_it = (got["mooney_2_steps_100pct"].rc == 0
                       and got["mooney_2_steps_100pct"].normal_termination)
    print(f"single_step_works_far_past_ten_percent="
          f"{int(far_past_ten_percent)}")
    print(f"mooney_rivlin_fails_in_one_step_at_100pct={int(mooney_fails)}")
    print(f"two_steps_are_enough={int(substeps_fix_it)}")
    good = far_past_ten_percent and mooney_fails and substeps_fix_it
    return L.report(good, "substep_threshold", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
