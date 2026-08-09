"""Tier-2: there is no <micro_viscosity> parameter on `polar fluid`.

Verifies febio::polar_fluid#1. The polar viscosity is a PROPERTY —
<polar type="polar linear"> with tau/alpha/beta/gamma — so the parameter
name that the literature suggests is simply not a tag.
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


POLAR = ('      <polar type="polar linear">\n'
         "        <tau>0.001</tau>\n"
         "        <alpha>0.0</alpha>\n"
         "        <beta>0.001</beta>\n"
         "        <gamma>0.001</gamma>\n"
         "      </polar>\n")


def main() -> int:
    right = L.template("polar_fluid_3d_channel")
    wrong = L.swap(right, POLAR,
                   "      <micro_viscosity>0.001</micro_viscosity>\n")
    return L.parse_error("micro_viscosity_tag", wrong=wrong, right=right,
                         message='tag "micro_viscosity"',
                         also=("unrecognized tag",))


if __name__ == "__main__":
    sys.exit(main())
