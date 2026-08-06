"""Tier-2: the BC is `prescribed fluid temperature`, not
`prescribed temperature`.

Verifies febio::heat#2. Every temperature BC in FEBio 4.12 is scoped to
the thermo-fluid module and acts on a FLUID temperature DOF, so the
obvious name is unregistered. Rejected at parse by attribute value.
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


def main() -> int:
    right = L.template("heat_3d_bar", n=2)
    wrong = L.swap(right, 'type="prescribed fluid temperature"',
                   'type="prescribed temperature"')
    return L.parse_error("prescribed_temperature_bc",
                         wrong=wrong, right=right,
                         message='tag "bc"',
                         also=('invalid value for attribute "type"',))


if __name__ == "__main__":
    sys.exit(main())
