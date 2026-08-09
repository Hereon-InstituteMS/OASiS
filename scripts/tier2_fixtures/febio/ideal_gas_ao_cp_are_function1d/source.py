"""Tier-2: <ao> and <cp> on `ideal gas` are FEFunction1D properties.

Verifies febio::heat#7. Each needs a type= attribute; a plain number is
rejected as an invalid attribute value, not as a bad number, which is the
part that misleads — the tag looks like a scalar parameter.
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
    wrong = L.swap(right, '<cp type="const"><value>3.5</value></cp>',
                   "<cp>3.5</cp>")
    return L.parse_error("cp_needs_type_attr", wrong=wrong, right=right,
                         message='tag "cp"',
                         also=('invalid value for attribute "type"',))


if __name__ == "__main__":
    sys.exit(main())
