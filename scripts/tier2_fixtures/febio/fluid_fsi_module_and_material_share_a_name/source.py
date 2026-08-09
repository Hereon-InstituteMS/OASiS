"""Tier-2: `fluid-FSI` is both a module and a material, and the material
only works inside the module.

Verifies febio::fluid_fsi#0. Naming <Module type="fluid"/> with a
`fluid-FSI` material fails on the SOLVER first, before the materials are
read — so the diagnostic points at <Control>.
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
    right = L.template("fluid_fsi_3d_block")
    wrong = L.swap(right, '<Module type="fluid-FSI"/>',
                   '<Module type="fluid"/>')
    return L.parse_error("fluid_fsi_module", wrong=wrong, right=right,
                         message='tag "solver"',
                         also=('invalid value for attribute "type"',))


if __name__ == "__main__":
    sys.exit(main())
