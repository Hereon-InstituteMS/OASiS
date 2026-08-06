"""Tier-2: with the module left as solid, a fluid deck fails on the SOLVER.

Verifies febio::fluid#0. The solver factory is module-scoped and <Control>
is read before <Material>, so the first rejection is the solver type —
a line you did not edit. The fixture pins the misdirection: the message
names "solver", and the <Material> section is never reached, so nothing in
the output mentions the material.
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
    right = L.template("fluid_3d_channel")
    wrong = L.swap(right, '<Module type="fluid"/>', '<Module type="solid"/>')
    w = L.run(wrong)
    r = L.run(right)
    msg = ('tag "solver"' in w.text
           and 'invalid value for attribute "type"' in w.text)
    silent_about_material = "Material1" not in w.text
    print(f"wrong: rc={w.rc} read_failed={int(w.read_failed)} "
          f"names_solver={int(msg)} "
          f"never_names_the_material={int(silent_about_material)}")
    print(f"right: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    good = (msg and silent_about_material and w.read_failed and w.rc != 0
            and r.rc == 0 and r.normal_termination
            and 'tag "solver"' not in r.text)
    if not good:
        print(w.text[:900])
    return L.report(good, "fluid_module_solver", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
