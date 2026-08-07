"""Tier-2: an uncoupled hyperelastic material without <k> is rejected
before anything is solved.

Verifies febio::hyperelasticity#0. The split the claim rests on is which
materials need <k> at all: Mooney-Rivlin, Ogden, Yeoh and Veronda-Westmann
are uncoupled and require it; neo-Hookean and Holmes-Mow are coupled, take
E and v, and have no <k> parameter.

The fixture runs the uncoupled material without <k> (rejected with
`K must be a positive number.`), the same material with <k> (runs), and
the coupled control (runs) — so the claim is pinned as a distinction and
not as a single error message.
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


NEO = ('    <material id="1" name="Material1" type="neo-Hookean">\n'
       "      <density>1.0</density><E>1000.0</E><v>0.3</v>\n"
       "    </material>")
MR_NO_K = ('    <material id="1" name="Material1" type="Mooney-Rivlin">\n'
           "      <density>1.0</density><c1>100.0</c1><c2>50.0</c2>\n"
           "    </material>")
MR_K = ('    <material id="1" name="Material1" type="Mooney-Rivlin">\n'
        "      <density>1.0</density><c1>100.0</c1><c2>50.0</c2>"
        "<k>10000.0</k>\n"
        "    </material>")
MSG = "K must be a positive number."


def main() -> int:
    base = L.template("hyperelasticity_3d_cube")
    w = L.run(L.swap(base, NEO, MR_NO_K))
    fixed = L.run(L.swap(base, NEO, MR_K))
    coupled = L.run(base)
    print(f"uncoupled_without_k: rc={w.rc} "
          f"read_success={int(w.read_success)} "
          f"read_failed={int(w.read_failed)} message={int(MSG in w.text)}")
    print(f"uncoupled_with_k: rc={fixed.rc} "
          f"normal={int(fixed.normal_termination)} "
          f"steps={fixed.steps_completed} "
          f"message_absent={int(MSG not in fixed.text)}")
    print(f"coupled_neo_hookean_no_k_parameter: rc={coupled.rc} "
          f"normal={int(coupled.normal_termination)} "
          f"steps={coupled.steps_completed} "
          f"message_absent={int(MSG not in coupled.text)}")
    good = (MSG in w.text and w.rc != 0
            and fixed.rc == 0 and fixed.normal_termination
            and MSG not in fixed.text
            and coupled.rc == 0 and coupled.normal_termination
            and MSG not in coupled.text)
    if not good:
        print(w.text[:900])
    return L.report(good, "uncoupled_needs_k", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
