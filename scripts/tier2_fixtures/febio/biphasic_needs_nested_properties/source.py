"""Tier-2: a flat parameter and a missing nested property are
DISTINGUISHABLE from the message alone.

Verifies febio::biphasic#1. The `biphasic` material takes nested <solid>
and <permeability> properties, each with its own type=. The pitfall's
useful content is that the two mistakes report differently:

  * a flat parameter is named back verbatim —
    `tag "E" (line N) : unrecognized tag`,
  * a missing nested PROPERTY is reported as
    `Component "Material1" needs to have property "permeability" defined`,
    quoting the material's own name.

The fixture requires BOTH messages and requires each to be ABSENT from
the other run, because it is the discrimination that is being claimed,
not either message on its own.
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


PERM = ('      <permeability type="perm-const-iso">\n'
        "        <perm>0.001</perm>\n"
        "      </permeability>\n")


def main() -> int:
    right = L.template("biphasic_3d_confined")
    flat = L.swap(right, "      <phi0>0.2</phi0>\n",
                  "      <phi0>0.2</phi0>\n      <E>1000.0</E>\n")
    missing = L.drop(right, PERM)

    wf = L.run(flat)
    wm = L.run(missing)
    r = L.run(right)

    flat_msg = 'tag "E"' in wf.text and "unrecognized tag" in wf.text
    miss_msg = wm.has('Component "Material1" needs to have property '
                      '"permeability" defined')
    print(f"flat_parameter: rc={wf.rc} read_failed={int(wf.read_failed)} "
          f"names_the_tag={int(flat_msg)} "
          f"not_a_missing_property_message="
          f"{int('needs to have property' not in wf.text)}")
    print(f"missing_property: rc={wm.rc} read_failed={int(wm.read_failed)} "
          f"names_the_property={int(miss_msg)} "
          f"not_an_unrecognized_tag_message="
          f"{int('unrecognized tag' not in wm.text)}")
    print(f"right: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    good = (flat_msg and "needs to have property" not in wf.text
            and miss_msg and "unrecognized tag" not in wm.text
            and wf.read_failed and wm.read_failed
            and wf.rc != 0 and wm.rc != 0
            and r.rc == 0 and r.normal_termination)
    return L.report(good, "biphasic_flat_vs_missing", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
