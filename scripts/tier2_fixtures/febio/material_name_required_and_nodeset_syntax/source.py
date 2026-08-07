"""Tier-2: <material> needs name=, and <NodeSet> takes a comma list.

Verifies febio::biphasic#8, which bundles two 4.x schema requirements.
Both are executed on the biphasic template because the claim is filed
under biphasic; both messages are pinned, and each is required absent
from the other run so neither half can carry the other.
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
    right = L.template("biphasic_3d_confined")
    no_name = L.swap(right,
                     '<material id="1" name="Material1" type="biphasic">',
                     '<material id="1" type="biphasic">')
    bad_set = L.swap(right, '<NodeSet name="fix_bottom">1,2,3,4</NodeSet>',
                     '<NodeSet name="fix_bottom"><n id="1"/><n id="2"/>'
                     '<n id="3"/><n id="4"/></NodeSet>')
    wn = L.run(no_name)
    ws = L.run(bad_set)
    r = L.run(right)
    n_msg = ('tag "material"' in wn.text
             and 'missing attribute "name"' in wn.text)
    s_msg = ('tag "NodeSet"' in ws.text and "invalid value:" in ws.text)
    print(f"material_no_name: rc={wn.rc} read_failed={int(wn.read_failed)} "
          f"message={int(n_msg)} "
          f"not_the_nodeset_message={int('tag \"NodeSet\"' not in wn.text)}")
    print(f"nodeset_children: rc={ws.rc} read_failed={int(ws.read_failed)} "
          f"message={int(s_msg)} "
          f"not_the_name_message="
          f"{int('missing attribute \"name\"' not in ws.text)}")
    print(f"right: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    good = (n_msg and s_msg and wn.read_failed and ws.read_failed
            and wn.rc != 0 and ws.rc != 0
            and 'tag "NodeSet"' not in wn.text
            and 'missing attribute "name"' not in ws.text
            and r.rc == 0 and r.normal_termination)
    return L.report(good, "name_attr_and_nodeset", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
