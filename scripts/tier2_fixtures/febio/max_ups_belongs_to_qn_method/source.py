"""Tier-2: <max_ups> lives inside <qn_method>, and only BFGS/Broyden own it.

Verifies febio::hyperelasticity#5. Three placements, three outcomes:

  * flat inside <solver> — `tag "max_ups" ... unrecognized tag`,
  * inside <qn_method type="JFNK"> — the SAME message, because that
    strategy does not own the parameter,
  * inside <qn_method type="BFGS"> — accepted, runs.

Plus an unregistered strategy name, which fails differently:
`tag "qn_method" ... invalid value for attribute "type"`. The two message
shapes are what tell a reader whether the tag or the strategy is wrong.
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


ANCHOR = "      <max_refs>25</max_refs>\n"


def main() -> int:
    base = L.template("hyperelasticity_3d_cube")
    flat = L.swap(base, ANCHOR, "      <max_ups>10</max_ups>\n")
    jfnk = L.swap(base, ANCHOR,
                  '      <qn_method type="JFNK">'
                  "<max_ups>10</max_ups></qn_method>\n")
    bfgs = L.swap(base, ANCHOR,
                  '      <qn_method type="BFGS">'
                  "<max_ups>10</max_ups></qn_method>\n")
    bad = L.swap(base, ANCHOR,
                 '      <qn_method type="NOPE">'
                 "<max_ups>10</max_ups></qn_method>\n")
    rf, rj, rb, rn = (L.run(flat), L.run(jfnk), L.run(bfgs), L.run(bad))
    tag_msg = 'tag "max_ups"' , "unrecognized tag"
    f_ok = all(t in rf.text for t in tag_msg) and rf.read_failed
    j_ok = all(t in rj.text for t in tag_msg) and rj.read_failed
    b_ok = rb.rc == 0 and rb.normal_termination
    n_ok = ('tag "qn_method"' in rn.text
            and 'invalid value for attribute "type"' in rn.text
            and rn.read_failed)
    print(f"flat_in_solver: rc={rf.rc} unrecognized_tag={int(f_ok)}")
    print(f"in_JFNK: rc={rj.rc} unrecognized_tag={int(j_ok)}")
    print(f"in_BFGS: rc={rb.rc} normal={int(rb.normal_termination)} "
          f"steps={rb.steps_completed}")
    print(f"unregistered_strategy: rc={rn.rc} invalid_type={int(n_ok)} "
          f"not_the_max_ups_message="
          f"{int('tag \"max_ups\"' not in rn.text)}")
    good = (f_ok and j_ok and b_ok and n_ok
            and rf.rc != 0 and rj.rc != 0 and rn.rc != 0
            and 'tag "max_ups"' not in rn.text)
    return L.report(good, "max_ups_placement", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
