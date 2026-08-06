"""Tier-2: factory type strings are case-SENSITIVE, enum values are not.

Verifies febio::linear_elasticity#8, both halves in one run, because the
claim is a CONTRAST and either half alone is uninformative:

  * type="HEX8" on <Elements> gives `Invalid element type`,
  * type="Zero Displacement" on <bc> gives
    `tag "bc" (line N) : invalid value for attribute "type"`,
  * while <analysis>STATIC</analysis>, static and Static all run and
    produce bit-identical logged stresses.

The third assertion is what stops a reader "fixing" a case-sensitivity
error by lower-casing an enum that never cared.
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


import hashlib

BND_CASE = ("  <Boundary>\n"
            '    <bc name="fix" type="Zero Displacement" node_set="bottom">\n'
            "      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n"
            "    </bc>\n"
            "  </Boundary>")


def control(word: str) -> str:
    return ("  <Control>\n"
            f"    <analysis>{word}</analysis>\n"
            "    <time_steps>2</time_steps>\n"
            "    <step_size>0.5</step_size>\n"
            '    <solver type="solid">\n'
            "      <symmetric_stiffness>symmetric</symmetric_stiffness>\n"
            "    </solver>\n"
            "  </Control>")


def main() -> int:
    mesh, _info = L.hex8_box(1)
    bad_elem = L.swap(mesh, '<Elements type="hex8"', '<Elements type="HEX8"')

    we = L.run(L.solid_deck(mesh=bad_elem))
    wb = L.run(L.solid_deck(boundary=BND_CASE))
    elem_msg = "Invalid element type" in we.text
    bc_msg = ('tag "bc"' in wb.text
              and 'invalid value for attribute "type"' in wb.text)
    print(f"elements_HEX8: rc={we.rc} read_failed={int(we.read_failed)} "
          f"invalid_element_type={int(elem_msg)}")
    print(f"bc_titlecase: rc={wb.rc} read_failed={int(wb.read_failed)} "
          f"invalid_type_attr={int(bc_msg)}")

    digests = {}
    for word in ("STATIC", "static", "Static"):
        r = L.run(L.solid_deck(control=control(word),
                              output=L.logfile(("element_data", "sx;sy;sz",
                                                "s.csv"))),
                  collect=("s.csv",))
        s = (r.files.get("s.csv") or "").strip()
        digests[word] = (r.rc, r.normal_termination,
                         hashlib.md5(s.encode()).hexdigest())
        print(f"analysis_{word}: rc={r.rc} "
              f"normal={int(r.normal_termination)} "
              f"stress_md5={digests[word][2][:12]}")
    enum_ran = all(rc == 0 and nt for rc, nt, _ in digests.values())
    enum_same = len({d for _, _, d in digests.values()}) == 1
    print(f"enum_case_insensitive_and_bit_identical="
          f"{int(enum_ran and enum_same)}")

    good = (elem_msg and we.read_failed and we.rc != 0
            and bc_msg and wb.read_failed and wb.rc != 0
            and enum_ran and enum_same)
    return L.report(good, "type_case_sensitivity", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
