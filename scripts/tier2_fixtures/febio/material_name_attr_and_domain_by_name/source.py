"""Tier-2: <material> needs name=, and <SolidDomain> must cite the NAME.

Verifies febio::linear_elasticity#0 — the first two errors a hand-written
FEBio 4 deck hits. Three observations, each against the same positive
control:

  * a <material> with only id= is rejected by the reader,
  * a <SolidDomain mat="1"> pointing at the numeric id is rejected,
  * mat= on the <Elements> tag itself (a 3.x leftover) is SILENTLY
    ignored — no mat=, mat="1", mat="Material1" and mat="NONSENSE" all
    run and write bit-identical node positions, so it is not a
    substitute for <MeshDomains>.

The third is the part that bites: the wrong-looking attribute that gets
no complaint is the one that leaves the domain unbound.
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


def main() -> int:
    mat_noname = ("  <Material>\n"
                  '    <material id="1" type="isotropic elastic">\n'
                  "      <density>1.0</density><E>1000.0</E><v>0.3</v>\n"
                  "    </material>\n"
                  "  </Material>")
    dom_by_id = ("  <MeshDomains>\n"
                 '    <SolidDomain name="Part1" mat="1"/>\n'
                 "  </MeshDomains>")

    if MUTATE:
        print("mutation=the_unnamed_material_gets_its_name_back")
        mat_noname = mat_noname.replace(
            '<material id="1" type=',
            '<material id="1" name="Material1" type=')
    w1 = L.run(L.solid_deck(material=mat_noname))
    w2 = L.run(L.solid_deck(domains=dom_by_id))
    ok = L.run(L.solid_deck())

    m1 = 'tag "material"' in w1.text and 'missing attribute "name"' in w1.text
    m2 = ('tag "SolidDomain"' in w2.text
          and 'invalid value for attribute "mat"' in w2.text)
    print(f"no_name_attr: rc={w1.rc} read_failed={int(w1.read_failed)} "
          f"message={int(m1)}")
    print(f"domain_by_id: rc={w2.rc} read_failed={int(w2.read_failed)} "
          f"message={int(m2)}")
    print(f"control: rc={ok.rc} read_success={int(ok.read_success)} "
          f"normal={int(ok.normal_termination)} steps={ok.steps_completed}")

    # mat= on <Elements> is silently ignored.
    mesh, _info = L.hex8_box(1)
    digests = {}
    for tag, attr in (("absent", ""), ("id", ' mat="1"'),
                      ("name", ' mat="Material1"'),
                      ("nonsense", ' mat="NONSENSE"')):
        mm = L.swap(mesh, '<Elements type="hex8" name="Part1">',
                    f'<Elements type="hex8" name="Part1"{attr}>')
        r = L.run(L.solid_deck(mesh=mm,
                               output=L.logfile(("node_data", "x;y;z",
                                                 "p.csv"))),
                  collect=("p.csv",))
        pos = (r.files.get("p.csv") or "").strip()
        digests[tag] = (r.rc, r.normal_termination,
                        hashlib.md5(pos.encode()).hexdigest())
        print(f"elements_mat_{tag}: rc={r.rc} "
              f"normal={int(r.normal_termination)} "
              f"pos_md5={digests[tag][2][:12]}")
    all_ran = all(rc == 0 and nt for rc, nt, _ in digests.values())
    identical = len({d for _, _, d in digests.values()}) == 1
    print(f"elements_mat_silently_ignored={int(all_ran and identical)}")

    good = (m1 and w1.read_failed and w1.rc != 0
            and m2 and w2.read_failed and w2.rc != 0
            and ok.rc == 0 and ok.normal_termination
            and all_ran and identical)
    if not good:
        print("NOTE: one half did not reproduce; outputs above.")
    return L.report(good, "material_name_and_domain",
                    "reproduced", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
