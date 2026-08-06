"""Tier-2: omitting <MeshDomains> reports a bad node_set, not a bad domain.

Verifies febio::linear_elasticity#1. Node sets are only resolvable once
the domains are built, so a deck with no <MeshDomains> fails on the FIRST
<bc> instead, and the reported line number points at the <bc> — sending
you to hunt in a section that is correct.

The fixture also pins the misdirection itself: the message names
node_set, and the words MeshDomains / SolidDomain / domain appear nowhere
in the output.

MUTATION CONTROL. T2_MUTATE=1 leaves <MeshDomains> in place in the
"wrong" slot — the pathology removed. The deck then reads and runs, the
node_set misdirection never appears, and
'meshdomains_missing=reproduced' is no longer printed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

def main() -> int:
    if MUTATE:
        print("mutation=the_wrong_slot_keeps_its_meshdomains_section")
    wrong = L.solid_deck(domains=None if MUTATE else "")
    right = L.solid_deck()
    w = L.run(wrong)
    r = L.run(right)

    msg = ('tag "bc"' in w.text
           and 'invalid value for attribute "node_set"' in w.text)
    misdirected = not any(t in w.text for t in
                          ("MeshDomains", "SolidDomain", "MeshDomain"))
    print(f"wrong: rc={w.rc} read_failed={int(w.read_failed)} "
          f"names_node_set={int(msg)} "
          f"never_names_the_missing_section={int(misdirected)}")
    print(f"right: rc={r.rc} read_success={int(r.read_success)} "
          f"normal={int(r.normal_termination)} steps={r.steps_completed}")
    good = (msg and misdirected and w.read_failed and w.rc != 0
            and r.rc == 0 and r.normal_termination)
    if not good:
        print(w.text[:1200])
    return L.report(good, "meshdomains_missing", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
