"""Tier-2: a `biphasic` material does not accept <solute> children.

Verifies febio::multiphasic#1. Reported as an unknown TAG, so nothing in
the message mentions multiphasic — a reader is told the tag does not
exist rather than that the material is the wrong one.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

def main() -> int:
    right = L.template("biphasic_3d_confined")
    wrong = L.swap(right, '      <permeability type="perm-const-iso">',
                   '      <solute sol="1"/>\n'
                   '      <permeability type="perm-const-iso">')
    w = L.run(wrong)
    r = L.run(right)
    msg = 'tag "solute"' in w.text and "unrecognized tag" in w.text
    no_hint = "multiphasic" not in w.text
    print(f"wrong: rc={w.rc} read_failed={int(w.read_failed)} "
          f"unrecognized_tag={int(msg)} "
          f"never_mentions_multiphasic={int(no_hint)}")
    print(f"right: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    good = (msg and no_hint and w.read_failed and w.rc != 0
            and r.rc == 0 and r.normal_termination
            and "unrecognized tag" not in r.text)
    return L.report(good, "solute_in_biphasic", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
