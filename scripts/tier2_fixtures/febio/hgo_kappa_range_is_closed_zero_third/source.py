"""Tier-2: kappa is hard-limited to [0, 1/3], the bound is INCLUSIVE, and
the check fires at initialisation.

Verifies febio::fiber_reinforced#2. Seven values executed: 0, 0.1 and
1/3 accepted; 0.34, 0.5, 1.0 and -0.1 rejected. The message quotes the
material's OWN name — `Invalid value for parameter: Material1.kappa` —
which is the contrast with the unnamed CDF property in damage#2.

The deck reads successfully first, so a wrapper watching only for
`Reading file ...FAILED!` misses this entirely; the fixture asserts
read_success on every rejected value.
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


ACCEPT = ("0", "0.1", "0.3333333333")
REJECT = ("0.34", "0.5", "1.0", "-0.1")
MSG = "Invalid value for parameter: Material1.kappa"


def main() -> int:
    base = L.template("fiber_reinforced_3d_hgo")
    ok = 0
    for v in ACCEPT:
        r = L.run(L.swap(base, "<kappa>0.1</kappa>", f"<kappa>{v}</kappa>"),
                  timeout=400)
        print(f"kappa={v}: rc={r.rc} normal={int(r.normal_termination)} "
              f"steps={r.steps_completed} rejected={int(r.has(MSG))}")
        if r.rc == 0 and r.normal_termination and not r.has(MSG):
            ok += 1
    bad = 0
    for v in REJECT:
        r = L.run(L.swap(base, "<kappa>0.1</kappa>", f"<kappa>{v}</kappa>"),
                  timeout=400)
        named = r.has(MSG)
        after_read = r.read_success and not r.read_failed
        print(f"kappa={v}: rc={r.rc} read_success={int(r.read_success)} "
              f"names_material_and_parameter={int(named)} "
              f"init_failed={int(r.has('Model initialization failed'))}")
        if named and after_read and r.rc != 0:
            bad += 1
    print(f"accepted={ok} of {len(ACCEPT)} rejected={bad} of {len(REJECT)}")
    good = ok == len(ACCEPT) and bad == len(REJECT)
    return L.report(good, "kappa_range", "reproduced", "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
