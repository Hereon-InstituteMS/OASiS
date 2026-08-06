"""Tier-2: the material is `von-Mises plasticity` and the yield parameter
is Y — every wrong name is a hard parse error that NAMES the tag.

Verifies febio::plasticity#1, which itself corrects an earlier catalogue
entry that claimed the material was "J2 plasticity", the parameter "Y0",
and the failure a WARNING followed by a silent default to zero. The
fixture executes all three of those and requires:

  * `J2 plasticity` is an invalid material type,
  * `Y0` and `yield_stress` are unrecognized tags named back verbatim,
  * and NOTHING is ever silently defaulted — no WARNING appears and no
    run completes on the wrong spellings.

The last assertion is the one that keeps the correction honest.
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
    right = L.template("plasticity_3d_uniaxial")
    bad_type = L.swap(right, 'type="von-Mises plasticity"',
                      'type="J2 plasticity"')
    r = L.run(right)
    wt = L.run(bad_type)
    type_msg = ('tag "material"' in wt.text
                and 'invalid value for attribute "type"' in wt.text)
    print(f"J2_plasticity: rc={wt.rc} read_failed={int(wt.read_failed)} "
          f"invalid_type={int(type_msg)} "
          f"no_warning={int('WARNING' not in wt.text)}")
    named = 0
    for spelling in ("Y0", "yield_stress"):
        w = L.run(L.swap(right, "<Y>250.0</Y>",
                         f"<{spelling}>250.0</{spelling}>"))
        hit = (f'tag "{spelling}"' in w.text
               and "unrecognized tag" in w.text)
        silent = w.normal_termination or w.rc == 0
        print(f"parameter_{spelling}: rc={w.rc} "
              f"read_failed={int(w.read_failed)} named_verbatim={int(hit)} "
              f"silently_accepted={int(silent)}")
        if hit and w.read_failed and w.rc != 0 and not silent:
            named += 1
    print(f"right: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    print(f"wrong_names_rejected_by_name={named} of 2")
    good = (type_msg and wt.read_failed and wt.rc != 0
            and "WARNING" not in wt.text and named == 2
            and r.rc == 0 and r.normal_termination)
    return L.report(good, "plasticity_names", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
