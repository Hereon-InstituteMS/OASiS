"""Tier-2: there is no `isotropic Fourier` material in FEBio 4.12.

Verifies febio::heat#1. The FEMATERIAL_ID factory carries no
solid-conduction law at all, so the name every other FE code would
suggest is simply not registered. The pitfall also claims no short or
underscore variant works — the fixture executes all three spellings
("isotropic Fourier", "Fourier", "isotropic_Fourier") and requires all
three to be rejected identically.

The right answer is the `thermo-fluid` material with the fluid held at
rest, which is what the shipped heat_3d_bar template emits.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

SPELLINGS = ("isotropic Fourier", "Fourier", "isotropic_Fourier")
REAL = '<material id="1" name="Material1" type="thermo-fluid">'


def main() -> int:
    right = L.template("heat_3d_bar", n=2)
    r = L.run(right)
    print(f"control: rc={r.rc} read_success={int(r.read_success)} "
          f"normal={int(r.normal_termination)} steps={r.steps_completed}")
    bad = 0
    for s in SPELLINGS:
        w = L.run(L.swap(right, REAL,
                         f'<material id="1" name="Material1" type="{s}">'))
        msg = ('tag "material"' in w.text
               and 'invalid value for attribute "type"' in w.text)
        print(f"spelling={s!r}: rc={w.rc} read_failed={int(w.read_failed)} "
              f"invalid_type={int(msg)}")
        if msg and w.read_failed and w.rc != 0:
            bad += 1
    print(f"rejected_spellings={bad} of {len(SPELLINGS)}")
    good = (bad == len(SPELLINGS) and r.rc == 0 and r.normal_termination
            and 'invalid value for attribute "type"' not in r.text)
    return L.report(good, "isotropic_fourier", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
