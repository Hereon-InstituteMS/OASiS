"""Tier-2: `von-Mises plasticity` has exactly E, v, Y, H — no
back-stress, no kinematic term.

Verifies febio::plasticity#3. Both plausible additions are rejected by
name, so nothing is silently ignored; the fixture executes both and
requires the run never completes with the extra tag present.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

def main() -> int:
    right = L.template("plasticity_3d_uniaxial")
    r = L.run(right)
    rejected = 0
    for tag, value in (("kinematic", "1"), ("beta", "0.5")):
        w = L.run(L.swap(right, "      <H>1000.0</H>\n",
                         f"      <H>1000.0</H>\n"
                         f"      <{tag}>{value}</{tag}>\n"))
        hit = f'tag "{tag}"' in w.text and "unrecognized tag" in w.text
        print(f"extra_{tag}: rc={w.rc} read_failed={int(w.read_failed)} "
              f"named_verbatim={int(hit)} "
              f"silently_ignored={int(w.rc == 0)}")
        if hit and w.read_failed and w.rc != 0:
            rejected += 1
    print(f"right: rc={r.rc} normal={int(r.normal_termination)} "
          f"steps={r.steps_completed}")
    print(f"rejected_by_name={rejected} of 2")
    good = rejected == 2 and r.rc == 0 and r.normal_termination
    return L.report(good, "no_kinematic_hardening", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
