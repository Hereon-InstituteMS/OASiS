"""Tier-2: <Dmax> is range-checked to [0,1] at INITIALISATION, and the
message carries an EMPTY name before the dot.

Verifies febio::damage#2's syntax half. Dmax = 1.0 is legal and runs;
1.2 and -0.1 are rejected with `Invalid value for parameter:` followed by
`.Dmax` — note the empty name, because the CDF property is unnamed, so
searching for a material name there finds nothing. The deck READS
successfully first.

NOT ASSERTED HERE: the second half of that pitfall, that a fully damaged
region can invert elements and that the failure is mesh-dependent. That
needs a converged two-mesh study and is left to a numerical fixture; this
one covers the range check and says so.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

def main() -> int:
    base = L.template("damage_3d_cycle")
    ok_run = L.run(L.swap(base, "<Dmax>0.9</Dmax>", "<Dmax>1.0</Dmax>"),
                   timeout=600)
    print(f"Dmax_1.0: rc={ok_run.rc} "
          f"normal={int(ok_run.normal_termination)} "
          f"steps={ok_run.steps_completed}")
    rejected = 0
    for value in ("1.2", "-0.1"):
        w = L.run(L.swap(base, "<Dmax>0.9</Dmax>", f"<Dmax>{value}</Dmax>"))
        msg = w.has("Invalid value for parameter:")
        empty_name = ".Dmax" in w.text and "Material1.Dmax" not in w.text
        init = w.has("Model initialization failed")
        print(f"Dmax_{value}: rc={w.rc} read_success={int(w.read_success)} "
              f"invalid_value={int(msg)} empty_name_before_dot="
              f"{int(empty_name)} init_failed={int(init)}")
        if msg and empty_name and init and w.read_success and w.rc != 0:
            rejected += 1
    print(f"out_of_range_rejected={rejected} of 2")
    good = (rejected == 2 and ok_run.rc == 0 and ok_run.normal_termination
            and not ok_run.has("Invalid value for parameter:"))
    return L.report(good, "dmax_range_check", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
