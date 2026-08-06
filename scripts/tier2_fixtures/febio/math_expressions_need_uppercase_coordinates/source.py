"""Tier-2: lowercase x/y/z in a math expression SEGFAULTS after a clean read.

Verifies febio::elasticity_mms#2. FEBio's 4.x math parser uses UPPERCASE
X, Y, Z for material coordinates. A body-load expression written with
lowercase symbols is NOT a parse error: the deck reads `...SUCCESS!`, the
first time step starts, and the process is killed during stiffness
assembly. No diagnostic names the bad symbol.

The fixture asserts the read succeeded and the signal arrived — the
combination is what makes "treat any early-timestep segfault in a deck
with math expressions as a symbol-vocabulary suspect" actionable.

MUTATION CONTROL. T2_MUTATE=1 leaves the body-load expression in
UPPERCASE coordinates in the "wrong" slot — the pathology removed.
There is no segfault, and 'lowercase_coordinate_symbols=reproduced' is
no longer printed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

def main() -> int:
    right = L.template("elasticity_mms_3d_cube_hex8", n=2)
    i, j = right.find("<Loads>"), right.find("</Loads>")
    if i < 0 or j < 0:
        L.die("the MMS template no longer has a <Loads> section, so the "
              "expression this fixture mutates is not there")
    loads = right[i:j]
    lowered = (loads.replace("*X)", "*x)").replace("*Y)", "*y)")
               .replace("*Z)", "*z)"))
    if lowered == loads:
        L.die("no uppercase coordinate symbols found in the body load — "
              "nothing was triggered")
    if MUTATE:
        print("mutation=the_wrong_slot_keeps_uppercase_coordinates")
        wrong = right
    else:
        wrong = right[:i] + lowered + right[j:]

    w = L.run(wrong, timeout=600)
    r = L.run(right, timeout=600)
    print(f"lowercase: rc={w.rc} sigsegv={int(w.segfault)} "
          f"read_success={int(w.read_success)} "
          f"read_failed={int(w.read_failed)} "
          f"names_a_symbol={int('x' in w.text and 'invalid' in w.text)} "
          f"error_box={int('ERROR' in w.out)}")
    print(f"uppercase: rc={r.rc} read_success={int(r.read_success)} "
          f"normal={int(r.normal_termination)} steps={r.steps_completed}")
    good = (w.segfault and w.read_success and not w.read_failed
            and "ERROR" not in w.out
            and r.rc == 0 and r.normal_termination)
    if not good:
        print(w.out[:900])
    return L.report(good, "lowercase_coordinate_symbols", "reproduced",
                    "not_reproduced")


if __name__ == "__main__":
    sys.exit(main())
