"""Tier-2: <solver type="thermo-fluid"/> alone inherits skyline and aborts.

Verifies febio::heat#3. The thermo-fluid stiffness matrix is
non-symmetric and the default solver on a USE_MKL=OFF build cannot store
it, so the <linear_solver> child is not optional.

The failure arrives AFTER the reader says SUCCESS, and the message is
inside the 71-column ERROR box, so the fixture matches the short
fragment `does not support the requested` rather than the whole
sentence — see linear_elasticity#9 for why.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

SOLVER_OK = ('    <solver type="thermo-fluid">\n'
             '      <linear_solver type="bicgstab"/>\n'
             "    </solver>\n")
SOLVER_BARE = '    <solver type="thermo-fluid"/>\n'


def main() -> int:
    right = L.template("heat_3d_bar", n=2)
    wrong = L.swap(right, SOLVER_OK, SOLVER_BARE)
    return L.init_error("thermofluid_needs_bicgstab",
                        wrong=wrong, right=right,
                        message="does not support the requested",
                        also=("Please select a different linear solver.",))


if __name__ == "__main__":
    sys.exit(main())
