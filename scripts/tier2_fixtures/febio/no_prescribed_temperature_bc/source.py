"""Tier-2: the BC is `prescribed fluid temperature`, not
`prescribed temperature`.

Verifies febio::heat#2. Every temperature BC in FEBio 4.12 is scoped to
the thermo-fluid module and acts on a FLUID temperature DOF, so the
obvious name is unregistered. Rejected at parse by attribute value.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

def main() -> int:
    right = L.template("heat_3d_bar", n=2)
    wrong = L.swap(right, 'type="prescribed fluid temperature"',
                   'type="prescribed temperature"')
    return L.parse_error("prescribed_temperature_bc",
                         wrong=wrong, right=right,
                         message='tag "bc"',
                         also=('invalid value for attribute "type"',))


if __name__ == "__main__":
    sys.exit(main())
