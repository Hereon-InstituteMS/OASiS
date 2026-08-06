"""Tier-2: `fluid-FSI` is both a module and a material, and the material
only works inside the module.

Verifies febio::fluid_fsi#0. Naming <Module type="fluid"/> with a
`fluid-FSI` material fails on the SOLVER first, before the materials are
read — so the diagnostic points at <Control>.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

def main() -> int:
    right = L.template("fluid_fsi_3d_block")
    wrong = L.swap(right, '<Module type="fluid-FSI"/>',
                   '<Module type="fluid"/>')
    return L.parse_error("fluid_fsi_module", wrong=wrong, right=right,
                         message='tag "solver"',
                         also=('invalid value for attribute "type"',))


if __name__ == "__main__":
    sys.exit(main())
