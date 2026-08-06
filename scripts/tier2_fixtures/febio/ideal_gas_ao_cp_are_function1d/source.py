"""Tier-2: <ao> and <cp> on `ideal gas` are FEFunction1D properties.

Verifies febio::heat#7. Each needs a type= attribute; a plain number is
rejected as an invalid attribute value, not as a bad number, which is the
part that misleads — the tag looks like a scalar parameter.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

def main() -> int:
    right = L.template("heat_3d_bar", n=2)
    wrong = L.swap(right, '<cp type="const"><value>3.5</value></cp>',
                   "<cp>3.5</cp>")
    return L.parse_error("cp_needs_type_attr", wrong=wrong, right=right,
                         message='tag "cp"',
                         also=('invalid value for attribute "type"',))


if __name__ == "__main__":
    sys.exit(main())
