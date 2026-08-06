"""Tier-2: there is no <micro_viscosity> parameter on `polar fluid`.

Verifies febio::polar_fluid#1. The polar viscosity is a PROPERTY —
<polar type="polar linear"> with tau/alpha/beta/gamma — so the parameter
name that the literature suggests is simply not a tag.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

POLAR = ('      <polar type="polar linear">\n'
         "        <tau>0.001</tau>\n"
         "        <alpha>0.0</alpha>\n"
         "        <beta>0.001</beta>\n"
         "        <gamma>0.001</gamma>\n"
         "      </polar>\n")


def main() -> int:
    right = L.template("polar_fluid_3d_channel")
    wrong = L.swap(right, POLAR,
                   "      <micro_viscosity>0.001</micro_viscosity>\n")
    return L.parse_error("micro_viscosity_tag", wrong=wrong, right=right,
                         message='tag "micro_viscosity"',
                         also=("unrecognized tag",))


if __name__ == "__main__":
    sys.exit(main())
