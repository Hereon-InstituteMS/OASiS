"""Tier-2: Poisson's ratio is <v>; <nu> is an unrecognized tag.

Verifies febio::linear_elasticity#5. nu is the spelling used across
FEniCSx / deal.II / NGSolve, and it is the single most likely
cross-backend transliteration error. FEBio names the offending tag back
verbatim, so it is cheap to diagnose once you know to read the tag name.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import _febio_lib as L  # noqa: E402

MAT_NU = ("  <Material>\n"
          '    <material id="1" name="Material1" type="isotropic elastic">\n'
          "      <density>1.0</density><E>1000.0</E><nu>0.3</nu>\n"
          "    </material>\n"
          "  </Material>")


def main() -> int:
    return L.parse_error(
        "nu_instead_of_v",
        wrong=L.solid_deck(material=MAT_NU),
        right=L.solid_deck(),
        message='tag "nu"',
        also=("unrecognized tag",))


if __name__ == "__main__":
    sys.exit(main())
