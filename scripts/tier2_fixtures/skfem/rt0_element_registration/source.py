"""Tier-2: skfem ElementTriRT0 registration + DOF count.

Pitfall (skfem mixed_poisson#1): The Raviart-Thomas family in
skfem is named ElementTriRT0, ElementTriRT1, etc. — NOT the
full 'ElementTriRaviartThomas' spelling. ElementTriRT0 has 3
DOFs per triangle (one normal-flux DOF per edge).
"""
from __future__ import annotations

import sys

import skfem


def main() -> int:
    rt0_present = hasattr(skfem, "ElementTriRT0")
    full_present = hasattr(skfem, "ElementTriRaviartThomas")
    rt0_nbfun = None
    if rt0_present:
        rt0_nbfun = skfem.Basis(
            skfem.MeshTri(), skfem.ElementTriRT0()).Nbfun
    print(f"RT0_present={rt0_present}")
    print(f"RT0_Nbfun={rt0_nbfun}")
    print(f"RaviartThomas_full_name_present={full_present}")
    if rt0_present and rt0_nbfun == 3 and not full_present:
        return 0
    print("ERROR: did not observe expected RT registration",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
