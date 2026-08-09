"""Tier-2: skfem ElementTriRT0 registration + DOF count.

Pitfall (skfem mixed_poisson#1): The Raviart-Thomas family in
skfem is named ElementTriRT0, ElementTriRT1, etc. — NOT the
full 'ElementTriRaviartThomas' spelling. ElementTriRT0 has 3
DOFs per triangle (one normal-flux DOF per edge).

Mutation control: T2_MUTATE=1 changes the name looked up in
the second probe from the long spelling
'ElementTriRaviartThomas' to the abbreviated 'ElementTriRT0'
-- the documented fix, applied at the class-name lookup the
pitfall is about. That name resolves, so the expectation
"RaviartThomas_full_name_present=False" disappears from the
output (it prints True) and the fixture goes red. Re-run:
T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import skfem

MUTATE = os.environ.get("T2_MUTATE") == "1"

# THE PATHOLOGY: reaching for the full Raviart-Thomas spelling.
# The documented fix is the abbreviated ElementTriRT0 name.
FULL_NAME = ("ElementTriRaviartThomas" if not MUTATE
             else "ElementTriRT0")


def main() -> int:
    rt0_present = hasattr(skfem, "ElementTriRT0")
    full_present = hasattr(skfem, FULL_NAME)
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
