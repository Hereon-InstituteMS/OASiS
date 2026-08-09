"""Tier-2: skfem Basis.Nbfun is per-element, not global.

Pitfall (skfem stokes#4): On a mixed-system DOF split, users
sometimes reach for basis.Nbfun thinking it's the global count;
it isn't. Nbfun is the number of shape functions per element
(3 for ElementTriP1, 6 for ElementTriP2, ...). The global count
is basis.N (or A.shape[0] after assembly).

Using Nbfun to slice a global vector silently produces wrong
slicing for any refined mesh.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology site
-- the two `basis.Nbfun` reads become `basis.N`, the global count a user
actually wants.  The per-element values are then gone from the output, so
P1_Nbfun=3, P2_Nbfun=6 and (since N == N) global_differs all disappear.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import skfem

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = skfem.MeshTri().refined(2)
    p1 = skfem.Basis(mesh, skfem.ElementTriP1())
    p2 = skfem.Basis(mesh, skfem.ElementTriP2())
    # PATHOLOGY: reaching for Nbfun where the global count is meant.
    # T2_MUTATE=1 applies the documented fix -- read basis.N instead.
    p1_count = p1.N if MUTATE else p1.Nbfun
    p2_count = p2.N if MUTATE else p2.Nbfun
    print(f"P1_Nbfun={p1_count}, P1_N_global={p1.N}")
    print(f"P2_Nbfun={p2_count}, P2_N_global={p2.N}")
    if (p1_count == 3
            and p2_count == 6
            and p1.N != p1_count
            and p2.N != p2_count):
        print("global_differs: confirmed Nbfun != N for both bases")
        return 0
    print("ERROR: Nbfun coincided with N or had unexpected values",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
