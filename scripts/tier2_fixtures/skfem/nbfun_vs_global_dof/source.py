"""Tier-2: skfem Basis.Nbfun is per-element, not global.

Pitfall (skfem stokes#4): On a mixed-system DOF split, users
sometimes reach for basis.Nbfun thinking it's the global count;
it isn't. Nbfun is the number of shape functions per element
(3 for ElementTriP1, 6 for ElementTriP2, ...). The global count
is basis.N (or A.shape[0] after assembly).

Using Nbfun to slice a global vector silently produces wrong
slicing for any refined mesh.
"""
from __future__ import annotations

import sys

import skfem


def main() -> int:
    mesh = skfem.MeshTri().refined(2)
    p1 = skfem.Basis(mesh, skfem.ElementTriP1())
    p2 = skfem.Basis(mesh, skfem.ElementTriP2())
    print(f"P1_Nbfun={p1.Nbfun}, P1_N_global={p1.N}")
    print(f"P2_Nbfun={p2.Nbfun}, P2_N_global={p2.N}")
    if (p1.Nbfun == 3
            and p2.Nbfun == 6
            and p1.N != p1.Nbfun
            and p2.N != p2.Nbfun):
        print("global_differs: confirmed Nbfun != N for both bases")
        return 0
    print("ERROR: Nbfun coincided with N or had unexpected values",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
