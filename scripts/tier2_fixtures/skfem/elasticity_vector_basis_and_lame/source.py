"""Tier-2: skfem ElementVector basis Nbfun + lame_parameters analytic.

Pitfalls (skfem linear_elasticity#0 + #1):

  #0 Vector elasticity needs ElementVector(ElementQuad1()) —
     wrapping the scalar element. Basis.Nbfun for the vector
     basis is 2× that of the scalar basis in 2D (4 → 8).
  #1 skfem.models.elasticity.lame_parameters(E, nu) returns
     (lam, mu) matching the analytic Lame formulas exactly.

Mutation control (INVERTED direction): this fixture executes
no pathology -- it only checks the correct spellings -- so
there is nothing to remove.  ``T2_MUTATE=1 python source.py``
instead ACTIVATES the documented mistake of pitfall #0 at the
site it is about: the "vector" basis is built from the bare
scalar ``ElementQuad1()`` without the ``ElementVector(...)``
wrapper.  ``vector_Nbfun`` then prints 4 instead of 8, so the
expectation ``vector_Nbfun=8`` disappears and the fixture goes
red.  (``scalar_Nbfun=4`` and ``lame_match=True`` are
pathology-independent and survive by design.)
"""
from __future__ import annotations

import math
import os
import sys

import skfem
from skfem.models.elasticity import lame_parameters

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = skfem.MeshQuad().refined(2)
    # Under T2_MUTATE the ElementVector wrapper is dropped.
    vec_elem = (skfem.ElementVector(skfem.ElementQuad1()) if not MUTATE
                else skfem.ElementQuad1())
    vec = skfem.Basis(mesh, vec_elem)
    sca = skfem.Basis(mesh, skfem.ElementQuad1())
    print(f"vector_Nbfun={vec.Nbfun}")
    print(f"scalar_Nbfun={sca.Nbfun}")
    # Lame parameters
    E, nu = 210e9, 0.3
    lam, mu = lame_parameters(E, nu)
    lam_ref = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu_ref = E / (2 * (1 + nu))
    match = math.isclose(lam, lam_ref) and math.isclose(mu, mu_ref)
    print(f"lame_match={match}")
    if vec.Nbfun == 8 and sca.Nbfun == 4 and match:
        return 0
    print("ERROR: empirical check did not match catalog claims",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
