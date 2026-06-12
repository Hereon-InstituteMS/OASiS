"""Tier-2: skfem ElementVector basis Nbfun + lame_parameters analytic.

Pitfalls (skfem linear_elasticity#0 + #1):

  #0 Vector elasticity needs ElementVector(ElementQuad1()) —
     wrapping the scalar element. Basis.Nbfun for the vector
     basis is 2× that of the scalar basis in 2D (4 → 8).
  #1 skfem.models.elasticity.lame_parameters(E, nu) returns
     (lam, mu) matching the analytic Lame formulas exactly.
"""
from __future__ import annotations

import math
import sys

import skfem
from skfem.models.elasticity import lame_parameters


def main() -> int:
    mesh = skfem.MeshQuad().refined(2)
    vec = skfem.Basis(mesh, skfem.ElementVector(skfem.ElementQuad1()))
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
