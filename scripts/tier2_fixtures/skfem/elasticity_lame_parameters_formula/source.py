"""Tier-2: lame_parameters(E, nu) returns (lam, mu) in that order.

Claim: skfem linear_elasticity#1 -- skfem.models.elasticity.lame_parameters(E,
nu) returns (lam, mu) with lam = E*nu/((1+nu)*(1-2*nu)) and mu = E/(2*(1+nu)),
matching to float64 precision.

Wrong variant: unpacking the result as (mu, lam). Nothing raises -- both are
positive floats of a similar magnitude -- so the mistake shows up only as a
wrong stiffness. This fixture measures the ratio at nu=0.3 to show the swap is
detectable, and checks the nu -> 0.5 incompressible limit where lam diverges
while mu stays finite (which is what identifies which slot is which).
"""
from __future__ import annotations

import math
import sys

from skfem.models.elasticity import lame_parameters


def main() -> int:
    ok = True
    E, nu = 210e9, 0.3
    lam, mu = lame_parameters(E, nu)
    lam_formula = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_formula = E / (2.0 * (1.0 + nu))
    print(f"lam_matches_formula={math.isclose(lam, lam_formula, rel_tol=1e-15)}")
    print(f"mu_matches_formula={math.isclose(mu, mu_formula, rel_tol=1e-15)}")
    if not (math.isclose(lam, lam_formula, rel_tol=1e-15)
            and math.isclose(mu, mu_formula, rel_tol=1e-15)):
        print(f"FAIL: lame_parameters({E}, {nu}) = ({lam!r}, {mu!r}) does not "
              f"match ({lam_formula!r}, {mu_formula!r})", file=sys.stderr)
        ok = False

    # --- WRONG variant: unpack as (mu, lam) -----------------------------
    order_right = lam > mu and not math.isclose(lam, mu, rel_tol=1e-3)
    print(f"returns_lam_then_mu_not_mu_then_lam={order_right}")
    print(f"lam_over_mu_gt_1p4={lam / mu > 1.4}")
    if not order_right:
        print(f"FAIL: at nu={nu} the two Lame constants are not separable by "
              f"magnitude ({lam!r} vs {mu!r}), so the swap would be invisible",
              file=sys.stderr)
        ok = False

    # nu -> 0.5: lam diverges, mu does not. This is the structural check
    # that identifies which return slot is lam.
    lam_a, mu_a = lame_parameters(E, 0.3)
    lam_b, mu_b = lame_parameters(E, 0.4999)
    blows_up = (lam_b / lam_a) > 100.0 and (mu_b / mu_a) < 2.0
    print(f"incompressible_limit_lam_blows_up={blows_up}")
    print(f"incompressible_limit_mu_stays_finite={(mu_b / mu_a) < 2.0}")
    if not blows_up:
        print(f"FAIL: nu -> 0.5 did not separate the two constants "
              f"(lam ratio {lam_b / lam_a:.3g}, mu ratio {mu_b / mu_a:.3g})",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
