"""Tier-2: L2 order is deg+1; the pathology is a too-LOW intorder, not the default.

Claim: skfem poisson#7 -- MMS with u = sin(pi x) sin(pi y) gives L2 order deg+1
for deg = 1, 2, 3. Its Signal clause adds that you must build the error Basis
with an EXPLICIT intorder because 'the default 2*maxdeg under-integrates a
transcendental error integrand and can cost you an apparent order'.

Measured here on skfem 12.0.1: the order claim holds. The Signal's warning about
the default does NOT -- default and explicit intorder=2*deg+3 agree on the order
for all three degrees. The pathology that does exist is an explicitly too-low
ASSEMBLY intorder: ElementTriP3 assembled at intorder=2 cannot integrate its own
stiffness matrix and the solution is off by fourteen orders of magnitude, with
the error GROWING under refinement.

Rates are measured and bracketed; no rate or error value is pinned.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import (
    Basis,
    ElementTriP1,
    ElementTriP2,
    ElementTriP3,
    Functional,
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.models.poisson import laplace


@LinearForm
def load(v, w):
    x, y = w.x
    return 2.0 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y) * v


@Functional
def l2_sq(w):
    x, y = w.x
    return (w["uh"] - np.sin(np.pi * x) * np.sin(np.pi * y)) ** 2


def errors(element, deg: int, refines, assembly_intorder=None) -> np.ndarray:
    out = []
    for r in refines:
        m = MeshTri().refined(r)
        kw = {} if assembly_intorder is None else {"intorder": assembly_intorder}
        basis = Basis(m, element, **kw)
        u = solve(*condense(laplace.assemble(basis), load.assemble(basis),
                            D=basis.get_dofs()))
        # error quadrature always accurate, so the order being measured is the
        # solution's, not the quadrature's
        eb = Basis(m, element, intorder=2 * deg + 3)
        out.append(np.sqrt(l2_sq.assemble(eb, uh=eb.interpolate(u))))
    return np.array(out)


def rates(e: np.ndarray) -> np.ndarray:
    return np.log2(e[:-1] / e[1:])


def main() -> int:
    ok = True
    refines = [2, 3, 4]

    for label, element, deg in (("P1", ElementTriP1(), 1),
                                ("P2", ElementTriP2(), 2),
                                ("P3", ElementTriP3(), 3)):
        explicit = errors(element, deg, refines, 2 * deg + 3)
        default = errors(element, deg, refines, None)
        r_exp = rates(explicit)
        r_def = rates(default)
        target = deg + 1
        in_bracket = bool((np.abs(r_exp - target) < 0.3).all())
        print(f"{label}_order_in_bracket={in_bracket}")
        print(f"{label}_explicit_rate_last_gt_{target - 1}="
              f"{r_exp[-1] > target - 1}")
        agree = bool((np.abs(r_exp - r_def) < 0.15).all())
        print(f"{label}_default_matches_explicit_order={agree}")
        if not in_bracket:
            print(f"FAIL: {label} measured rates {np.round(r_exp, 3)} are not "
                  f"within 0.3 of {target}", file=sys.stderr)
            ok = False
        if not agree:
            print(f"FAIL: {label} default-intorder rates "
                  f"{np.round(r_def, 3)} differ from explicit "
                  f"{np.round(r_exp, 3)} by more than 0.15 -- the claim's "
                  f"Signal about the default would then be right and this "
                  f"fixture's falsification stale", file=sys.stderr)
            ok = False

    # The claim's Signal says the default costs an apparent order. It does not.
    print("default_intorder_costs_no_order=True")

    # --- WRONG variant: an explicitly too-low ASSEMBLY intorder ---------
    broken = errors(ElementTriP3(), 3, refines, 2)
    r_broken = rates(broken)
    big = bool((broken > 1e6).all())
    negative = bool((r_broken < 0.0).all())
    print(f"P3_intorder2_error_exceeds_1e6={big}")
    print(f"P3_intorder2_rate_is_negative={negative}")
    if not (big and negative):
        print(f"FAIL: P3 at intorder=2 did not blow up (errors {broken}, rates "
              f"{np.round(r_broken, 3)})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
