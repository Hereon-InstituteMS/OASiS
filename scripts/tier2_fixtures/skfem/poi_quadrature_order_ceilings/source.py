"""Tier-2: get_quadrature order ceilings, and three messages that look alike.

Claim: skfem poisson#6 -- skfem.quadrature.get_quadrature has a hard order
ceiling per reference domain: TRIANGLE <= 19, TETRAHEDRON <= 9. The triangle
ceiling message contains a typo ('quadratureis', no space); the tetrahedron one
is correctly spaced but says 'not available' rather than 'not implemented'; and
passing a string instead of a Refdom class raises a THIRD message that is easy to
mistake for the ceiling error.

Wrong variant: asking for an order above the ceiling (what a very-high-order tet
element needs, since it wants 2k > 9), and passing 'triangle' as a string.
"""
from __future__ import annotations

import sys

from skfem.quadrature import get_quadrature
from skfem.refdom import RefTet, RefTri


def main() -> int:
    ok = True

    # --- RIGHT variant: at the ceiling ----------------------------------
    X, W = get_quadrature(RefTri, 19)
    print(f"tri_order_19_ok=True")
    print(f"tri_npts_19={X.shape[1]}")
    print(f"tri_weight_sum_is_half={abs(W.sum() - 0.5) < 1e-12}")
    if abs(W.sum() - 0.5) > 1e-12:
        print(f"FAIL: triangle weights sum to {W.sum()!r}, expected 1/2",
              file=sys.stderr)
        ok = False

    Xt, Wt = get_quadrature(RefTet, 9)
    print(f"tet_order_9_ok=True")
    print(f"tet_npts_9={Xt.shape[1]}")
    print(f"tet_weight_sum_is_sixth={abs(Wt.sum() - 1.0 / 6.0) < 1e-12}")
    print(f"tet_ceiling_is_9_not_8=True")
    if abs(Wt.sum() - 1.0 / 6.0) > 1e-12:
        print(f"FAIL: tetrahedron weights sum to {Wt.sum()!r}, expected 1/6",
              file=sys.stderr)
        ok = False

    # --- WRONG variant (a): one order above each ceiling ----------------
    tri_msg = ""
    try:
        get_quadrature(RefTri, 20)
    except NotImplementedError as exc:
        tri_msg = str(exc)
    print(f"tri_order_20_msg={tri_msg!r}")
    if "quadratureis not implemented" not in tri_msg:
        print(f"FAIL: triangle ceiling message changed: {tri_msg!r}",
              file=sys.stderr)
        ok = False

    tet_msg = ""
    try:
        get_quadrature(RefTet, 10)
    except NotImplementedError as exc:
        tet_msg = str(exc)
    print(f"tet_order_10_msg={tet_msg!r}")
    if "of quadrature is not available" not in tet_msg:
        print(f"FAIL: tetrahedron ceiling message changed: {tet_msg!r}",
              file=sys.stderr)
        ok = False

    distinct = tri_msg != tet_msg
    print(f"two_ceiling_messages_differ={distinct}")
    if not distinct:
        print("FAIL: the two ceiling messages are identical, so the claim's "
              "distinction is gone", file=sys.stderr)
        ok = False

    # --- WRONG variant (b): a string instead of a Refdom class ----------
    str_msg = ""
    try:
        get_quadrature("triangle", 2)
    except NotImplementedError as exc:
        str_msg = str(exc)
    print(f"string_domain_msg={str_msg!r}")
    if "is not supported" not in str_msg:
        print(f"FAIL: the string-domain message changed: {str_msg!r}",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
