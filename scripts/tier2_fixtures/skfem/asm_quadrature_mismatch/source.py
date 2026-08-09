"""Tier-2: skfem asm() raises Quadrature mismatch on uneven intorder.

Pitfall (skfem stokes#6): When using asm() with two different
bases (e.g. P2 velocity + P1 pressure), the integration order
on both must match. If one is built with intorder=2 and the
other with intorder=6, asm() raises:

  ValueError: Quadrature mismatch: trial and test functions
  should have same number of integration points.

The fix is to set intorder=N on BOTH bases with the same N
(typically N >= max(p_trial, p_test) * 2).

Mutation control: T2_MUTATE=1 gives the P1 basis the SAME intorder as the
P2 one, which is the documented fix. asm() then returns a matrix instead of
raising, and the three expectations "ValueError", "Quadrature mismatch" and
"same number of integration points" all disappear from the output.
"""
from __future__ import annotations

import os
import sys
import traceback

import skfem
from skfem.helpers import inner

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = skfem.MeshTri().refined(2)
    p2 = skfem.Basis(mesh, skfem.ElementTriP2(), intorder=2)
    # the pathology: a DIFFERENT intorder on the second basis
    p1 = skfem.Basis(mesh, skfem.ElementTriP1(),
                     intorder=2 if MUTATE else 6)

    @skfem.BilinearForm
    def b_form(u, v, w):
        return inner(u, v)

    try:
        skfem.asm(b_form, p2, p1)
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: asm() succeeded across mismatched intorder "
          "(catalog claim wrong)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
