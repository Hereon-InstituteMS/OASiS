"""Tier-2: a CoefficientFunction has no rich comparison, so `gap < 0` dies at
the expression, not at assembly -- IfPos is the active-set indicator.

Claim: ngsolve contact#2 -- "IfPos(-gap, 1, 0) identifies active contact nodes
-- evaluates at integration points.  Signal: the Python comparison `gap < 0`
fails IMMEDIATELY at expression construction (not at form assembly, as the prior
text said) with the literal TypeError(\"'<' not supported between instances of
'ngsolve.fem.CoefficientFunction' and 'int'\") -- NGSolve simply does not define
rich comparison on CoefficientFunction, so there is no 'CoefficientFunction
comparison' message to grep for."

Wrong variant: `gap < 0` (and the other three orderings) in place of IfPos.

What this fixture pins, all re-measured on this run:
  * the literal TypeError text, and that it surfaces before any BilinearForm
    exists -- checked by having no form in scope at that point;
  * that the wording the claim says is absent really is absent from the message;
  * that <, <=, > and >= all fail the same way, so no ordering happens to work;
  * that == does NOT raise -- it is defined and returns a plain Python bool, so
    an agent testing "does comparison work?" with == gets a misleading yes;
  * that IfPos(-gap, 1, 0) is a working indicator: its integral over the region
    where gap < 0 equals that region's area computed independently, and the
    complementary indicator sums with it to the full domain area.
"""
from __future__ import annotations

import sys

from netgen.geom2d import unit_square
from ngsolve import CoefficientFunction, IfPos, Integrate, Mesh, y


LITERAL = ("'<' not supported between instances of "
           "'ngsolve.fem.CoefficientFunction' and 'int'")
ABSENT = "CoefficientFunction comparison"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))
    # Signed gap: negative (penetrating) below y = 0.3, positive above.
    y0 = 0.3
    gap = y - y0
    print(f"gap_is_coefficientfunction="
          f"{isinstance(gap, CoefficientFunction)}")

    msgs = {}
    for label, fn in (("lt", lambda: gap < 0),
                      ("le", lambda: gap <= 0),
                      ("gt", lambda: gap > 0),
                      ("ge", lambda: gap >= 0)):
        try:
            fn()
            msgs[label] = ""
        except TypeError as exc:
            msgs[label] = str(exc)
        print(f"{label}_raises={bool(msgs[label])}")

    print(f"lt_message_literal={LITERAL in msgs['lt']}")
    print(f"claimed_absent_wording_really_absent="
          f"{all(ABSENT not in m for m in msgs.values())}")
    print(f"all_four_orderings_raise="
          f"{all(bool(m) for m in msgs.values())}")

    # == is defined and gives a bool -- the misleading control.
    eq = (gap == 0)
    print(f"eq_returns_without_raising=True")
    print(f"eq_returns_plain_bool={isinstance(eq, bool)}")

    # The working indicator.
    active = IfPos(-gap, 1.0, 0.0)
    inactive = IfPos(gap, 1.0, 0.0)
    a_area = float(Integrate(active, mesh))
    i_area = float(Integrate(inactive, mesh))
    print(f"active_indicator_area={a_area:.6f}")
    print(f"expected_active_area={y0:.6f}")
    print(f"ifpos_indicator_matches_geometry={abs(a_area - y0) < 5e-3}")
    print(f"indicators_partition_the_domain={abs(a_area + i_area - 1.0) < 1e-9}")

    ok = (
        isinstance(gap, CoefficientFunction)
        and LITERAL in msgs["lt"]
        and all(bool(m) for m in msgs.values())
        and all(ABSENT not in m for m in msgs.values())
        and isinstance(eq, bool)
        and abs(a_area - y0) < 5e-3
        and abs(a_area + i_area - 1.0) < 1e-9
    )
    if ok:
        return 0
    print("FAIL: CF comparison / IfPos indicator invariant not held",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
