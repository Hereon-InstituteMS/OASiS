"""Tier-2: Python's abs() is not defined on a CoefficientFunction expression
either -- IfPos or sqrt(z*z) is the symbolic absolute value.

Claim: ngsolve dg_methods#9 -- "Python abs() also does NOT work on a NGSolve
CoefficientFunction expression.  Use IfPos(z, z, -z) (or sqrt(z*z)) for symbolic
absolute value.  Signal: TypeError with the literal text 'bad operand type for
abs(): ngsolve.fem.CoefficientFunction' raised from a Python-level abs() applied
to a Trace, InnerProduct, or other CF-valued expression."

Wrong variant: abs(expr) where expr is any CoefficientFunction.

What this fixture pins, all re-measured on this run:
  * the literal message, raised from all three expression shapes the claim
    names -- a bare CF, an InnerProduct, and a Trace;
  * the reported class in the message is ngsolve.fem.CoefficientFunction even
    though the operands came out of InnerProduct/Trace, so the text is stable
    across call sites;
  * IfPos(z, z, -z) and sqrt(z*z) both give the true absolute value: they agree
    with each other and with math.fabs at sample points straddling the sign
    change, and their integral over the unit square matches an independent
    two-piece integration of the positive and negative parts.
"""
from __future__ import annotations

import math
import sys

from netgen.geom2d import unit_square
from ngsolve import (
    CoefficientFunction,
    IfPos,
    InnerProduct,
    Integrate,
    Mesh,
    Trace,
    grad,
    sqrt,
    x,
    y,
)
from ngsolve import GridFunction, H1


LITERAL = "bad operand type for abs(): 'ngsolve.fem.CoefficientFunction'"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.25))
    fes = H1(mesh, order=2)
    gfu = GridFunction(fes)
    gfu.Set(x * y)

    z = x - 0.5                                    # bare CF, changes sign
    ip = InnerProduct(grad(gfu), grad(gfu))        # InnerProduct-valued
    tr = Trace(CoefficientFunction((x, y, y, x), dims=(2, 2)))  # Trace-valued

    msgs = {}
    for label, expr in (("bare_cf", z), ("innerproduct", ip), ("trace", tr)):
        try:
            abs(expr)
            msgs[label] = ""
        except TypeError as exc:
            msgs[label] = str(exc)
        print(f"{label}_raises={bool(msgs[label])}")
        print(f"{label}_message_literal={LITERAL in msgs[label]}")

    all_literal = all(LITERAL in m for m in msgs.values())
    print(f"all_three_shapes_give_same_literal={all_literal}")

    # The two documented replacements.
    a_ifpos = IfPos(z, z, -z)
    a_sqrt = sqrt(z * z)
    pts = [0.05, 0.3, 0.5, 0.7, 0.95]
    err_ifpos = max(
        abs(float(a_ifpos(mesh(p, 0.5))) - math.fabs(p - 0.5)) for p in pts)
    err_sqrt = max(
        abs(float(a_sqrt(mesh(p, 0.5))) - math.fabs(p - 0.5)) for p in pts)
    print(f"ifpos_max_err={err_ifpos:.3e}")
    print(f"sqrt_max_err={err_sqrt:.3e}")
    print(f"ifpos_is_absolute_value={err_ifpos < 1e-12}")
    print(f"sqrt_is_absolute_value={err_sqrt < 1e-12}")

    # ...and they integrate to the same thing as a hand-split two-piece integral.
    i_ifpos = float(Integrate(a_ifpos, mesh))
    i_sqrt = float(Integrate(a_sqrt, mesh))
    i_split = float(Integrate(IfPos(z, z, 0), mesh)) \
        + float(Integrate(IfPos(-z, -z, 0), mesh))
    print(f"integral_ifpos={i_ifpos:.10f}")
    print(f"integral_sqrt={i_sqrt:.10f}")
    print(f"integral_hand_split={i_split:.10f}")
    agree = abs(i_ifpos - i_split) < 1e-10 and abs(i_sqrt - i_split) < 1e-10
    print(f"both_replacements_agree_with_split_integral={agree}")

    ok = all_literal and err_ifpos < 1e-12 and err_sqrt < 1e-12 and agree
    if ok:
        return 0
    print("FAIL: CoefficientFunction abs() invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
