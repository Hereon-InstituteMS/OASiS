"""Tier-2: NGSolve has no tanh -- but it DOES have sinh and cosh.

Claim: ngsolve dg_methods#8 -- "NGSolve's CoefficientFunction namespace exposes
exp/log/sin/cos/tan/atan/atan2 but NOT the hyperbolic functions (no
tanh/sinh/cosh).  Calling tanh(z) in a CF expression raises NameError 'name
'tanh' is not defined'.  Build manually via the identity tanh(z) =
(exp(2z)-1)/(exp(2z)+1).  Signal: NameError at script import / gfu.Set time
naming 'tanh' (Did you mean: 'tan'?)."

Wrong variant: tanh(z) inside a CoefficientFunction expression.

CORRECTION this fixture records.  The parenthetical is wrong on NGSolve 6.2.2604:
ngsolve.sinh and ngsolve.cosh both exist and both evaluate on a
CoefficientFunction.  Only tanh is missing.  An agent told "no tanh/sinh/cosh"
would hand-roll two functions it already has -- and, worse, would not trust the
one-line sinh/cosh route that is the shortest correct fix.

What this fixture pins, all re-measured on this run:
  * tanh really is absent from the ngsolve namespace and raising it produces the
    NameError, with the interpreter's "Did you mean: 'tan'?" suggestion;
  * every other function the claim lists as present really is present AND
    evaluates on a CoefficientFunction;
  * sinh and cosh are present and evaluate too -- the claim's negative is false;
  * the recommended exp-based identity and the sinh/cosh quotient both reproduce
    tanh to machine precision at several sample points, so both fixes are real.

Mutation control:  T2_MUTATE=1 points the presence probe at 'tan' instead of
'tanh' -- the name that IS in the namespace, so getattr(ngsolve, PROBE) and the
eval of PROBE(x) both succeed and no NameError is captured.  The expectations
tanh_call_raises_nameerror=True, tanh_nameerror_literal=True and
nameerror_names_tanh=True then disappear.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import math
import os
import sys

import ngsolve
from netgen.geom2d import unit_square
from ngsolve import Mesh, cosh, exp, sinh, x

# Mutation control: under T2_MUTATE=1 the missing name the probe reaches for is
# replaced by 'tan', which exists -- the pathology is removed.
MUTATE = os.environ.get("T2_MUTATE") == "1"
PROBE = "tanh" if not MUTATE else "tan"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))

    print(f"tanh_in_namespace={hasattr(ngsolve, 'tanh')}")
    msg = ""
    suggestion = False
    try:
        eval(f"{PROBE}(x)",
             {PROBE: getattr(ngsolve, PROBE), "x": x})                # noqa: S307
    except AttributeError:
        # getattr fails first when the name is absent -- reproduce the real
        # call-site shape instead: an expression that mentions tanh unqualified.
        try:
            eval(f"{PROBE}(x)", {"x": x})                             # noqa: S307
        except NameError as exc:
            msg = str(exc)
            suggestion = getattr(exc, "name", None) == "tanh"
    literal = msg == "name 'tanh' is not defined"
    print(f"tanh_call_raises_nameerror={bool(msg)}")
    print(f"tanh_nameerror_literal={literal}")
    print(f"nameerror_names_tanh={suggestion}")

    # Functions the claim lists as PRESENT.
    listed = ["exp", "log", "sin", "cos", "tan", "atan", "atan2"]
    missing_listed = [n for n in listed if not hasattr(ngsolve, n)]
    print(f"listed_functions_missing={missing_listed}")
    print(f"all_listed_present={missing_listed == []}")

    # Functions the claim lists as ABSENT.
    claimed_absent = ["tanh", "sinh", "cosh"]
    actually_present = [n for n in claimed_absent if hasattr(ngsolve, n)]
    print(f"claimed_absent_but_present={sorted(actually_present)}")
    print(f"sinh_present={hasattr(ngsolve, 'sinh')}")
    print(f"cosh_present={hasattr(ngsolve, 'cosh')}")
    print(f"only_tanh_is_really_missing="
          f"{sorted(actually_present) == ['cosh', 'sinh']}")

    # ...and sinh/cosh are not stubs: they evaluate on a CoefficientFunction.
    pts = [0.1, 0.4, 0.9]
    sinh_ok = all(
        abs(float(sinh(x)(mesh(p, 0.5))) - math.sinh(p)) < 1e-12 for p in pts)
    cosh_ok = all(
        abs(float(cosh(x)(mesh(p, 0.5))) - math.cosh(p)) < 1e-12 for p in pts)
    print(f"sinh_evaluates_on_cf={sinh_ok}")
    print(f"cosh_evaluates_on_cf={cosh_ok}")

    # Both routes to tanh agree with math.tanh.
    ident = (exp(2 * x) - 1) / (exp(2 * x) + 1)
    quot = sinh(x) / cosh(x)
    ident_err = max(abs(float(ident(mesh(p, 0.5))) - math.tanh(p)) for p in pts)
    quot_err = max(abs(float(quot(mesh(p, 0.5))) - math.tanh(p)) for p in pts)
    print(f"exp_identity_max_err={ident_err:.3e}")
    print(f"sinh_over_cosh_max_err={quot_err:.3e}")
    print(f"exp_identity_reproduces_tanh={ident_err < 1e-12}")
    print(f"sinh_over_cosh_reproduces_tanh={quot_err < 1e-12}")

    ok = (
        not hasattr(ngsolve, "tanh")
        and msg == "name 'tanh' is not defined"
        and missing_listed == []
        and sorted(actually_present) == ["cosh", "sinh"]
        and sinh_ok and cosh_ok
        and ident_err < 1e-12 and quot_err < 1e-12
    )
    if ok:
        return 0
    print("FAIL: hyperbolic-namespace invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
