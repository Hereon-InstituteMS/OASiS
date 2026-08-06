"""Tier-2: a mis-shaped body force is rejected -- but by two different messages,
neither of which is the one the claim quotes, and both at __iadd__.

Claim: ngsolve linear_elasticity#1 -- "Body forces are constructed with
CoefficientFunction taking a Python tuple shaped to match the vector FESpace:
CoefficientFunction((fx, fy)) for 2D, CoefficientFunction((fx, fy, fz)) for 3D.
A mismatch (e.g. scalar fx for a VectorH1 space) raises a shape mismatch from the
assembly routine.  Signal: BilinearForm/LinearForm.Assemble() raises with
'dimensions do not match' or similar from the C++ kernel."

Wrong variant: a scalar source, and a 3-component source, on a 2D VectorH1 space.

THREE CORRECTIONS this fixture records.

  (a) The failure point.  Both mismatches raise from LinearForm.__iadd__, not
      from .Assemble() -- so a guard wrapped around Assemble as the Signal
      instructs never sees either of them.
  (b) The message for the claim's OWN example.  A scalar fx against a vector
      test function does not produce a dimension message at all: the product is
      vector-valued, and NGSolve says 'SymbolicLFI needs scalar-valued
      CoefficientFunction' -- the same string mhd#7 is about.  Grepping for a
      dimension mismatch on the claim's own example finds nothing.
  (c) The wording.  The 3-component case DOES give a dimension message, but the
      literal text is "Dimensions don't match", contracted, not "dimensions do
      not match".  A substring guard written from the claim misses it.

What this fixture pins, all re-measured on this run:
  * the correctly shaped 2-tuple assembles and gives a non-zero load vector;
  * the scalar case raises with the SymbolicLFI scalar-valued message and NOT
    with any dimension wording;
  * the 3-tuple case raises with the literal "Dimensions don't match" and
    reports both operand dimensions, 3 and 2;
  * the claim's own phrasing 'dimensions do not match' appears in neither
    message;
  * both raise at __iadd__ and neither reaches Assemble;
  * the 3D analogue works: a 3-tuple on a 3D VectorH1 assembles, and the 2-tuple
    that was correct in 2D is the mismatch there -- so the rule really is
    "match the space", checked in both dimensions.
"""
from __future__ import annotations

import sys

import numpy
from netgen.csg import unit_cube
from netgen.geom2d import unit_square
from ngsolve import CoefficientFunction, LinearForm, Mesh, VectorH1, dx

SCALAR_MSG = "SymbolicLFI needs scalar-valued CoefficientFunction"
DIM_MSG = "Dimensions don't match"
CLAIMED = "dimensions do not match"


def attempt(fes, cf):
    _, v = fes.TnT()
    f = LinearForm(fes)
    stage = "form_constructed"
    msg = ""
    try:
        f += cf * v * dx
        stage = "iadd_accepted"
        f.Assemble()
        stage = "assembled"
    except Exception as exc:                                   # noqa: BLE001
        msg = str(exc)
    peak = (float(numpy.abs(f.vec.FV().NumPy()).max())
            if stage == "assembled" else float("nan"))
    return stage, msg, peak


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = VectorH1(mesh, order=1)

    st_ok, msg_ok, peak_ok = attempt(fes, CoefficientFunction((1.0, 2.0)))
    print(f"correct_2tuple_stage={st_ok} peak={peak_ok:.6f}")
    print(f"correct_2tuple_assembles={st_ok == 'assembled'}")
    print(f"correct_2tuple_load_is_nonzero={peak_ok > 1e-12}")

    st_s, msg_s, _ = attempt(fes, CoefficientFunction(1.0))
    print(f"scalar_stage={st_s}")
    print(f"scalar_raised={bool(msg_s)}")
    print(f"scalar_message={msg_s!r}")
    print(f"scalar_gives_the_symboliclfi_message={SCALAR_MSG in msg_s}")
    print(f"scalar_gives_no_dimension_wording="
          f"{DIM_MSG not in msg_s and CLAIMED not in msg_s.lower()}")
    print(f"scalar_raised_at_iadd={st_s == 'form_constructed'}")

    st_3, msg_3, _ = attempt(fes, CoefficientFunction((1.0, 2.0, 3.0)))
    print(f"three_tuple_stage={st_3}")
    print(f"three_tuple_raised={bool(msg_3)}")
    print(f"three_tuple_message={msg_3!r}")
    print(f"three_tuple_literal_is_contracted={DIM_MSG in msg_3}")
    print(f"three_tuple_reports_both_dims="
          f"{'3' in msg_3 and '2' in msg_3}")
    print(f"three_tuple_raised_at_iadd={st_3 == 'form_constructed'}")

    print(f"claimed_wording_in_neither_message="
          f"{CLAIMED not in msg_s.lower() and CLAIMED not in msg_3.lower()}")
    print(f"neither_reached_assemble="
          f"{st_s != 'assembled' and st_3 != 'assembled'}")

    # 3D: the rule is "match the space", so the roles swap.
    mesh3 = Mesh(unit_cube.GenerateMesh(maxh=0.6))
    fes3 = VectorH1(mesh3, order=1)
    st3_ok, _, peak3 = attempt(fes3, CoefficientFunction((1.0, 2.0, 3.0)))
    st3_bad, msg3_bad, _ = attempt(fes3, CoefficientFunction((1.0, 2.0)))
    print(f"3d_three_tuple_stage={st3_ok} peak={peak3:.6f}")
    print(f"3d_three_tuple_assembles={st3_ok == 'assembled'}")
    print(f"3d_two_tuple_stage={st3_bad}")
    print(f"3d_two_tuple_raised={bool(msg3_bad)}")
    print(f"3d_two_tuple_gives_dimension_message={DIM_MSG in msg3_bad}")
    rule = (st_ok == "assembled" and st3_ok == "assembled"
            and st_3 == "form_constructed"
            and st3_bad == "form_constructed")
    print(f"rule_is_match_the_space={rule}")

    ok = (
        st_ok == "assembled" and peak_ok > 1e-12
        and SCALAR_MSG in msg_s and DIM_MSG not in msg_s
        and st_s == "form_constructed"
        and DIM_MSG in msg_3 and st_3 == "form_constructed"
        and CLAIMED not in msg_s.lower() and CLAIMED not in msg_3.lower()
        and st3_ok == "assembled" and st3_bad == "form_constructed"
        and DIM_MSG in msg3_bad
    )
    if ok:
        return 0
    print("FAIL: body-force shape invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
