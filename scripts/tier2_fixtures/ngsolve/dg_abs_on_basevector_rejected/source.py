"""Tier-2: Python's abs() is not defined on ngsolve.la.BaseVector, and the
numpy route is the one that works.

Claim: ngsolve dg_methods#7 -- "Python's builtin abs() is NOT defined on
ngsolve.la.BaseVector.  max(abs(gfu.vec)) raises TypeError 'bad operand type for
abs(): ngsolve.la.BaseVector'.  Convert to numpy via gfu.vec.FV().NumPy() then
reduce: float(numpy.abs(gfu.vec.FV().NumPy()).max()).  Same pattern applies to
compound spaces -- gfu.components[i].vec.FV().NumPy()."

Wrong variant: abs(gfu.vec), and max(abs(gfu.vec)).

What this fixture pins, all re-measured on this run:
  * the runtime class of a GridFunction's .vec is ngsolve.la.BaseVector, so the
    quoted message names a class that really is the one in play;
  * abs() on it raises TypeError with the literal text, and so does the
    max(abs(...)) spelling the claim gives;
  * the FV().NumPy() route succeeds and returns a finite number that agrees with
    an independent reduction over the same coefficients;
  * the compound-space spelling gfu.components[i].vec.FV().NumPy() works too,
    and the per-component maxima bound the whole-vector maximum -- checked on a
    real L2 * L2 product space rather than asserted.

Mutation control:  T2_MUTATE=1 applies the documented fix at the pathology site
-- abs(gfu.vec) becomes abs(gfu.vec.FV().NumPy()), i.e. the operand handed to
abs() is the numpy view instead of the BaseVector.  abs() then succeeds, no
TypeError is captured, and the expectations abs_raises_typeerror=True and
abs_message_literal=True are gone.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy
from netgen.geom2d import unit_square
from ngsolve import GridFunction, L2, Mesh, x, y


LITERAL = "bad operand type for abs(): 'ngsolve.la.BaseVector'"

# Mutation control: under T2_MUTATE=1 the operand of abs() is the documented
# numpy view instead of the raw BaseVector -- the pathology itself, removed.
MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = L2(mesh, order=2)
    gfu = GridFunction(fes)
    gfu.Set(x * x - 0.5 * y)

    cls = f"{type(gfu.vec).__module__}.{type(gfu.vec).__name__}"
    print(f"vec_class={cls}")
    print(f"vec_class_is_basevector={cls == 'ngsolve.la.BaseVector'}")

    m1 = ""
    operand = gfu.vec if not MUTATE else gfu.vec.FV().NumPy()
    try:
        abs(operand)
    except TypeError as exc:
        m1 = str(exc)
    print(f"abs_raises_typeerror={bool(m1)}")
    print(f"abs_message_literal={LITERAL in m1}")

    m2 = ""
    try:
        max(abs(gfu.vec))
    except TypeError as exc:
        m2 = str(exc)
    print(f"max_abs_spelling_also_raises={bool(m2)}")
    print(f"max_abs_message_literal={LITERAL in m2}")

    # The documented route.
    arr = gfu.vec.FV().NumPy()
    peak = float(numpy.abs(arr).max())
    independent = max(abs(float(c)) for c in arr)
    print(f"numpy_route_peak={peak:.10f}")
    print(f"numpy_route_finite={numpy.isfinite(peak)}")
    print(f"numpy_route_agrees_with_python_reduction="
          f"{abs(peak - independent) < 1e-13}")
    print(f"numpy_route_length_is_ndof={arr.size == fes.ndof}")

    # Compound spaces take the same route through .components[i].
    comp = fes * fes
    gfc = GridFunction(comp)
    gfc.components[0].Set(x)
    gfc.components[1].Set(-3.0 * y)
    p0 = float(numpy.abs(gfc.components[0].vec.FV().NumPy()).max())
    p1 = float(numpy.abs(gfc.components[1].vec.FV().NumPy()).max())
    pall = float(numpy.abs(gfc.vec.FV().NumPy()).max())
    print(f"component_peaks={p0:.10f},{p1:.10f}")
    print(f"compound_route_works={numpy.isfinite(p0) and numpy.isfinite(p1)}")
    print(f"component_peaks_bound_whole_vector="
          f"{abs(max(p0, p1) - pall) < 1e-12}")

    ok = (
        cls == "ngsolve.la.BaseVector"
        and LITERAL in m1
        and LITERAL in m2
        and numpy.isfinite(peak)
        and abs(peak - independent) < 1e-13
        and arr.size == fes.ndof
        and abs(max(p0, p1) - pall) < 1e-12
    )
    if ok:
        return 0
    print("FAIL: BaseVector abs() invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
