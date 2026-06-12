"""Tier-2: NGSolve pml.Radial REQUIRES origin positional arg.

Catalog under audit (src/backends/ngsolve/generators/
helmholtz.py L24): used
  pml.Radial(rad=0.7, alpha=2j)
which raises TypeError because the pybind11 signature is
  pml.Radial(origin, rad=1, alpha=1j).

Correct call:
  pml.Radial(origin=(0, 0), rad=0.7, alpha=2j)

This fixture asserts:
  * pml.Radial WITHOUT origin raises TypeError mentioning
    'incompatible function arguments'.
  * pml.Radial WITH origin=(0, 0) succeeds.
  * The fixed helmholtz template's exact call signature
    (origin=(0, 0), rad=0.7, alpha=2j) is constructable.
"""
from __future__ import annotations

import sys

from ngsolve import pml


def main() -> int:
    # (1) Wrong call raises TypeError.
    typeerror_observed = False
    try:
        pml.Radial(rad=0.7, alpha=2j)
    except TypeError as e:
        typeerror_observed = (
            "incompatible function arguments" in str(e)
            or "origin" in str(e))
    print(f"missing_origin_raises_typeerror="
          f"{typeerror_observed}")

    # (2) Correct call (matching the patched template).
    correct_ok = False
    try:
        obj = pml.Radial(origin=(0, 0), rad=0.7, alpha=2j)
        correct_ok = obj is not None
    except Exception as e:
        print(f"correct_call_error={e!r}", file=sys.stderr)
    print(f"call_with_origin_2d_ok={correct_ok}")

    # (3) 3-D variant
    correct_3d = False
    try:
        pml.Radial(origin=(0, 0, 0), rad=1.0, alpha=1j)
        correct_3d = True
    except Exception:
        pass
    print(f"call_with_origin_3d_ok={correct_3d}")

    # (4) Docstring records 'origin' (regression for
    # future pybind11 signature changes that drop the
    # kwarg name)
    doc = pml.Radial.__doc__ or ""
    doc_has_origin = "origin" in doc
    print(f"pml_radial_docstring_has_origin={doc_has_origin}")

    ok = (typeerror_observed and correct_ok and correct_3d
          and doc_has_origin)
    if ok:
        return 0
    print("FAIL: pml.Radial signature regression",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
