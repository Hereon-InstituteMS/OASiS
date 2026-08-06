"""Tier-2: the adaptor classes are not on core; the Properties helper is on KOA.

The claim's failure is a TypeError deep in CalculateGradient.
Before that, the class names themselves have to be found, and
they are not where a reader would first look.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.OptimizationApplication as KOA


# (label, callable-returning-value, must_succeed)
def _probe(label, fn, must_succeed):
    """A probe FAILS if it raises, and equally if it returns False.

    Both matter: some claims are 'this attribute does not resolve'
    (an exception) and some are 'this predicate is false' (a bool).
    Treating a returned False as success would let a fixture report a
    pass while observing the opposite of what it claims.
    """
    try:
        val = fn()
        ok = val is not False
        err = "" if ok else "returned False"
    except Exception as exc:
        ok = False
        err = f"{type(exc).__name__}: {str(exc).strip().splitlines()[0] if str(exc).strip() else ''}"
    print(f"probe[{label}]={ok}_expected={must_succeed}")
    if not ok:
        print(f"  message: {err[:150]}")
    return ok == must_succeed


def main() -> int:
    bad = 0
    if not _probe('KM.DoubleCombinedTensorAdaptor', lambda: (KM.DoubleCombinedTensorAdaptor), False):
        bad += 1
    if not _probe('KM.VariableTensorAdaptor', lambda: (KM.VariableTensorAdaptor), False):
        bad += 1
    if not _probe('KM.DoubleTensorAdaptor', lambda: (KM.DoubleTensorAdaptor), False):
        bad += 1
    if not _probe('KOA.OptimizationUtils', lambda: (KOA.OptimizationUtils), True):
        bad += 1
    if not _probe('CreateEntitySpecificPropertiesForContainer', lambda: (KOA.OptimizationUtils.CreateEntitySpecificPropertiesForContainer), True):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
