"""Tier-2: the BDF2 scheme lives on SW and needs a 3-deep buffer.

Nothing enforces the buffer depth. A run with the default
buffer silently uses an empty history slot for the first BDF
derivative and then 'recovers' with a polluted transient.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.ShallowWaterApplication as SW


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
    if not _probe('SW.ShallowWaterResidualBasedBDFScheme', lambda: (SW.ShallowWaterResidualBasedBDFScheme), True):
        bad += 1
    if not _probe('KM.ShallowWaterResidualBasedBDFScheme', lambda: (KM.ShallowWaterResidualBasedBDFScheme), False):
        bad += 1
    if not _probe('buffer_three_accepted', lambda: ((lambda m: (m.SetBufferSize(3), m.GetBufferSize() == 3)[1])(KM.Model().CreateModelPart('b1'))), True):
        bad += 1
    if not _probe('default_buffer_is_three', lambda: (KM.Model().CreateModelPart('b2').GetBufferSize() == 3), False):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
