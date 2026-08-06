"""Tier-2: the polynomial variable is CableNet's and its order is unchecked.

A reversed coefficient vector produces wildly wrong forces with
no error at all. The only pre-run facts are where the variable
lives and that a 2-entry vector is accepted.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.CableNetApplication as CN


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
    if not _probe('CN.SPRING_DEFORMATION_EMPIRICAL_POLYNOMIAL', lambda: (CN.SPRING_DEFORMATION_EMPIRICAL_POLYNOMIAL), True):
        bad += 1
    if not _probe('KM.SPRING_DEFORMATION_EMPIRICAL_POLYNOMIAL', lambda: (KM.SPRING_DEFORMATION_EMPIRICAL_POLYNOMIAL), False):
        bad += 1
    if not _probe('two_entry_vector_accepted', lambda: ((lambda p: (p.SetValue(CN.SPRING_DEFORMATION_EMPIRICAL_POLYNOMIAL, KM.Vector([1000.0, 0.0])), len(p.GetValue(CN.SPRING_DEFORMATION_EMPIRICAL_POLYNOMIAL)) == 2)[1])(KM.Model().CreateModelPart('sp').CreateNewProperties(1))), True):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
