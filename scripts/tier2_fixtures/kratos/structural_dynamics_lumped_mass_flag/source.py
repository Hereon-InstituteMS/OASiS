"""Tier-2: COMPUTE_LUMPED_MASS_MATRIX is a core flag, unset by default.

Consistent versus lumped is one boolean. It lives on core, it
is absent until set, and its absence means consistent.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA


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
    if not _probe('KM.COMPUTE_LUMPED_MASS_MATRIX', lambda: (KM.COMPUTE_LUMPED_MASS_MATRIX), True):
        bad += 1
    if not _probe('SMA.COMPUTE_LUMPED_MASS_MATRIX', lambda: (SMA.COMPUTE_LUMPED_MASS_MATRIX), False):
        bad += 1
    if not _probe('unset_lookup_raises', lambda: (KM.Model().CreateModelPart('m1').ProcessInfo[KM.COMPUTE_LUMPED_MASS_MATRIX]), False):
        bad += 1
    if not _probe('set_true_roundtrip', lambda: ((lambda m: (m.ProcessInfo.SetValue(KM.COMPUTE_LUMPED_MASS_MATRIX, True), m.ProcessInfo[KM.COMPUTE_LUMPED_MASS_MATRIX] is True)[1])(KM.Model().CreateModelPart('m2'))), True):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
