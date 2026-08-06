"""Tier-2: a missing echo_level raises from C++ GetValue.

Two catalog sections carry this claim with the same corrected
wording. Both are pinned so either going stale is caught.
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
    if not _probe('lookup_missing_echo_level', lambda: (KM.Parameters('{"problem_name":"x"}')['echo_level']), False):
        bad += 1
    if not _probe('lookup_present_echo_level', lambda: (KM.Parameters('{"problem_name":"x","echo_level":0}')['echo_level'].GetInt() == 0), True):
        bad += 1
    if not _probe('Has_reports_absent', lambda: (KM.Parameters('{"problem_name":"x"}').Has('echo_level')), False):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
