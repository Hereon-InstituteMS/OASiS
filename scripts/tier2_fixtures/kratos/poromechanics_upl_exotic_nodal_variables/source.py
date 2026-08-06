"""Tier-2: the exotic UPl nodal variables are all Poromechanics attributes.

The claim's own note is that the multi-threaded failure mode is
an uninformative 'terminate called recursively' core dump. The
checkable precondition is that every name in the required list
is reachable, and from which module.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.PoromechanicsApplication as PORO


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
    if not _probe('PORO.INITIAL_STRESS_TENSOR', lambda: (PORO.INITIAL_STRESS_TENSOR), True):
        bad += 1
    if not _probe('PORO.NODAL_EFFECTIVE_STRESS_TENSOR', lambda: (PORO.NODAL_EFFECTIVE_STRESS_TENSOR), True):
        bad += 1
    if not _probe('PORO.NODAL_MID_PLANE_LIQUID_PRESSURE', lambda: (PORO.NODAL_MID_PLANE_LIQUID_PRESSURE), True):
        bad += 1
    if not _probe('PORO.NODAL_SLIP_TENDENCY', lambda: (PORO.NODAL_SLIP_TENDENCY), True):
        bad += 1
    if not _probe('PORO.NODAL_JOINT_DAMAGE', lambda: (PORO.NODAL_JOINT_DAMAGE), True):
        bad += 1
    if not _probe('PORO.DT_LIQUID_PRESSURE', lambda: (PORO.DT_LIQUID_PRESSURE), True):
        bad += 1
    if not _probe('PORO.NORMAL_LIQUID_FLUX', lambda: (PORO.NORMAL_LIQUID_FLUX), True):
        bad += 1
    if not _probe('PORO.LIQUID_DISCHARGE', lambda: (PORO.LIQUID_DISCHARGE), True):
        bad += 1
    if not _probe('KM.INITIAL_STRESS_TENSOR', lambda: (KM.INITIAL_STRESS_TENSOR), False):
        bad += 1
    if not _probe('KM.DT_LIQUID_PRESSURE', lambda: (KM.DT_LIQUID_PRESSURE), False):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
