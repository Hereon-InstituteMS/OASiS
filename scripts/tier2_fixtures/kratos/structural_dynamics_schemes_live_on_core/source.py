"""Tier-2: the Newmark/Bossak schemes are core classes.

The instability is not guarded: no constructor rejects
gamma < 0.5. What is checkable before a run is that the
schemes are core, not application, classes.
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
    if not _probe('KM.ResidualBasedNewmarkDisplacementScheme', lambda: (KM.ResidualBasedNewmarkDisplacementScheme), True):
        bad += 1
    if not _probe('SMA.ResidualBasedNewmarkDisplacementScheme', lambda: (SMA.ResidualBasedNewmarkDisplacementScheme), False):
        bad += 1
    if not _probe('KM.ResidualBasedBossakDisplacementScheme', lambda: (KM.ResidualBasedBossakDisplacementScheme), True):
        bad += 1
    if not _probe('SMA.ResidualBasedBossakDisplacementScheme', lambda: (SMA.ResidualBasedBossakDisplacementScheme), False):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
