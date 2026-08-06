"""Tier-2: FACE_HEAT_FLUX is the flux variable and lives on core.

Both TEMPERATURE and FACE_HEAT_FLUX exist, so nothing rejects
the wrong target — the claim's failure is a wrong answer, not
an exception. The fixture pins where the correct variable lives
and that the prescribed process class is real.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication as CDA


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
    if not _probe('KM.FACE_HEAT_FLUX', lambda: (KM.FACE_HEAT_FLUX), True):
        bad += 1
    if not _probe('CDA.FACE_HEAT_FLUX', lambda: (CDA.FACE_HEAT_FLUX), False):
        bad += 1
    if not _probe('KM.TEMPERATURE', lambda: (KM.TEMPERATURE), True):
        bad += 1
    if not _probe('ApplyConstantScalarValueProcess_exists', lambda: (KM.ApplyConstantScalarValueProcess), True):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
