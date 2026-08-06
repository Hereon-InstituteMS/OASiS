"""Tier-2: RADIUS is a core nodal variable defaulting to zero.

A particle whose RADIUS is never set does not error — it keeps
0.0. The fixture pins where the variable lives and that the
unset value is a silent zero rather than a rejection.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.DEMApplication as DEM


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
    if not _probe('KM.RADIUS', lambda: (KM.RADIUS), True):
        bad += 1
    if not _probe('DEM.RADIUS', lambda: (DEM.RADIUS), False):
        bad += 1
    if not _probe('nodal_read_without_variable', lambda: ((lambda m: (m.SetBufferSize(1), m.CreateNewNode(1, 0.0, 0.0, 0.0), m.GetNode(1).GetSolutionStepValue(KM.RADIUS))[2])(KM.Model().CreateModelPart('r1'))), False):
        bad += 1
    if not _probe('nodal_default_is_zero', lambda: ((lambda m: (m.AddNodalSolutionStepVariable(KM.RADIUS), m.SetBufferSize(1), m.CreateNewNode(1, 0.0, 0.0, 0.0), m.GetNode(1).GetSolutionStepValue(KM.RADIUS) == 0.0)[3])(KM.Model().CreateModelPart('r2'))), True):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
