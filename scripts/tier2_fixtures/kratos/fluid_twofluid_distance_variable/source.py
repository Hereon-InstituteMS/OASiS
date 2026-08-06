"""Tier-2: DISTANCE is a core variable defaulting to zero everywhere.

The failure is silent: an unset DISTANCE is 0.0 at every node,
so the zero level set passes through the whole
domain rather than along an interface.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.FluidDynamicsApplication as FDA


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
    if not _probe('KM.DISTANCE', lambda: (KM.DISTANCE), True):
        bad += 1
    if not _probe('FDA.DISTANCE', lambda: (FDA.DISTANCE), False):
        bad += 1
    if not _probe('nodal_read_without_variable', lambda: ((lambda m: (m.SetBufferSize(1), m.CreateNewNode(1, 0.0, 0.0, 0.0), m.GetNode(1).GetSolutionStepValue(KM.DISTANCE))[2])(KM.Model().CreateModelPart('d1'))), False):
        bad += 1
    if not _probe('nodal_default_is_zero', lambda: ((lambda m: (m.AddNodalSolutionStepVariable(KM.DISTANCE), m.SetBufferSize(1), m.CreateNewNode(1, 0.0, 0.0, 0.0), m.GetNode(1).GetSolutionStepValue(KM.DISTANCE) == 0.0)[3])(KM.Model().CreateModelPart('d2'))), True):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
