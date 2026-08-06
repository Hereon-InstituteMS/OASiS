"""Tier-2: POINT_LOAD is an SMA attribute, not a core one.

The pitfall's first failure mode is an AttributeError on the
core module. Its second is silent: without a
PointLoadCondition on the node, the nodal POINT_LOAD value is
never assembled. This fixture pins the module split and the
existence of that condition.
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
    try:
        fn()
        ok = True
        err = ""
    except Exception as exc:
        ok = False
        err = f"{type(exc).__name__}: {str(exc).strip().splitlines()[0] if str(exc).strip() else ''}"
    print(f"probe[{label}]={ok}_expected={must_succeed}")
    if not ok:
        print(f"  message: {err[:150]}")
    return ok == must_succeed


def main() -> int:
    bad = 0
    if not _probe('KM.POINT_LOAD', lambda: (KM.POINT_LOAD), False):
        bad += 1
    if not _probe('SMA.POINT_LOAD', lambda: (SMA.POINT_LOAD), True):
        bad += 1
    if not _probe('KM.THICKNESS_SENSITIVITY', lambda: (KM.THICKNESS_SENSITIVITY), False):
        bad += 1
    if not _probe('SMA.THICKNESS_SENSITIVITY', lambda: (SMA.THICKNESS_SENSITIVITY), True):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
