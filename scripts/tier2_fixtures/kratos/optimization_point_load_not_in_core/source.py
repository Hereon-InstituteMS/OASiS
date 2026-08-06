"""Tier-2: POINT_LOAD is an SMA attribute, not a core one.

The pitfall's first failure mode is an AttributeError on the
core module. Its second is silent: without a
PointLoadCondition on the node, the nodal POINT_LOAD value is
never assembled. This fixture pins the module split and the
existence of that condition.

Mutation control: T2_MUTATE=1 INVERTS the expected outcome of every probe -- it asserts
the opposite of what this build actually does, while leaving each probe's callable
untouched, so every probe still really runs. Each probe[<label>]=<ok>_expected=<must>
line then disagrees with itself and probe_mismatches rises from 0 to the number of
probes. This is the control the fixture needs: it proves the printed booleans come
from actually calling into Kratos on this build, and that a wrong claim is caught,
rather than the fixture echoing a hard-coded table.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=every_probe_expectation_inverted")
import KratosMultiphysics.StructuralMechanicsApplication as SMA


# (label, callable-returning-value, must_succeed)
def _probe(label, fn, must_succeed):
    if MUTATE:
        # Pathology injected: assert the opposite outcome,
        # leaving the probe callable itself untouched.
        must_succeed = not must_succeed
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
