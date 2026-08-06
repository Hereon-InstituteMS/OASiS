"""Tier-2: FREESTREAM_VELOCITY and MACH_INFINITY do not exist.

The pitfall prescribes two variable names. Neither is
registered anywhere in Kratos 10.4.3. The real names are
FREE_STREAM_VELOCITY and FREE_STREAM_MACH, on the
CompressiblePotentialFlow module.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.FluidDynamicsApplication  # noqa: F401
import KratosMultiphysics.CompressiblePotentialFlowApplication as CPF


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
    if not _probe('CPF.FREESTREAM_VELOCITY', lambda: (CPF.FREESTREAM_VELOCITY), False):
        bad += 1
    if not _probe('CPF.MACH_INFINITY', lambda: (CPF.MACH_INFINITY), False):
        bad += 1
    if not _probe('KM.MACH_INFINITY', lambda: (KM.MACH_INFINITY), False):
        bad += 1
    if not _probe('CPF.FREE_STREAM_VELOCITY', lambda: (CPF.FREE_STREAM_VELOCITY), True):
        bad += 1
    if not _probe('CPF.FREE_STREAM_MACH', lambda: (CPF.FREE_STREAM_MACH), True):
        bad += 1
    if not _probe('CPF.FREE_STREAM_DENSITY', lambda: (CPF.FREE_STREAM_DENSITY), True):
        bad += 1
    if not _probe('KM.FREE_STREAM_MACH', lambda: (KM.FREE_STREAM_MACH), False):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
