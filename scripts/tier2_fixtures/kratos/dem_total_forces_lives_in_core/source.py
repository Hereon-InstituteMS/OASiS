"""Tier-2: TOTAL_FORCES is core; CONTACT_FORCES/ELASTIC_FORCES are DEM-module.

The pitfall's action-reaction cross-check needs TOTAL_FORCES.
Dotting it off the DEM module — the natural guess, since every
other force variable in that workflow lives there — raises
AttributeError.
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
    if not _probe('DEM.TOTAL_FORCES_Z', lambda: (DEM.TOTAL_FORCES_Z), False):
        bad += 1
    if not _probe('KM.TOTAL_FORCES_Z', lambda: (KM.TOTAL_FORCES_Z), True):
        bad += 1
    if not _probe('DEM.CONTACT_FORCES', lambda: (DEM.CONTACT_FORCES), True):
        bad += 1
    if not _probe('KM.CONTACT_FORCES', lambda: (KM.CONTACT_FORCES), False):
        bad += 1
    if not _probe('DEM.ELASTIC_FORCES', lambda: (DEM.ELASTIC_FORCES), True):
        bad += 1
    if not _probe('DEM.DEM_NODAL_AREA', lambda: (DEM.DEM_NODAL_AREA), True):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
