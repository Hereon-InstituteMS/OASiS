"""Tier-2: the required nodal variables span three modules.

A single AddNodalSolutionStepVariable list drawn from one
module cannot be assembled — the names come from two different
applications and none from core.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication  # noqa: F401
import KratosMultiphysics.DEMApplication as DEM
import KratosMultiphysics.DemStructuresCouplingApplication as DSC


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
    if not _probe('DSC.DEM_SURFACE_LOAD', lambda: (DSC.DEM_SURFACE_LOAD), True):
        bad += 1
    if not _probe('DSC.SMOOTHED_STRUCTURAL_VELOCITY', lambda: (DSC.SMOOTHED_STRUCTURAL_VELOCITY), True):
        bad += 1
    if not _probe('DEM.DELTA_DISPLACEMENT', lambda: (DEM.DELTA_DISPLACEMENT), True):
        bad += 1
    if not _probe('DEM.DEM_PRESSURE', lambda: (DEM.DEM_PRESSURE), True):
        bad += 1
    if not _probe('DEM.TANGENTIAL_ELASTIC_FORCES', lambda: (DEM.TANGENTIAL_ELASTIC_FORCES), True):
        bad += 1
    if not _probe('DEM.NON_DIMENSIONAL_VOLUME_WEAR', lambda: (DEM.NON_DIMENSIONAL_VOLUME_WEAR), True):
        bad += 1
    if not _probe('KM.DEM_PRESSURE', lambda: (KM.DEM_PRESSURE), False):
        bad += 1
    if not _probe('DEM.SMOOTHED_STRUCTURAL_VELOCITY', lambda: (DEM.SMOOTHED_STRUCTURAL_VELOCITY), False):
        bad += 1
    if not _probe('GetVariable_DEM_SURFACE_LOAD', lambda: (KM.KratosGlobals.GetVariable('DEM_SURFACE_LOAD')), True):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
