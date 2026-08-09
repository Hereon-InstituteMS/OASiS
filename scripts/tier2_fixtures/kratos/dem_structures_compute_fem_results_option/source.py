"""Tier-2: the enabling flag and the zeroed variable live in two different modules.

The claim's failure is one-way decoupling with rc=0. Finding
the flag means knowing it is DEM's, and finding the zeroed load
means knowing it is the coupling application's.

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
    if MUTATE:
        # Pathology injected: assert the opposite outcome,
        # leaving the probe callable itself untouched.
        must_succeed = not must_succeed
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
    if not _probe('DEM.COMPUTE_FEM_RESULTS_OPTION', lambda: (DEM.COMPUTE_FEM_RESULTS_OPTION), True):
        bad += 1
    if not _probe('KM.COMPUTE_FEM_RESULTS_OPTION', lambda: (KM.COMPUTE_FEM_RESULTS_OPTION), False):
        bad += 1
    if not _probe('DSC.DEM_SURFACE_LOAD', lambda: (DSC.DEM_SURFACE_LOAD), True):
        bad += 1
    if not _probe('DEM.DEM_SURFACE_LOAD', lambda: (DEM.DEM_SURFACE_LOAD), False):
        bad += 1
    if not _probe('KM.DEM_SURFACE_LOAD', lambda: (KM.DEM_SURFACE_LOAD), False):
        bad += 1
    if not _probe('DEM.DEM_NODAL_AREA', lambda: (DEM.DEM_NODAL_AREA), True):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
