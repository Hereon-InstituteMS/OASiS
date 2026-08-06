"""Tier-2: poromechanics variables resolve only through GetVariable.

Both obvious Python routes — core module and application module
— raise AttributeError. Only the kernel string lookup works.
The fixture also pins the *_LIQUID naming the pitfall warns
about.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.PoromechanicsApplication as PORO


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
    if not _probe('KM.DENSITY_SOLID', lambda: (KM.DENSITY_SOLID), False):
        bad += 1
    if not _probe('PORO.DENSITY_SOLID', lambda: (PORO.DENSITY_SOLID), False):
        bad += 1
    if not _probe('GetVariable_DENSITY_SOLID', lambda: (KM.KratosGlobals.GetVariable('DENSITY_SOLID')), True):
        bad += 1
    if not _probe('GetVariable_DENSITY_FLUID', lambda: (KM.KratosGlobals.GetVariable('DENSITY_FLUID')), False):
        bad += 1
    if not _probe('GetVariable_LIQUID_PRESSURE', lambda: (KM.KratosGlobals.GetVariable('LIQUID_PRESSURE')), True):
        bad += 1
    if not _probe('GetVariable_PERMEABILITY_XX', lambda: (KM.KratosGlobals.GetVariable('PERMEABILITY_XX')), True):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
