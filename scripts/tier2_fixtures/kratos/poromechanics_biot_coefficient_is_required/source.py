"""Tier-2: BIOT_COEFFICIENT is its own Properties entry.

The element Check names it by name when missing. Before that,
the checkable facts are that it is a distinct variable from the
bulk moduli and that it must be set on Properties by hand.

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
import KratosMultiphysics.PoromechanicsApplication as PORO


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
    if not _probe('PORO.BIOT_COEFFICIENT', lambda: (PORO.BIOT_COEFFICIENT), True):
        bad += 1
    if not _probe('KM.BIOT_COEFFICIENT', lambda: (KM.BIOT_COEFFICIENT), False):
        bad += 1
    if not _probe('properties_roundtrip', lambda: ((lambda p: (p.SetValue(PORO.BIOT_COEFFICIENT, 1.0), p.GetValue(PORO.BIOT_COEFFICIENT) == 1.0)[1])(KM.Model().CreateModelPart('bp').CreateNewProperties(1))), True):
        bad += 1
    if not _probe('BULK_MODULUS_SOLID_is_separate', lambda: (KM.KratosGlobals.GetVariable('BULK_MODULUS_SOLID') is not PORO.BIOT_COEFFICIENT), True):
        bad += 1
    if not _probe('BULK_MODULUS_LIQUID_exists', lambda: (KM.KratosGlobals.GetVariable('BULK_MODULUS_LIQUID')), True):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
