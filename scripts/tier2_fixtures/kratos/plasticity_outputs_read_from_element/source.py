"""Tier-2: the three plastic outputs live in two different modules.

The claim's parenthesis — 'PLASTIC_DISSIPATION is a KM
variable, the other two CLA' — is the operative part: three
related quantities, two different modules, and the natural
assumption that they share one is wrong.

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
import KratosMultiphysics.ConstitutiveLawsApplication as CLA


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
    if not _probe('CLA.EQUIVALENT_PLASTIC_STRAIN', lambda: (CLA.EQUIVALENT_PLASTIC_STRAIN), True):
        bad += 1
    if not _probe('KM.EQUIVALENT_PLASTIC_STRAIN', lambda: (KM.EQUIVALENT_PLASTIC_STRAIN), False):
        bad += 1
    if not _probe('CLA.UNIAXIAL_STRESS', lambda: (CLA.UNIAXIAL_STRESS), True):
        bad += 1
    if not _probe('KM.UNIAXIAL_STRESS', lambda: (KM.UNIAXIAL_STRESS), False):
        bad += 1
    if not _probe('KM.PLASTIC_DISSIPATION', lambda: (KM.PLASTIC_DISSIPATION), True):
        bad += 1
    if not _probe('CLA.PLASTIC_DISSIPATION', lambda: (CLA.PLASTIC_DISSIPATION), False):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
