"""Tier-2: ConvectionDiffusionSettings must be on ProcessInfo, and its absence is silent.

The claim is that the settings must be set before the solve.
Measured, the omission cannot be caught by the obvious guard:
reading the key returns None instead of raising, and the read
ITSELF inserts the key, so a later Has() reports True.

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
import KratosMultiphysics.ConvectionDiffusionApplication as CDA


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
    if not _probe('KM.ConvectionDiffusionSettings', lambda: (KM.ConvectionDiffusionSettings), True):
        bad += 1
    if not _probe('CDA.ConvectionDiffusionSettings', lambda: (CDA.ConvectionDiffusionSettings), False):
        bad += 1
    if not _probe('KM.CONVECTION_DIFFUSION_SETTINGS', lambda: (KM.CONVECTION_DIFFUSION_SETTINGS), True):
        bad += 1

    # The three-step sequence that makes the omission silent.
    _mp = KM.Model().CreateModelPart('cds')
    if not _probe('fresh_Has_is_true', lambda: (_mp.ProcessInfo.Has(KM.CONVECTION_DIFFUSION_SETTINGS)), False):
        bad += 1
    if not _probe('unset_subscript_is_None', lambda: (_mp.ProcessInfo[KM.CONVECTION_DIFFUSION_SETTINGS] is None), True):
        bad += 1
    if not _probe('Has_true_after_a_mere_read', lambda: (_mp.ProcessInfo.Has(KM.CONVECTION_DIFFUSION_SETTINGS)), True):
        bad += 1
    _mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, KM.ConvectionDiffusionSettings())
    if not _probe('after_set_subscript_is_None', lambda: (_mp.ProcessInfo[KM.CONVECTION_DIFFUSION_SETTINGS] is None), False):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
