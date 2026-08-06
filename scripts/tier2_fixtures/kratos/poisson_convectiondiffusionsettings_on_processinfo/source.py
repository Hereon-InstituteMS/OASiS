"""Tier-2: ConvectionDiffusionSettings is core and must be on ProcessInfo.

The elements read their diffusion / source / unknown variables
through this settings object. Without it on ProcessInfo the
lookup fails; the class itself is not where the application's
name suggests.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication as CDA


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
    if not _probe('KM.ConvectionDiffusionSettings', lambda: (KM.ConvectionDiffusionSettings), True):
        bad += 1
    if not _probe('CDA.ConvectionDiffusionSettings', lambda: (CDA.ConvectionDiffusionSettings), False):
        bad += 1
    if not _probe('KM.CONVECTION_DIFFUSION_SETTINGS', lambda: (KM.CONVECTION_DIFFUSION_SETTINGS), True):
        bad += 1
    if not _probe('lookup_before_set', lambda: (KM.Model().CreateModelPart('p1').ProcessInfo[KM.CONVECTION_DIFFUSION_SETTINGS]), False):
        bad += 1
    if not _probe('lookup_after_set', lambda: ((lambda m: (m.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, KM.ConvectionDiffusionSettings()), m.ProcessInfo[KM.CONVECTION_DIFFUSION_SETTINGS] is not None)[1])(KM.Model().CreateModelPart('p2'))), True):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
