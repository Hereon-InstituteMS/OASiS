"""Tier-2: the smoothing scheme's nodal set spans core and Poromechanics.

The claim is that a purely mechanical run still needs a
poromechanics nodal set. Where each variable lives is the
checking surface, and three of the four are on neither the
module that owns the scheme nor core.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.PoromechanicsApplication as PORO
import KratosMultiphysics.DamApplication as DAM


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
    if not _probe('DAM.IncrementalUpdateStaticSmoothingScheme', lambda: (DAM.IncrementalUpdateStaticSmoothingScheme), True):
        bad += 1
    if not _probe('KM.NODAL_AREA', lambda: (KM.NODAL_AREA), True):
        bad += 1
    if not _probe('PORO.NODAL_CAUCHY_STRESS_TENSOR', lambda: (PORO.NODAL_CAUCHY_STRESS_TENSOR), True):
        bad += 1
    if not _probe('KM.NODAL_CAUCHY_STRESS_TENSOR', lambda: (KM.NODAL_CAUCHY_STRESS_TENSOR), False):
        bad += 1
    if not _probe('PORO.NODAL_JOINT_WIDTH', lambda: (PORO.NODAL_JOINT_WIDTH), True):
        bad += 1
    if not _probe('PORO.NODAL_JOINT_AREA', lambda: (PORO.NODAL_JOINT_AREA), True):
        bad += 1
    if not _probe('DAM.NODAL_CAUCHY_STRESS_TENSOR', lambda: (DAM.NODAL_CAUCHY_STRESS_TENSOR), False):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
