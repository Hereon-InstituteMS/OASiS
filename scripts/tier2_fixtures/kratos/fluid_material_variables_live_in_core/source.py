"""Tier-2: DENSITY and DYNAMIC_VISCOSITY are core variables.

Omitting either key gives a converged, trivial answer with no
error. What a fixture can pin is where the two
variables live, since reaching for the fluid module raises.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.FluidDynamicsApplication as FDA


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
    if not _probe('KM.DENSITY', lambda: (KM.DENSITY), True):
        bad += 1
    if not _probe('FDA.DENSITY', lambda: (FDA.DENSITY), False):
        bad += 1
    if not _probe('KM.DYNAMIC_VISCOSITY', lambda: (KM.DYNAMIC_VISCOSITY), True):
        bad += 1
    if not _probe('FDA.DYNAMIC_VISCOSITY', lambda: (FDA.DYNAMIC_VISCOSITY), False):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
