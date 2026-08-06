"""Tier-2: Dam thermal laws: Python class name != registered string.

The two routes into a constitutive law — a Python instance on
Properties, or a name in Materials.json — take DIFFERENT
strings for these laws, and each spelling fails on the other
route.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication  # noqa: F401
import KratosMultiphysics.PoromechanicsApplication  # noqa: F401
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
    if not _probe('python_attr_ThermalLinearElastic2DPlaneStrain', lambda: (DAM.ThermalLinearElastic2DPlaneStrain), True):
        bad += 1
    if not _probe('registered_ThermalLinearElastic2DPlaneStrain', lambda: (KM.KratosGlobals.Kernel.HasConstitutiveLaw('ThermalLinearElastic2DPlaneStrain')), False):
        bad += 1
    if not _probe('registered_ThermalLinearPlaneStrain', lambda: (KM.KratosGlobals.Kernel.HasConstitutiveLaw('ThermalLinearPlaneStrain')), True):
        bad += 1
    if not _probe('registered_ThermalLinearPlaneStress', lambda: (KM.KratosGlobals.Kernel.HasConstitutiveLaw('ThermalLinearPlaneStress')), True):
        bad += 1
    if not _probe('registered_ThermalElasticIsotropic3D', lambda: (KM.KratosGlobals.Kernel.HasConstitutiveLaw('ThermalElasticIsotropic3D')), True):
        bad += 1
    if not _probe('python_attr_ThermalLinearPlaneStrain', lambda: (DAM.ThermalLinearPlaneStrain), False):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
