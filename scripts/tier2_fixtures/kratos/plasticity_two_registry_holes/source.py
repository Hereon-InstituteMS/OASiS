"""Tier-2: two CL classes exist in Python but not in the string registry.

Materials.json resolves constitutive laws by string through the
kernel registry. These two classes are importable from Python
but missing from that registry, so the two routes disagree —
the defect the pitfall describes.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication  # noqa: F401
import KratosMultiphysics.ConstitutiveLawsApplication as CLA


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
    if not _probe('kinematic_python_attr', lambda: (CLA.SmallStrainKinematicPlasticityPlaneStrainVonMisesVonMises), True):
        bad += 1
    if not _probe('kinematic_registered', lambda: (KM.KratosGlobals.Kernel.HasConstitutiveLaw('SmallStrainKinematicPlasticityPlaneStrainVonMisesVonMises')), False):
        bad += 1
    if not _probe('factory_python_attr', lambda: (CLA.FiniteStrainIsotropicPlasticityFactory), True):
        bad += 1
    if not _probe('factory_registered', lambda: (KM.KratosGlobals.Kernel.HasConstitutiveLaw('FiniteStrainIsotropicPlasticityFactory')), False):
        bad += 1
    if not _probe('isotropic_sibling_registered', lambda: (KM.KratosGlobals.Kernel.HasConstitutiveLaw('SmallStrainIsotropicPlasticityPlaneStrainVonMisesVonMises')), True):
        bad += 1
    print(f"probe_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
