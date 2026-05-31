"""Tier-2: Kratos CONSTITUTIVE_LAW binding wants an instance.

The pitfall (Kratos linear_elasticity, from PR #24 retroactive
post-mortem): ``prop.SetValue(KM.CONSTITUTIVE_LAW, ...)`` accepts
a constructed instance. Passing the class itself (without ``()``)
raises pybind11 TypeError because the registered signature
expects a ``ConstitutiveLaw`` object, not a type. A community
user copying example code that omits the parens silently gets
this — except pybind11 catches it at the binding boundary.
"""
from __future__ import annotations

import sys
import traceback

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("test")
    mp.SetBufferSize(1)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    prop = mp.CreateNewProperties(1)
    try:
        # Bug: SMA.LinearElasticPlaneStrain2DLaw is the CLASS,
        # not an instance. The correct form is
        # SMA.LinearElasticPlaneStrain2DLaw() (with parens).
        prop.SetValue(KM.CONSTITUTIVE_LAW,
                      SMA.LinearElasticPlaneStrain2DLaw)
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: Kratos accepted the class without instantiation",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
