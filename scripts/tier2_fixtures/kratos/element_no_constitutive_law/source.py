"""Tier-2: SmallDisplacementElement Initialize without constitutive law.

Pitfall (Kratos linear_elasticity#1): SmallDisplacementElement
(and other solid elements) requires CONSTITUTIVE_LAW set on
their Properties. Material parameters alone (YOUNG_MODULUS,
POISSON_RATIO) are not enough; the element checks for the law
at Initialize:

  RuntimeError: Error: A constitutive law needs to be specified
  for the element with ID 1
  ... in applications/StructuralMechanicsApplication/
  custom_elements/solid_elements/base_solid_element.cpp:249
"""
from __future__ import annotations

import sys
import traceback

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA  # noqa: F401


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("Test")
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    for i in range(1, 5):
        mp.CreateNewNode(i, float(i % 2), float(i // 2), 0.0)
    prop = mp.CreateNewProperties(1)
    prop.SetValue(KM.YOUNG_MODULUS, 1.0)
    prop.SetValue(KM.POISSON_RATIO, 0.3)
    # NOTE: NO CONSTITUTIVE_LAW assigned.
    elem = mp.CreateNewElement("SmallDisplacementElement2D3N", 1,
                                [1, 2, 3], prop)
    try:
        info = KM.ProcessInfo()
        elem.Initialize(info)
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: Initialize accepted Properties without "
          "CONSTITUTIVE_LAW (catalog claim wrong)",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
