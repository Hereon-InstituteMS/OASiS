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

Mutation control: T2_MUTATE=1 assigns a CONSTITUTIVE_LAW to the Properties before Initialize, which is the documented requirement the fixture omits, so Initialize succeeds and the RuntimeError disappears.
"""
from __future__ import annotations

import os
import sys
import traceback

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA  # noqa: F401

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=constitutive_law_assigned_to_the_properties")


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("Test")
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    for i in range(1, 5):
        mp.CreateNewNode(i, float(i % 2), float(i // 2), 0.0)
    prop = mp.CreateNewProperties(1)
    prop.SetValue(KM.YOUNG_MODULUS, 1.0)
    prop.SetValue(KM.POISSON_RATIO, 0.3)
    # NOTE: NO CONSTITUTIVE_LAW assigned (unless mutated).
    if MUTATE:
        prop.SetValue(KM.CONSTITUTIVE_LAW,
                      SMA.LinearElasticPlaneStrain2DLaw())
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
