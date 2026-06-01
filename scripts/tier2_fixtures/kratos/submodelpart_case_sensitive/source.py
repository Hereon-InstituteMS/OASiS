"""Tier-2: Kratos SubModelPart names are case-sensitive.

Pitfall (Kratos linear_elasticity#2): SubModelPart names must
match EXACTLY between the .mdpa and ProjectParameters.json.
Kratos is case-sensitive and does not strip whitespace.
ModelPart.GetSubModelPart('inlet') when the actual name is
'Inlet' raises:

  RuntimeError: Error: There is no sub model part with name
  "inlet" in model part "Structure" ... from
  ModelPart::ErrorNonExistingSubModelPart in
  kratos/sources/model_part.cpp:2406
"""
from __future__ import annotations

import sys
import traceback

import KratosMultiphysics as KM


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("Structure")
    mp.CreateSubModelPart("Inlet")
    try:
        # Wrong case — should raise.
        mp.GetSubModelPart("inlet")
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: GetSubModelPart accepted lower-case name when "
          "the actual name was 'Inlet' (catalog claim wrong)",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
