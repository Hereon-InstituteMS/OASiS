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

Mutation control: T2_MUTATE=1 asks for the sub model part by its actual name 'Inlet' instead of the lower-case 'inlet', removing the case mismatch. The lookup then succeeds and the RuntimeError with 'no sub model part with name' disappears.
"""
from __future__ import annotations

import os
import sys
import traceback

import KratosMultiphysics as KM

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=submodelpart_requested_with_its_actual_case")


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("Structure")
    mp.CreateSubModelPart("Inlet")
    try:
        # Wrong case — should raise. Under mutation the case is correct.
        mp.GetSubModelPart("Inlet" if MUTATE else "inlet")
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: GetSubModelPart accepted lower-case name when "
          "the actual name was 'Inlet' (catalog claim wrong)",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
