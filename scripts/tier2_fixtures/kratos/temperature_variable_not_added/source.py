"""Tier-2: Kratos TEMPERATURE accessed before being added.

Pitfall (Kratos heat#0): TEMPERATURE must be added to a
ModelPart's variables list via AddNodalSolutionStepVariable
BEFORE any Node is created. Accessing the variable on a Node
without that step raises:

  RuntimeError: This container only can store the variables
  specified in its variables list. The variables list doesn't
  have this variable: TEMPERATURE ...
  ... in kratos/containers/variables_list_data_value_container

Same family as fluid#0 (VELOCITY) and linear_elasticity#9
mode (b) (variable never added).
"""
from __future__ import annotations

import sys
import traceback

import KratosMultiphysics as KM


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("Heat")
    # NOTE: skip AddNodalSolutionStepVariable(KM.TEMPERATURE)
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    try:
        mp.Nodes[1].GetSolutionStepValue(KM.TEMPERATURE)
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: GetSolutionStepValue accepted unregistered "
          "TEMPERATURE variable (catalog claim wrong)",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
