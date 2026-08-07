"""Tier-2: Kratos VELOCITY accessed on a Node before being added.

Pitfall (Kratos fluid#0): VELOCITY (and PRESSURE, DISPLACEMENT,
etc.) must be added to a ModelPart's variables list via
ModelPart.AddNodalSolutionStepVariable BEFORE the first Node
is created. Accessing the variable later via
GetSolutionStepValue raises:

  RuntimeError: This container only can store the variables
  specified in its variables list. The variables list doesn't
  have this variable: VELOCITY ...
  ... in kratos/containers/variables_list_data_value_container

Family-related fixtures: heat#0 (TEMPERATURE) and
linear_elasticity#9 mode (b) (variable never added).

Mutation control: T2_MUTATE=1 ADDS VELOCITY to the solution-step variable list before the node is created. GetSolutionStepValue then returns a value instead of raising, so the RuntimeError and its 'variables list' / 'doesn't have this variable' text disappear.
"""
from __future__ import annotations

import os
import sys
import traceback

import KratosMultiphysics as KM

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=the_solution_step_variable_is_added_before_the_node")


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("Fluid")
    # NOTE: skip AddNodalSolutionStepVariable(KM.VELOCITY) unless mutated.
    if MUTATE:
        mp.AddNodalSolutionStepVariable(KM.VELOCITY)
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    try:
        mp.Nodes[1].GetSolutionStepValue(KM.VELOCITY)
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: GetSolutionStepValue accepted unregistered "
          "VELOCITY variable (catalog claim wrong)",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
