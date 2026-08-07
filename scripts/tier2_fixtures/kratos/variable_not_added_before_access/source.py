"""Tier-2: Kratos solution-step variable accessed before being added.

Pitfall (Kratos linear_elasticity, retroactive from PR #24):
``mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)`` must be called
BEFORE the first node is created. Reading a variable that wasn't
added raises RuntimeError with a precise diagnostic message.

Mutation control: T2_MUTATE=1 ADDS DISPLACEMENT to the solution-step variable list before the node is created, i.e. it does the documented correct thing. GetSolutionStepValue then returns a value instead of raising, so the RuntimeError and its 'variables list' text disappear.
"""
from __future__ import annotations

import os
import sys
import traceback

import KratosMultiphysics as KM

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=displacement_variable_added_before_the_node")


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("test")
    if MUTATE:
        # Pathology removed: the variable IS added before the node exists.
        mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    mp.SetBufferSize(1)
    # NOTE: DISPLACEMENT NOT added (unless mutated).
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    try:
        mp.Nodes[1].GetSolutionStepValue(KM.DISPLACEMENT)
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: GetSolutionStepValue returned a value for a "
          "variable that wasn't added", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
