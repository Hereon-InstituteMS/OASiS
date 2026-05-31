"""Tier-2: Kratos solution-step variable accessed before being added.

Pitfall (Kratos linear_elasticity, retroactive from PR #24):
``mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)`` must be called
BEFORE the first node is created. Reading a variable that wasn't
added raises RuntimeError with a precise diagnostic message.
"""
from __future__ import annotations

import sys
import traceback

import KratosMultiphysics as KM


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("test")
    mp.SetBufferSize(1)
    # NOTE: DISPLACEMENT NOT added.
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
