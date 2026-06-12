"""Tier-2: Kratos element name missing node count.

Pitfall (kratos.linear_elasticity #0): 'Element names MUST include
node count: SmallDisplacementElement2D3N, not SmallDisplacement2D'.
The wrong shorter name is not in the Kratos Registry, so
CreateNewElement raises RuntimeError 'Element ... is not
registered'.
"""
from __future__ import annotations

import sys
import traceback

import KratosMultiphysics as KM


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("test")
    mp.SetBufferSize(1)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    mp.CreateNewNode(2, 1.0, 0.0, 0.0)
    mp.CreateNewNode(3, 1.0, 1.0, 0.0)
    mp.CreateNewNode(4, 0.0, 1.0, 0.0)
    prop = mp.CreateNewProperties(1)
    try:
        # Wrong: no node-count suffix. Should be
        # SmallDisplacementElement2D4N or similar.
        mp.CreateNewElement("SmallDisplacement2D", 1, [1, 2, 3, 4], prop)
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: Kratos accepted unregistered element name",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
