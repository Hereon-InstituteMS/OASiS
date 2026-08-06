"""Tier-2: ShallowWaterApplication registers Boussinesq*, not ShallowWater*.

The Application class is named ShallowWater but the element
registration uses the Boussinesq stem (the depth-averaged
Boussinesq equations). The obvious name is not registered.

Mutation control: T2_MUTATE=1 puts the CORRECT registered name BoussinesqElement2D3N in the slot where the fixture expects the wrong ShallowWaterElement name, so nothing is rejected and the rejected_name line disappears.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.ShallowWaterApplication  # noqa: F401

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=wrong_shallowwater_name_replaced_by_boussinesq")


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("t")
    mp.SetBufferSize(1)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    for i, (x, y, z) in enumerate([(0, 0, 0), (1, 0, 0), (1, 1, 0)]):
        mp.CreateNewNode(i + 1, float(x), float(y), float(z))
    prop = mp.CreateNewProperties(1)
    ids = list(range(1, 3 + 1))

    # The RIGHT name must work, or this fixture is testing nothing.
    try:
        mp.CreateNewElement("BoussinesqElement2D3N", 1, ids, prop)
        print("registered_ok=BoussinesqElement2D3N")
    except Exception:
        traceback.print_exc()
        print("FAIL: the CORRECT name is not registered either — "
              "the fixture cannot discriminate", file=sys.stderr)
        return 3

    # The WRONG name from the pitfall must be rejected.
    try:
        probe_name = ("BoussinesqElement2D3N" if MUTATE
                      else "ShallowWaterElement2D3N")
        mp.CreateNewElement(probe_name, 2, ids, prop)
    except Exception as exc:
        print("rejected_name=ShallowWaterElement2D3N")
        print(str(exc).splitlines()[0])
        return 0
    print("FAIL: Kratos ACCEPTED ShallowWaterElement2D3N", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
