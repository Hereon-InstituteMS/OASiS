"""Tier-2: poromechanics element stem is UPl, not UPw.

Kratos 10.4.x renamed the u-Pw (water-pressure) formulation to
u-Pl (liquid-pressure). The historical UPw* element names are
gone from the registry. CreateNewElement with the UPw stem
raises; the UPl stem constructs.

Mutation control: T2_MUTATE=1 puts the CORRECT registered name UPlSmallStrainElement2D4N in the slot where the fixture expects the wrong UPw name, so nothing is rejected and the rejected_name line disappears.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.PoromechanicsApplication  # noqa: F401

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=wrong_upw_name_replaced_by_the_registered_upl_name")


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("t")
    mp.SetBufferSize(1)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    for i, (x, y, z) in enumerate([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]):
        mp.CreateNewNode(i + 1, float(x), float(y), float(z))
    prop = mp.CreateNewProperties(1)
    ids = list(range(1, 4 + 1))

    # The RIGHT name must work, or this fixture is testing nothing.
    try:
        mp.CreateNewElement("UPlSmallStrainElement2D4N", 1, ids, prop)
        print("registered_ok=UPlSmallStrainElement2D4N")
    except Exception:
        traceback.print_exc()
        print("FAIL: the CORRECT name is not registered either — "
              "the fixture cannot discriminate", file=sys.stderr)
        return 3

    # The WRONG name from the pitfall must be rejected.
    try:
        probe_name = ("UPlSmallStrainElement2D4N" if MUTATE
                      else "UPwSmallStrainElement2D4N")
        mp.CreateNewElement(probe_name, 2, ids, prop)
    except Exception as exc:
        print("rejected_name=UPwSmallStrainElement2D4N")
        print(str(exc).splitlines()[0])
        return 0
    print("FAIL: Kratos ACCEPTED UPwSmallStrainElement2D4N", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
