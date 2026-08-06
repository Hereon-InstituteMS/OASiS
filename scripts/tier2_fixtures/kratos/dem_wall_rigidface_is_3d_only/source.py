"""Tier-2: RigidFace walls are 3D conditions; no 2D spelling exists.

Kratos DEM is 3D internally. The wall entities are registered
as CONDITIONS named RigidFace3D3N / RigidFace3D4N; there is no
2D RigidFace and no RigidEdge.

Mutation control: T2_MUTATE=1 INVERTS the expected registration status of every entity --
it claims the opposite of what this build registers, while still really calling
CreateNewElement/CreateNewCondition for each name. Each registered[<name>]=<got>_expected=<must>
line then disagrees with itself and registry_mismatches rises from 0 to the number of
entities. This proves the booleans come from a real registry lookup on this build and
that a wrong claim is caught.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=every_entity_registration_expectation_inverted")
import KratosMultiphysics.DEMApplication  # noqa: F401


# (entity_name, node_count, must_be_registered)
CASES = [('RigidFace3D3N', 3, True), ('RigidFace3D4N', 4, True), ('RigidFace2D2N', 2, False), ('RigidEdge3D2N', 2, False)]

_PTS = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0),
        (0.5, 0.0, 0.0), (1.0, 0.5, 0.0), (0.5, 1.0, 0.0), (0.0, 0.5, 0.0),
        (0.5, 0.5, 0.0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.0), (0.75, 0.25, 0.0),
        (0.5, 0.0, 1.0), (1.0, 0.5, 1.0), (0.5, 1.0, 1.0), (0.0, 0.5, 1.0),
        (0.0, 0.0, 0.5), (1.0, 0.0, 0.5), (1.0, 1.0, 0.5), (0.0, 1.0, 0.5),
        (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.5, 0.5, 1.0)]


def main() -> int:
    model = KM.Model()
    mp = model.CreateModelPart("t")
    mp.SetBufferSize(1)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    for i, (x, y, z) in enumerate(_PTS):
        mp.CreateNewNode(i + 1, x, y, z)
    prop = mp.CreateNewProperties(1)

    wrong = 0
    eid = 1
    for name, nnodes, must_exist in CASES:
        if MUTATE:
            # Pathology injected: claim the opposite
            # registration status; the lookup still runs.
            must_exist = not must_exist
        eid += 1
        ids = list(range(1, nnodes + 1))
        try:
            mp.CreateNewCondition(name, eid, ids, prop)
            got = True
            err = ""
        except Exception as exc:
            got = False
            err = str(exc).splitlines()[0].strip()
        print(f"registered[{name}]={got}_expected={must_exist}")
        if got != must_exist:
            wrong += 1
            print(f"FAIL: {name} registered={got} expected={must_exist} "
                  f"{err[:160]}", file=sys.stderr)
        elif not got:
            # Record the real message text so the fixture pins the signal,
            # not just the boolean.
            print(f"  message: {err[:150]}")
    print(f"registry_mismatches={wrong}")
    return 0 if wrong == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
