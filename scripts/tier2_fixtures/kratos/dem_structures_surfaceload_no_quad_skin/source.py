"""Tier-2: only the triangle DEM surface-load condition exists.

SkinDetectionProcess3D on a hexahedral mesh produces quad skin
facets. There is no 4-node DEM surface-load condition to put on
them, so the DEM->FEM load transfer cannot be wired. Mesh with
tetrahedra.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication  # noqa: F401
import KratosMultiphysics.DEMApplication  # noqa: F401
import KratosMultiphysics.DemStructuresCouplingApplication  # noqa: F401


# (entity_name, node_count, must_be_registered)
CASES = [('SurfaceLoadFromDEMCondition3D3N', 3, True), ('LineLoadFromDEMCondition2D2N', 2, True), ('SurfaceLoadFromDEMCondition3D4N', 4, False)]

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
