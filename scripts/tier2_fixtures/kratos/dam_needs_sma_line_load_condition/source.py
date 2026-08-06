"""Tier-2: there is no Dam-specific 2D pressure condition.

Without a load condition the nodal pressure is set and never
assembled: rc=0, zero displacement, zero reactions. The
condition that does the work belongs to StructuralMechanics.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.PoromechanicsApplication as PORO
import KratosMultiphysics.DamApplication as DAM


# (entity_name, node_count, must_be_registered)
CASES = [('LineLoadCondition2D2N', 2, True), ('DamLineLoadCondition2D2N', 2, False), ('HydroLineLoadCondition2D2N', 2, False)]

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
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
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
