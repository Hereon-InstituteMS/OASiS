"""Tier-2: no 2D spheric particle is registered.

Kratos DEM carries 3D kinematics internally. A 2D-sounding
element name is simply absent from the registry, so a template
that writes one fails at element construction.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.DEMApplication as DEM


# (entity_name, node_count, must_be_registered)
CASES = [('SphericParticle3D', 1, True), ('SphericParticle2D', 1, False), ('NanoParticle3D', 1, True)]

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
            mp.CreateNewElement(name, eid, ids, prop)
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
