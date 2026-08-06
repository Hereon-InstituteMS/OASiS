"""Tier-2: SlidingCableElement3D3N requires CONSTITUTIVE_LAW on Properties.

The element's Init() errors by name when the law is missing.
A 1D law such as TrussConstitutiveLaw satisfies it. This is
unique to SlidingCable among the cable_net elements — Ring and
EmpiricalSpring read properties directly.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.CableNetApplication as CN



def _build(with_law: bool):
    model = KM.Model()
    mp = model.CreateModelPart("sc" + str(with_law))
    for v in (KM.DISPLACEMENT, KM.REACTION):
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(2)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    for i, (x, y, z) in enumerate([(0.0, 0.0, 0.0), (1.0, 0.2, 0.0),
                                   (2.0, 0.0, 0.0)]):
        mp.CreateNewNode(i + 1, x, y, z)
    prop = mp.CreateNewProperties(1)
    prop.SetValue(KM.YOUNG_MODULUS, 2.1e11)
    prop.SetValue(KM.DENSITY, 7850.0)
    prop.SetValue(SMA.CROSS_AREA, 1e-4)
    if with_law:
        prop.SetValue(KM.CONSTITUTIVE_LAW, SMA.TrussConstitutiveLaw())
    el = mp.CreateNewElement("SlidingCableElement3D3N", 1, [1, 2, 3], prop)
    return mp, el


def main() -> int:
    bad = 0

    mp, el = _build(with_law=False)
    print("created_without_constitutive_law=True")
    try:
        el.Initialize(mp.ProcessInfo)
        print("initialize_without_law_raised=False")
        print("FAIL: Initialize accepted a SlidingCable with no "
              "CONSTITUTIVE_LAW", file=sys.stderr)
        bad += 1
    except Exception as exc:
        msg = str(exc).replace("\n", " ")
        print("initialize_without_law_raised=True")
        if "constitutive law needs to be specified for the element" in msg:
            print("observed=constitutive_law_needed_for_element")
        else:
            print(f"FAIL: unexpected message {msg[:150]}", file=sys.stderr)
            bad += 1

    mp2, el2 = _build(with_law=True)
    try:
        el2.Initialize(mp2.ProcessInfo)
        print("initialize_with_law_raised=False")
    except Exception as exc:
        print("initialize_with_law_raised=True")
        print(f"FAIL: a 1D TrussConstitutiveLaw did not satisfy the element: "
              f"{str(exc).splitlines()[0][:120]}", file=sys.stderr)
        bad += 1

    print(f"sliding_cable_law_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
