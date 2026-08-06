"""Tier-2: generic isotropic plasticity Check() required properties.

Pitfall (kratos.constitutive_laws #4). The law's Check() names
missing properties one at a time. Measured ordering matters:
HARDENING_CURVE is demanded FIRST, before the yield stresses and
the fracture energy, which is not the order the claim lists.

Mutation control: T2_MUTATE=1 gives every Check() step the FULL property set instead of withholding one property at a time, removing the missing-property condition. Check then passes at every step and reports nothing, so the ordered step[i]_reported=<PROPERTY> lines disappear.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication  # noqa: F401
import KratosMultiphysics.ConstitutiveLawsApplication as CLA

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=every_check_step_given_the_full_property_set")

_PTS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
_BASE = {KM.YOUNG_MODULUS: 2.1e11, KM.POISSON_RATIO: 0.3, KM.DENSITY: 7850.0}


def check_with(extra):
    model = KM.Model()
    mp = model.CreateModelPart("c" + str(id(model)))
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    mp.SetBufferSize(2)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    for i, (x, y, z) in enumerate(_PTS):
        mp.CreateNewNode(i + 1, float(x), float(y), float(z))
    prop = mp.CreateNewProperties(1)
    for k, v in {**_BASE, **extra}.items():
        prop.SetValue(k, v)
    law = CLA.SmallStrainIsotropicPlasticity3DVonMisesVonMises()
    prop.SetValue(KM.CONSTITUTIVE_LAW, law)
    el = mp.CreateNewElement("SmallDisplacementElement3D8N", 1,
                             list(range(1, 9)), prop)
    el.Initialize(mp.ProcessInfo)
    try:
        law.Check(prop, el.GetGeometry(), mp.ProcessInfo)
        return None
    except Exception as exc:
        return str(exc).strip().splitlines()[0]


def main() -> int:
    bad = 0
    YT, YC = CLA.YIELD_STRESS_TENSION, CLA.YIELD_STRESS_COMPRESSION
    FE, HC = KM.FRACTURE_ENERGY, CLA.HARDENING_CURVE

    # Check() names ONE missing property at a time, in a fixed order.
    steps = [
        ({}, "HARDENING_CURVE"),
        ({HC: 0}, "FRACTURE_ENERGY"),
        ({HC: 0, FE: 1e10}, "YIELD_STRESS_TENSION"),
        ({HC: 0, FE: 1e10, YT: 2.5e8}, "YIELD_STRESS_COMPRESSION"),
    ]
    full_props = {HC: 0, FE: 1e10, YT: 2.5e8, YC: 2.5e8}
    for i, (props, expected) in enumerate(steps):
        msg = check_with(full_props if MUTATE else props)
        if msg is None:
            print(f"step[{i}]_reported=NONE")
            print(f"FAIL: step {i} passed Check; expected it to demand "
                  f"{expected}", file=sys.stderr)
            bad += 1
            continue
        if f"{expected} is not a defined value" in msg:
            print(f"step[{i}]_reported={expected}")
        else:
            print(f"step[{i}]_reported=OTHER")
            print(f"FAIL: step {i} expected {expected}, got {msg[:120]}",
                  file=sys.stderr)
            bad += 1

    # All four present -> Check passes.
    msg = check_with({HC: 0, FE: 1e10, YT: 2.5e8, YC: 2.5e8})
    print(f"all_four_present_check={'FAILED' if msg else 'PASSED'}")
    if msg is not None:
        print(f"FAIL: Check still fails with all four set: {msg[:130]}",
              file=sys.stderr)
        bad += 1

    # THE CLAIM SAYS plain KM.YIELD_STRESS is NOT accepted. It is.
    msg = check_with({HC: 0, FE: 1e10, KM.YIELD_STRESS: 2.5e8})
    print(f"plain_yield_stress_accepted={msg is None}")
    if msg is not None:
        print(f"FAIL: plain KM.YIELD_STRESS was rejected — the catalog "
              f"correction recorded in this fixture would itself be wrong: "
              f"{msg[:110]}", file=sys.stderr)
        bad += 1

    print(f"cl_check_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
