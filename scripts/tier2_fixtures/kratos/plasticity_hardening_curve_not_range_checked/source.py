"""Tier-2: generic isotropic plasticity Check() required properties.

Pitfall (kratos.constitutive_laws #4). The law's Check() names
missing properties one at a time. Measured ordering matters:
HARDENING_CURVE is demanded FIRST, before the yield stresses and
the fracture energy, which is not the order the claim lists.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication  # noqa: F401
import KratosMultiphysics.ConstitutiveLawsApplication as CLA

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
    common = {YT: 2.5e8, YC: 2.5e8, FE: 1e10}

    # Curves that need nothing beyond the common four.
    for curve in (0, 1, 3, 5):
        msg = check_with({**common, HC: curve})
        print(f"curve[{curve}]_check={'FAILED' if msg else 'PASSED'}")
        if msg is not None:
            print(f"FAIL: curve {curve} rejected: {msg[:120]}", file=sys.stderr)
            bad += 1

    # THREE curves demand an extra property, not two as the claim states.
    for curve, needed in ((2, "MAXIMUM_STRESS"),
                          (4, "CURVE_FITTING_PARAMETERS"),
                          (6, "EQUIVALENT_STRESS_VECTOR_PLASTICITY_POINT_CURVE")):
        msg = check_with({**common, HC: curve})
        print(f"curve[{curve}]_check={'FAILED' if msg else 'PASSED'}")
        if msg is None or needed not in msg:
            print(f"FAIL: curve {curve} did not demand {needed}: {msg}",
                  file=sys.stderr)
            bad += 1
        else:
            print(f"curve[{curve}]_demands={needed}")

    # Out-of-range values pass with no error at all — the defect itself.
    for curve in (7, 8):
        msg = check_with({**common, HC: curve})
        print(f"out_of_range_curve[{curve}]_check="
              f"{'FAILED' if msg else 'PASSED'}")
        if msg is not None:
            print(f"FAIL: out-of-range curve {curve} WAS rejected, so the "
                  f"claim that the enum is unbounded would be false: "
                  f"{msg[:110]}", file=sys.stderr)
            bad += 1

    print(f"hardening_curve_range_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
