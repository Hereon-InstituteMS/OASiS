"""Tier-2: generic isotropic plasticity Check() required properties.

Pitfall (kratos.constitutive_laws #4). The law's Check() names
missing properties one at a time. Measured ordering matters:
HARDENING_CURVE is demanded FIRST, before the yield stresses and
the fracture energy, which is not the order the claim lists.

Mutation control: T2_MUTATE=1 changes the material deck handed to the law,
never the probe. Two things go away at once. (1) The extra properties curves
2, 4 and 6 demand are SUPPLIED (MAXIMUM_STRESS/-_POSITION,
CURVE_FITTING_PARAMETERS/TANGENCY_REGION2/PLASTIC_STRAIN_INDICATORS, and the
two PLASTICITY_POINT_CURVE vectors), so those three Check() diagnostics stop
being emitted and curve[2|4|6]_demands=... disappear. (2) The out-of-range
runs additionally set POISSON_RATIO = 0.6, a bound the SAME Check() call
really does enforce ("POISSON_RATIO is above the upper bound 0.5"), so both
come back FAILED and out_of_range_curve[7|8]_check=PASSED disappear. Part (2)
is the load-bearing half: it proves PASSED is a verdict read back from
Kratos's own Check() on this build, not a string this fixture always prints.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication  # noqa: F401
import KratosMultiphysics.ConstitutiveLawsApplication as CLA

MUTATE = os.environ.get("T2_MUTATE") == "1"

_PTS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
_BASE = {KM.YOUNG_MODULUS: 2.1e11, KM.POISSON_RATIO: 0.3, KM.DENSITY: 7850.0}


def _vec(values):
    v = KM.Vector(len(values))
    for i, x in enumerate(values):
        v[i] = x
    return v


def _demanded_extras():
    """The properties curves 2, 4 and 6 ask for — withheld unless mutating.

    Withholding them is what makes the curve[N]_demands=... diagnostics
    appear, so supplying them is the antidote to that half of the claim.
    """
    if not MUTATE:
        return {}
    get = KM.KratosGlobals.GetVariable
    return {
        get("MAXIMUM_STRESS"): 3.0e8,
        get("MAXIMUM_STRESS_POSITION"): 0.5,
        CLA.CURVE_FITTING_PARAMETERS: _vec([0.0, 1.0, 0.0, 0.0]),
        CLA.TANGENCY_REGION2: True,
        CLA.PLASTIC_STRAIN_INDICATORS: _vec([0.0, 0.0]),
        CLA.EQUIVALENT_STRESS_VECTOR_PLASTICITY_POINT_CURVE:
            _vec([2.5e8, 3.0e8]),
        CLA.PLASTIC_STRAIN_VECTOR_PLASTICITY_POINT_CURVE:
            _vec([0.0, 0.01]),
    }


# A property this same Check() really does bound-check, added to the
# out-of-range runs only when mutating.
_A_BOUNDED_PROPERTY = {KM.POISSON_RATIO: 0.6} if MUTATE else {}


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
    if MUTATE:
        print("mutation=demanded_extras_supplied_and_out_of_range_runs_given_"
              "a_property_kratos_really_bounds")
    extras = _demanded_extras()
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
        msg = check_with({**common, HC: curve, **extras})
        print(f"curve[{curve}]_check={'FAILED' if msg else 'PASSED'}")
        if msg is None or needed not in msg:
            print(f"FAIL: curve {curve} did not demand {needed}: {msg}",
                  file=sys.stderr)
            bad += 1
        else:
            print(f"curve[{curve}]_demands={needed}")

    # Out-of-range values pass with no error at all — the defect itself.
    for curve in (7, 8):
        msg = check_with({**common, HC: curve, **_A_BOUNDED_PROPERTY})
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
