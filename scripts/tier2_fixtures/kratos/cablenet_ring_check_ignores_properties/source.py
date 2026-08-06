"""Tier-2: RingElement3D3N accepts empty Properties and returns zero stiffness.

The claim is that this element's Check verifies only the id, the
current length and the node count — not CROSS_AREA or
YOUNG_MODULUS, which it later reads. So an empty Properties
object gets through and the stiffness comes out zero with no
error, which is what makes the Newton failure unactionable.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.CableNetApplication  # noqa: F401


ELEMENT = "RingElement3D3N"


def _ring(props: dict):
    model = KM.Model()
    mp = model.CreateModelPart("rg" + str(len(props)))
    for v in (KM.DISPLACEMENT, KM.REACTION):
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(2)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    for i, (x, y, z) in enumerate([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                                   (1.0, 1.0, 0.0)]):
        mp.CreateNewNode(i + 1, x, y, z)
    prop = mp.CreateNewProperties(1)
    for k, v in props.items():
        prop.SetValue(k, v)
    el = mp.CreateNewElement(ELEMENT, 1, [1, 2, 3], prop)
    return mp, el


def _stiffness(props: dict):
    mp, el = _ring(props)
    el.Initialize(mp.ProcessInfo)
    K = KM.Matrix(9, 9)
    el.CalculateLeftHandSide(K, mp.ProcessInfo)
    return max(abs(K[i, j]) for i in range(9) for j in range(9))


FULL = {KM.YOUNG_MODULUS: 2.1e11, KM.DENSITY: 7850.0, SMA.CROSS_AREA: 1e-4}


def main() -> int:
    bad = 0

    # Empty Properties: no CROSS_AREA, no YOUNG_MODULUS, no DENSITY.
    try:
        empty_k = _stiffness({})
        print("initialize_with_empty_properties_raised=False")
        print(f"empty_properties_max_abs_K={empty_k:.6e}")
    except Exception as exc:
        print("initialize_with_empty_properties_raised=True")
        print(f"FAIL: the element rejected empty Properties, so it does NOT "
              f"have the weak check this pitfall describes: "
              f"{str(exc).splitlines()[0][:120]}", file=sys.stderr)
        return 2

    # The consequence the claim names: k_0 = 0, silently.
    print(f"empty_properties_stiffness_is_exactly_zero={empty_k == 0.0}")
    if empty_k != 0.0:
        print(f"FAIL: expected an exactly-zero stiffness matrix from empty "
              f"Properties, got {empty_k:.6e}", file=sys.stderr)
        bad += 1

    full_k = _stiffness(FULL)
    print(f"full_properties_stiffness_is_nonzero={full_k > 0.0}")
    if full_k <= 0.0:
        print(f"FAIL: a fully specified {ELEMENT} also gave zero stiffness, "
              f"so the fixture is not discriminating", file=sys.stderr)
        bad += 1

    print(f"ring_check_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
