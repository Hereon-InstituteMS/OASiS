"""Tier-2: ROTATION is required additionally for beams and shells.

Pitfall (kratos.linear_elasticity #4). The claim's sharp part is
WHEN the failure fires: creating the beam element and calling
Initialize does NOT raise. The error appears at the first
GetSolutionStepValue(ROTATION_X), as the generic
variables-list container error.

Mutation control: T2_MUTATE=1 builds the model part WITH ROTATION added, doing the documented correct thing, so the first ROTATION_X read succeeds instead of raising and the variables-list diagnostic never appears.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=rotation_variable_added_to_the_beam_model_part")


def build(with_rotation: bool):
    model = KM.Model()
    mp = model.CreateModelPart("beam" + str(with_rotation))
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    if with_rotation:
        mp.AddNodalSolutionStepVariable(KM.ROTATION)
    mp.SetBufferSize(2)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    mp.CreateNewNode(2, 1.0, 0.0, 0.0)
    prop = mp.CreateNewProperties(1)
    prop.SetValue(KM.YOUNG_MODULUS, 2.1e11)
    prop.SetValue(KM.POISSON_RATIO, 0.3)
    prop.SetValue(KM.DENSITY, 7850.0)
    prop.SetValue(SMA.CROSS_AREA, 0.01)
    elem = mp.CreateNewElement("CrBeamElement3D2N", 1, [1, 2], prop)
    return mp, elem


def main() -> int:
    bad = 0

    # Without ROTATION: creation succeeds, Initialize succeeds.
    mp, elem = build(with_rotation=bool(MUTATE))
    print("created_without_rotation=True")
    try:
        elem.Initialize(mp.ProcessInfo)
        print("initialize_without_rotation_raised=False")
    except Exception as exc:
        print("initialize_without_rotation_raised=True")
        print(f"FAIL: Initialize raised, contradicting the claim that it "
              f"does not: {str(exc).splitlines()[0][:120]}", file=sys.stderr)
        bad += 1

    # The failure surfaces at the first ROTATION read.
    try:
        mp.GetNode(1).GetSolutionStepValue(KM.ROTATION_X)
        print("rotation_read_raised=False")
        print("FAIL: reading ROTATION_X succeeded without the variable added",
              file=sys.stderr)
        bad += 1
    except Exception as exc:
        msg = str(exc).replace("\n", " ")
        print("rotation_read_raised=True")
        if "variables list doesn't have this variable" in msg:
            print("observed=variables_list_container_error")
        else:
            print(f"FAIL: unexpected message {msg[:150]}", file=sys.stderr)
            bad += 1

    # With ROTATION added the same read is clean — the discriminator.
    mp2, _ = build(with_rotation=True)
    try:
        val = mp2.GetNode(1).GetSolutionStepValue(KM.ROTATION_X)
        print(f"rotation_read_with_variable={val}")
    except Exception as exc:
        print(f"FAIL: ROTATION_X unreadable even after adding ROTATION: "
              f"{str(exc).splitlines()[0][:120]}", file=sys.stderr)
        bad += 1

    print(f"beam_rotation_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
