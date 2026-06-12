"""Tier-2: Kratos contact conditions need 'Condition' suffix AND shape.

Probed via mcp__open-fem-agent__prepare_simulation(kratos, contact):
the catalog lists four contact 'conditions':

  ALMFrictionlessMortarContact
  ALMFrictionalMortarContact
  PenaltyFrictionlessMortarContact
  PenaltyFrictionalMortarContact

None of these names work as the eletype string passed to
model_part.CreateNewCondition. The real registered names from
KratosContactStructuralMechanicsApplication have:

  (1) the 'Condition' suffix (NOT just the base name)
  (2) a shape descriptor (2D2N / 3D3N / 3D4N / 3D4N3N / etc.)

So the actual usable strings are e.g.:

  ALMFrictionlessMortarContactCondition2D2N
  ALMFrictionalMortarContactCondition2D2N
  PenaltyFrictionlessMortarContactCondition2D2N
  PenaltyFrictionalMortarContactCondition2D2N
  ALMFrictionlessMortarContactCondition3D3N
  ...

This fixture asserts:
  * The catalog-listed base names (no suffix) FAIL with
    'is not registered' at CreateNewCondition.
  * The corrected names (Condition + 2D2N / 3D3N) succeed.

Also surfaces the install gap: KratosContactStructuralMechanics
Application was not in the .venv until installed during this
probe (2026-06-01).
"""
from __future__ import annotations

import sys

import KratosMultiphysics as KM
import KratosMultiphysics.ContactStructuralMechanicsApplication  # noqa: F401


def main() -> int:
    mp = KM.Model().CreateModelPart("test")
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    mp.CreateNewNode(2, 1.0, 0.0, 0.0)
    mp.CreateNewNode(3, 0.0, 1.0, 0.0)
    props = mp.CreateNewProperties(0)

    # (1) The catalog-listed BASE names should all fail.
    base_names = [
        "ALMFrictionlessMortarContact",
        "ALMFrictionalMortarContact",
        "PenaltyFrictionlessMortarContact",
        "PenaltyFrictionalMortarContact",
    ]
    all_base_rejected = True
    for name in base_names:
        ok_reject = False
        try:
            mp.CreateNewCondition(name, len(mp.Conditions) + 1,
                                  [1, 2], props)
        except Exception as exc:  # noqa: BLE001
            ok_reject = "is not registered" in str(exc)
        if not ok_reject:
            all_base_rejected = False
            print(f"base_rejected_{name}=False", file=sys.stderr)
    print(f"all_base_names_rejected={all_base_rejected}")

    # (2) Corrected 2D2N names should succeed.
    corrected_2d = [
        "ALMFrictionlessMortarContactCondition2D2N",
        "ALMFrictionalMortarContactCondition2D2N",
        "PenaltyFrictionlessMortarContactCondition2D2N",
        "PenaltyFrictionalMortarContactCondition2D2N",
    ]
    all_2d_ok = True
    for name in corrected_2d:
        try:
            mp.CreateNewCondition(name, len(mp.Conditions) + 1,
                                  [1, 2], props)
        except Exception:  # noqa: BLE001
            all_2d_ok = False
    print(f"all_corrected_2D_ok={all_2d_ok}")

    # (3) Corrected 3D3N triangular surface name.
    name_3d = "ALMFrictionlessMortarContactCondition3D3N"
    try:
        mp.CreateNewCondition(name_3d, len(mp.Conditions) + 1,
                              [1, 2, 3], props)
        ok_3d = True
    except Exception:  # noqa: BLE001
        ok_3d = False
    print(f"corrected_3D3N_ok={ok_3d}")

    if all_base_rejected and all_2d_ok and ok_3d:
        return 0
    print("FAIL: contact-condition naming invariant not held",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
