"""Tier-2: Kratos RANS + ShallowWater element naming.

Catalog drift caught by scripts/kratos_eletype_scanner.py.

RANS catalog had:
  RansKEpsilonKElement2D3N
  RansKEpsilonEpsilonElement2D3N
  RansKOmegaSSTKElement2D3N
  RansKOmegaSSTOmegaElement2D3N

None registered. Real RANS names DROP the 'Element' suffix and
add a stabilization-scheme tag in the middle:

  RansKEpsilonK{AFC,CWD,RFC}2D3N        — k transport, k-eps
  RansKEpsilonEpsilon{AFC,CWD,RFC}2D3N  — eps transport, k-eps
  RansKOmegaK{AFC,CWD,RFC}2D3N          — k transport, k-omega
  RansKOmegaOmega{AFC,CWD,RFC}2D3N      — omega transport, k-omega
  RansKOmegaSSTK{AFC,CWD,RFC}2D3N       — k-omega SST
  RansKOmegaSSTOmega{AFC,CWD,RFC}2D3N

ShallowWaterApplication catalog had:
  ShallowWaterElement2D3N
  ShallowWaterElement2D4N

Neither registered. The real registered name is
  BoussinesqElement2D{3,4}N

This fixture asserts both ends:
  * Catalog-listed wrong names fail with 'is not registered'.
  * Corrected names register successfully.
"""
from __future__ import annotations

import sys

import KratosMultiphysics as KM
import KratosMultiphysics.FluidDynamicsApplication  # required by RANS
import KratosMultiphysics.RANSApplication  # noqa: F401
import KratosMultiphysics.ShallowWaterApplication  # noqa: F401


def main() -> int:
    mp = KM.Model().CreateModelPart("probe")
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    mp.CreateNewNode(2, 1.0, 0.0, 0.0)
    mp.CreateNewNode(3, 0.0, 1.0, 0.0)
    mp.CreateNewNode(4, 1.0, 1.0, 0.0)
    props = mp.CreateNewProperties(0)

    wrong_names = [
        "RansKEpsilonKElement2D3N",
        "RansKEpsilonEpsilonElement2D3N",
        "RansKOmegaSSTKElement2D3N",
        "RansKOmegaSSTOmegaElement2D3N",
        "ShallowWaterElement2D3N",
        "ShallowWaterElement2D4N",
    ]
    all_wrong_rejected = True
    for n in wrong_names:
        ok = False
        try:
            mp.CreateNewElement(n, len(mp.Elements) + 1, [1, 2, 3], props)
        except Exception as exc:  # noqa: BLE001
            ok = "is not registered" in str(exc)
        if not ok:
            all_wrong_rejected = False
    print(f"all_catalog_wrong_names_rejected={all_wrong_rejected}")

    # Correct names — sample one of each
    correct = [
        ("RansKEpsilonKAFC2D3N", 3),
        ("RansKEpsilonEpsilonCWD2D3N", 3),
        ("RansKOmegaSSTKRFC2D3N", 3),
        ("BoussinesqElement2D3N", 3),
        ("BoussinesqElement2D4N", 4),
    ]
    all_correct_ok = True
    for n, count in correct:
        try:
            mp.CreateNewElement(n, len(mp.Elements) + 1,
                                list(range(1, count + 1)), props)
        except Exception as exc:  # noqa: BLE001
            print(f"correct_{n}_fail={exc}", file=sys.stderr)
            all_correct_ok = False
    print(f"all_corrected_names_ok={all_correct_ok}")

    if all_wrong_rejected and all_correct_ok:
        return 0
    print("FAIL: rans/sw naming invariant not held",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
