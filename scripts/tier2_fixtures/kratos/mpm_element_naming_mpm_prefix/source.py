"""Tier-2: Kratos MPM element names require the 'MPM' prefix.

Catalog under audit (kratos mpm physics):
  elements.2D = ["UpdatedLagrangianPQ2D", "UpdatedLagrangianPQ2D4N"]
  elements.3D = ["UpdatedLagrangianPQ3D8N"]
  elements.axisymmetric = ["UpdatedLagrangianAxisym"]

All four catalog-listed names FAIL CreateNewElement with
'is not registered'. The real registered names from
KratosMPMApplication all start with the 'MPM' prefix:

  MPMUpdatedLagrangian
  MPMUpdatedLagrangian2D3N
  MPMUpdatedLagrangian2D4N
  MPMUpdatedLagrangian3D4N
  MPMUpdatedLagrangian3D8N
  MPMUpdatedLagrangianAxisymmetry2D3N
  MPMUpdatedLagrangianAxisymmetry2D4N
  MPMUpdatedLagrangianPQ
  MPMUpdatedLagrangianUP / MPMUpdatedLagrangianUP2D3N

An LLM agent pasting any catalog-listed base name into an
.mdpa or model_part.CreateNewElement call hits 'is not
registered' from kratos/python/add_model_part_to_python.cpp.

Verified empirically 2026-06-01 (Kratos 10.4.2 +
KratosMPMApplication 10.4.2).
"""
from __future__ import annotations

import sys

import KratosMultiphysics as KM
import KratosMultiphysics.MPMApplication  # noqa: F401


def main() -> int:
    mp = KM.Model().CreateModelPart("test")
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    mp.CreateNewNode(2, 1.0, 0.0, 0.0)
    mp.CreateNewNode(3, 1.0, 1.0, 0.0)
    mp.CreateNewNode(4, 0.0, 1.0, 0.0)
    mp.CreateNewNode(5, 0.0, 0.0, 1.0)
    mp.CreateNewNode(6, 1.0, 0.0, 1.0)
    mp.CreateNewNode(7, 1.0, 1.0, 1.0)
    mp.CreateNewNode(8, 0.0, 1.0, 1.0)
    props = mp.CreateNewProperties(0)

    # (1) Catalog-listed base names — none should work.
    base = [
        "UpdatedLagrangianPQ2D",
        "UpdatedLagrangianPQ2D4N",
        "UpdatedLagrangianPQ3D8N",
        "UpdatedLagrangianAxisym",
    ]
    all_base_rejected = True
    for name in base:
        ok_reject = False
        try:
            n = 8 if "3D8N" in name else 4
            mp.CreateNewElement(name, len(mp.Elements) + 1,
                                list(range(1, n + 1)), props)
        except Exception as exc:  # noqa: BLE001
            ok_reject = "is not registered" in str(exc)
        if not ok_reject:
            all_base_rejected = False
    print(f"all_base_names_rejected={all_base_rejected}")

    # (2) Corrected names — must succeed.
    corrected = [
        ("MPMUpdatedLagrangian2D4N", 4),
        ("MPMUpdatedLagrangian2D3N", 3),
        ("MPMUpdatedLagrangian3D4N", 4),
        ("MPMUpdatedLagrangian3D8N", 8),
        ("MPMUpdatedLagrangianAxisymmetry2D4N", 4),
        ("MPMUpdatedLagrangianAxisymmetry2D3N", 3),
    ]
    all_corrected_ok = True
    for name, n in corrected:
        try:
            mp.CreateNewElement(name, len(mp.Elements) + 1,
                                list(range(1, n + 1)), props)
        except Exception as exc:  # noqa: BLE001
            print(f"corrected_{name}_fail={exc}", file=sys.stderr)
            all_corrected_ok = False
    print(f"all_corrected_mpm_prefix_ok={all_corrected_ok}")

    if all_base_rejected and all_corrected_ok:
        return 0
    print("FAIL: MPM naming invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
