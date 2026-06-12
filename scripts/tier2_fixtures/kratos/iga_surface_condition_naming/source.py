"""Tier-2: Kratos IgaApplication SurfaceLoadCondition naming.

Catalog under audit had:

  elements: [
    Shell3pElement,
    Shell5pElement,
    Shell5pHierarchicElement,
    SurfaceLoadCondition,         ← actually a CONDITION + needs shape
  ]

Empirically:
  Shell3pElement              registered as Element OK
  Shell5pElement              registered as Element OK
  SurfaceLoadCondition (no shape)  NOT registered
  SurfaceLoadCondition3D3N    registered as Condition (via SMA)
  SurfaceLoadCondition3D4N    same
"""
from __future__ import annotations

import sys

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication  # noqa: F401
import KratosMultiphysics.IgaApplication  # noqa: F401


def main() -> int:
    mp = KM.Model().CreateModelPart("test")
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    for i in range(1, 12):
        mp.CreateNewNode(i, float(i), 0.0, 0.0)
    props = mp.CreateNewProperties(0)

    # 'SurfaceLoadCondition' alone fails as both element & condition.
    elt_fail = False
    cond_fail = False
    try:
        mp.CreateNewElement(
            "SurfaceLoadCondition", 1, [1, 2, 3], props)
    except Exception as exc:  # noqa: BLE001
        elt_fail = "is not registered" in str(exc)
    try:
        mp.CreateNewCondition(
            "SurfaceLoadCondition", 1, [1, 2, 3], props)
    except Exception as exc:  # noqa: BLE001
        cond_fail = "is not registered" in str(exc)
    print(f"base_name_as_element_rejected={elt_fail}")
    print(f"base_name_as_condition_rejected={cond_fail}")

    # SurfaceLoadCondition with shape succeeds via SMA.
    sma_3d3n_ok = False
    try:
        mp.CreateNewCondition(
            "SurfaceLoadCondition3D3N", 1, [1, 2, 3], props)
        sma_3d3n_ok = True
    except Exception:  # noqa: BLE001
        pass
    print(f"surface_load_3D3N_condition_ok={sma_3d3n_ok}")

    sma_3d4n_ok = False
    try:
        mp.CreateNewCondition(
            "SurfaceLoadCondition3D4N", 2, [1, 2, 3, 4], props)
        sma_3d4n_ok = True
    except Exception:  # noqa: BLE001
        pass
    print(f"surface_load_3D4N_condition_ok={sma_3d4n_ok}")

    # Shell* elements register as expected.
    shell_ok = True
    for i, name in enumerate(("Shell3pElement", "Shell5pElement")):
        try:
            mp.CreateNewElement(
                name, 100 + i, [1, 2, 3], props)
        except Exception:  # noqa: BLE001
            shell_ok = False
    print(f"shell3p_and_shell5p_register={shell_ok}")

    ok = (elt_fail and cond_fail and sma_3d3n_ok and sma_3d4n_ok
          and shell_ok)
    if ok:
        return 0
    print("FAIL: IGA condition-naming invariant not held",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
