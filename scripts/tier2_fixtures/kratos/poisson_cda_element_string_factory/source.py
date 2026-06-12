"""Tier-2: ConvectionDiffusionApplication element-name access surface.

The catalog lists element types as 'LaplacianElement2D3N/3D4N' and
'EulerianConvDiff2D3N/3D4N'. These are real Kratos elements, but
their access mechanism is non-obvious: they are registered at the
C++ level and reachable ONLY via the string-based factory call
  model_part.CreateNewElement("LaplacianElement2D3N", id, nodes, props)
They are NOT Python attributes on the CDA module
  hasattr(CDA, "LaplacianElement2D3N") is False
so users who try
  CDA.LaplacianElement2D3N(...)   # AttributeError
are following a reasonable-looking but wrong access pattern.

This fixture asserts:
  * CDA imports (forces the ModuleNotFoundError to surface if
    KratosConvectionDiffusionApplication is missing — was missing
    until 2026-06-01 install)
  * The Laplacian and EulerianConvDiff element names work via
    CreateNewElement
  * Those names are NOT Python attributes on CDA
  * Wrong-named element (e.g. ConvDiff2D3N, no Eulerian prefix)
    is rejected with 'is not registered!' from Kratos.
"""
from __future__ import annotations

import sys

import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication as CDA  # noqa: F401


def main() -> int:
    # (1) Module attribute check: names are NOT Python attributes
    for name in ("LaplacianElement2D3N", "LaplacianElement3D4N",
                 "EulerianConvDiff2D3N", "EulerianConvDiff3D4N"):
        has_attr = hasattr(CDA, name)
        print(f"attr_check_{name}={has_attr}")
        if has_attr:
            print(f"ERROR: {name} unexpectedly is a Python "
                  f"attribute on CDA", file=sys.stderr)
            return 2

    # (2) CreateNewElement string factory check
    mp = KM.Model().CreateModelPart("test")
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    mp.CreateNewNode(2, 1.0, 0.0, 0.0)
    mp.CreateNewNode(3, 0.0, 1.0, 0.0)
    mp.CreateNewNode(4, 0.0, 0.0, 1.0)
    props = mp.CreateNewProperties(0)
    next_id = 1
    ok_factory: list[str] = []
    for name, nodes in (("LaplacianElement2D3N", [1, 2, 3]),
                        ("LaplacianElement3D4N", [1, 2, 3, 4]),
                        ("EulerianConvDiff2D3N", [1, 2, 3]),
                        ("EulerianConvDiff3D4N", [1, 2, 3, 4])):
        try:
            mp.CreateNewElement(name, next_id, nodes, props)
            ok_factory.append(name)
            next_id += 1
            print(f"factory_ok_{name}=True")
        except Exception as exc:  # noqa: BLE001
            print(f"factory_ok_{name}=False: {type(exc).__name__}")
            return 2

    # (3) Wrong name is rejected
    wrong_caught = False
    err_msg = ""
    try:
        mp.CreateNewElement("ConvDiff2D3N", next_id, [1, 2, 3], props)
    except Exception as exc:  # noqa: BLE001
        err_msg = str(exc)
        wrong_caught = ("is not registered" in err_msg
                        or "is not register" in err_msg)
    print(f"wrong_name_rejected={wrong_caught}")
    if not wrong_caught:
        print(f"ERROR: wrong name not rejected as expected — "
              f"err_msg={err_msg[:200]}", file=sys.stderr)
        return 2

    if (len(ok_factory) == 4 and wrong_caught):
        return 0
    print("ERROR: factory/reject coverage incomplete",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
