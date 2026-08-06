"""Tier-2: mesh.Curve(order) fixes the geometry error of a curved surface -
and Curve(1) buys nothing, because a fresh mesh is already at curve order 1.

Claim: ngsolve surface_pde#3 - mesh.Curve(order) improves the geometry
approximation.  On the unit sphere at maxh=0.3 the surface area against the
exact 4*pi is ~2% low without curving, Curve(1) is a no-op, and Curve(2..5)
drive the error down monotonically.

Wrong variant: mesh a curved surface and either skip mesh.Curve entirely or
  call mesh.Curve(1) believing it curves anything.  Both leave a ~2%
  geometry error in every ds integral.

Observed on NGSolve 6.2.2604, OCCGeometry(Sphere(r=1)), maxh=0.3, ne=422
(2026-08-06), Integrate(1*ds) against 4*pi = 12.56637061:
  no Curve  12.31844056  (-1.973%)
  Curve(1)  12.31844056  (-1.973%, bitwise identical - a no-op)
  Curve(2)  12.56426008  (-1.680e-04)
  Curve(3)  12.56657695  (+1.642e-05)
  Curve(4)  12.56637398  (+2.680e-07)
  Curve(5)  12.56637022  (-3.117e-08)
The mechanism behind the Curve(1) surprise: mesh.GetCurveOrder() returns 1
on a freshly generated mesh, so Curve(1) asks for what is already there.
(mesh.Curve(0) segfaults NGSolve 6.2.2604 and is not exercised here.)
"""
from __future__ import annotations

import math
import sys

from netgen.occ import OCCGeometry, Pnt, Sphere
from ngsolve import CoefficientFunction, Integrate, Mesh, ds

EXACT = 4.0 * math.pi


def area_for(geo, order):
    mesh = Mesh(geo.GenerateMesh(maxh=0.3))
    if order is not None:
        mesh.Curve(order)
    return (Integrate(CoefficientFunction(1) * ds, mesh),
            mesh.GetCurveOrder(), mesh.ne)


def main() -> int:
    ok = True
    geo = OCCGeometry(Sphere(Pnt(0, 0, 0), r=1.0))

    a_none, co_none, ne = area_for(geo, None)
    print(f"sphere_mesh_ne={ne}")
    print(f"fresh_mesh_curve_order={co_none}")

    areas, orders = {}, {}
    for k in (1, 2, 3, 4, 5):
        areas[k], orders[k], _ = area_for(geo, k)

    def rel(a):
        return abs(a - EXACT) / EXACT

    print(f"nocurve_rel_err_gt_1pct={rel(a_none) > 0.01}")
    if not rel(a_none) > 0.01:
        print(f"FAIL: the uncurved sphere area is already accurate, "
              f"rel_err={rel(a_none):.3e}", file=sys.stderr)
        ok = False

    print("curve1_area_bitwise_identical_to_nocurve="
          f"{areas[1] == a_none}")
    print(f"curve1_get_curve_order={orders[1]}")
    print(f"curve1_rel_err_gt_1pct={rel(areas[1]) > 0.01}")
    if areas[1] != a_none:
        print(f"FAIL: Curve(1) changed the area from {a_none:.8f} to "
              f"{areas[1]:.8f}; it is no longer the documented no-op",
              file=sys.stderr)
        ok = False

    print(f"curve2_rel_err_lt_1em3={rel(areas[2]) < 1e-3}")
    print(f"curve3_rel_err_lt_1em4={rel(areas[3]) < 1e-4}")
    print(f"curve5_rel_err_lt_1em6={rel(areas[5]) < 1e-6}")
    print("curve_error_decreases_2_to_5="
          f"{rel(areas[2]) > rel(areas[3]) > rel(areas[5])}")
    if not (rel(areas[2]) < 1e-3 and rel(areas[3]) < 1e-4
            and rel(areas[5]) < 1e-6):
        print(f"FAIL: curving did not deliver the documented accuracy: "
              f"C2={rel(areas[2]):.3e} C3={rel(areas[3]):.3e} "
              f"C5={rel(areas[5]):.3e}", file=sys.stderr)
        ok = False
    if not rel(areas[2]) > rel(areas[3]) > rel(areas[5]):
        print("FAIL: the geometry error is not monotone in the curve order",
              file=sys.stderr)
        ok = False

    print("curve2_beats_nocurve_by_gt_10x="
          f"{rel(a_none) / rel(areas[2]) > 10}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
