"""Tier-2: dolfinx FiniteElement.interpolation_points is PROPERTY.

The dolfinx 0.10 _version_info banner (in deep_knowledge.py)
correctly notes:

  '- element.interpolation_points is a property, not a method'

But 5 fenics generator templates in
src/backends/fenics/generators/advanced.py call it as a method:

  D_expr = fem.Expression(D_u,
            V_vis.element.interpolation_points())

That raises:

  TypeError: 'numpy.ndarray' object is not callable

The correct usage drops the parentheses:

  D_expr = fem.Expression(D_u,
            V_vis.element.interpolation_points)

Verified empirically 2026-06-01 against dolfinx 0.10.0:
  hasattr(elem, 'interpolation_points')          == True
  type(elem.interpolation_points).__name__       == 'ndarray'
  callable(elem.interpolation_points)            == False
  elem.interpolation_points.shape                == (1, 2) for DG0

Catalog claim was correct in the banner; the templates were wrong.
This is the kind of internal contradiction a catalog drift audit
catches: knowledge field says X, template field says not-X.
"""
from __future__ import annotations

import os
import sys

from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI


# Mutation control, added 2026-08-06 when this fixture was re-keyed onto the
# claim it actually verifies (nonlinear_pde#8). T2_MUTATE=1 uses the PROPERTY
# the way dolfinx 0.10 intends, so nothing is raised and the expectation
# call_as_method_raises_typeerror=True is lost. Without this the fixture passed
# whether or not the pathology existed, which is no evidence at all.
MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    m = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = fem.functionspace(m, ("DG", 0))
    elem = V.element

    has_attr = hasattr(elem, "interpolation_points")
    print(f"has_interpolation_points={has_attr}")

    ip = elem.interpolation_points
    type_name = type(ip).__name__
    print(f"interpolation_points_type={type_name}")
    print(f"interpolation_points_callable={callable(ip)}")

    # Call-as-method MUST raise TypeError 'not callable'
    raised = False
    msg = ""
    try:
        if MUTATE:
            _ = elem.interpolation_points          # the correct spelling
        else:
            elem.interpolation_points()            # the mistake
    except TypeError as exc:
        msg = str(exc)
        raised = "not callable" in msg
    print(f"call_as_method_raises_typeerror={raised}")
    print(f"call_as_method_diag_has_not_callable="
          f"{'not callable' in msg}")

    # Property access works
    shape_ok = (hasattr(ip, "shape") and ip.shape[1] == 2)
    print(f"property_shape_ok={shape_ok}")

    ok = (has_attr and type_name == "ndarray"
          and not callable(ip) and raised and shape_ok)
    if ok:
        return 0
    print("FAIL: property invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
