"""Tier-2 for fenics heat#6: fem.Constant is a constant. A callable or a
SpatialCoordinate expression is rejected; a spatially varying property has to be
a fem.Function on its own space.

Mutation control: T2_MUTATE=1 interpolates the same law into a fem.Function,
which is the documented cure, and nothing raises.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"



def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    law = lambda x: np.where(x[0] < 0.5, 1.0, 10.0)  # noqa: E731

    callable_err = ""
    coord_err = ""
    int_ok = None
    if not MUTATE:
        try:
            dolfinx.fem.Constant(msh, law)
            print("constant_from_callable_raised=False")
        except Exception as exc:
            callable_err = f"{type(exc).__name__}: {exc}"
            print(f"constant_from_callable_raised=True {callable_err}")
        try:
            dolfinx.fem.Constant(msh, ufl.SpatialCoordinate(msh))
            print("constant_from_spatialcoordinate_raised=False")
        except Exception as exc:
            coord_err = f"{type(exc).__name__}: {exc}"
            print(f"constant_from_spatialcoordinate_raised=True {coord_err}")
        try:
            dolfinx.fem.Constant(msh, 1)
            int_ok = True
        except Exception as exc:
            int_ok = False
            print(f"constant_from_python_int_error={type(exc).__name__}: {exc}")
        print(f"constant_from_python_int_accepted={int_ok}")

    # The cure: a Function on its own space, which assembles.
    Q = dolfinx.fem.functionspace(msh, ("DG", 0))
    k = dolfinx.fem.Function(Q)
    k.interpolate(law)
    v = ufl.TestFunction(dolfinx.fem.functionspace(msh, ("Lagrange", 1)))
    total = float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(k * ufl.dx)))
    del v
    print(f"function_route_integral={total:.6f}")
    print(f"function_route_works={abs(total - 5.5) < 1e-9}")

    if MUTATE:
        print("VERDICT=function_route_only")
        return 1
    if callable_err and coord_err:
        print("VERDICT=constant_rejects_callable_and_expression")
        return 0
    print("VERDICT=constant_accepted_non_constant")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
