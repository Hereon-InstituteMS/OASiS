"""Tier-2: FEniCSx UFL curl() on scalar space raises."""
from __future__ import annotations

import sys
import traceback

import ufl
from mpi4py import MPI

from dolfinx.fem import form, functionspace
from dolfinx.mesh import CellType, create_unit_square


def main() -> int:
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4, CellType.triangle)
    V = functionspace(mesh, ("Lagrange", 1))  # scalar
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    try:
        # curl on scalar → undefined; UFL catches via rank check.
        form(ufl.curl(u) * ufl.curl(v) * ufl.dx)
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: UFL accepted curl() on scalar space",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
