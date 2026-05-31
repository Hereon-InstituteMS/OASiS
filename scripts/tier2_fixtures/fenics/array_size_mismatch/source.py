"""Tier-2: FEniCSx Function.x.array size mismatch."""
from __future__ import annotations

import sys
import traceback

import numpy as np
from mpi4py import MPI

from dolfinx.fem import Function, functionspace
from dolfinx.mesh import CellType, create_unit_square


def main() -> int:
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4, CellType.triangle)
    V = functionspace(mesh, ("Lagrange", 1))
    u = Function(V)
    n = V.dofmap.index_map.size_local
    # 2x too large — numpy raises on the slice assignment.
    try:
        u.x.array[:] = np.zeros(2 * n)
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: array slice accepted wrong-size source",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
