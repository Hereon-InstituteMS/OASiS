"""Tier-2: FEniCSx Function.x.array size mismatch.

Mutation control: T2_MUTATE=1 hands the slice a CORRECTLY sized source
array. The assignment then succeeds, nothing raises, and the fixture
prints the resulting array length and checksum instead of a traceback —
so both expected strings ('ValueError', 'broadcast input array')
disappear.
"""
from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from mpi4py import MPI

from dolfinx.fem import Function, functionspace
from dolfinx.mesh import CellType, create_unit_square

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4, CellType.triangle)
    V = functionspace(mesh, ("Lagrange", 1))
    u = Function(V)
    n = V.dofmap.index_map.size_local
    # 2x too large — numpy raises on the slice assignment.
    # MUTATE: exactly n, the length the slice actually wants.
    src_len = n if MUTATE else 2 * n
    print(f"dofs_local={n} source_len={src_len}")
    try:
        u.x.array[:] = np.arange(src_len, dtype=u.x.array.dtype)
    except Exception:
        traceback.print_exc()
        return 1
    print(f"assignment_accepted=True array_len={u.x.array.size} "
          f"array_sum={float(u.x.array.sum()):.1f}")
    if MUTATE:
        return 0
    print("ERROR: array slice accepted wrong-size source",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
