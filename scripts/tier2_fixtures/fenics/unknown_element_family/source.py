"""Tier-2: FEniCSx (dolfinx) rejects unknown element family.

basix raises ValueError("Unknown element family: ...") when the
family string in the FunctionSpace constructor is a typo or an
unsupported family for the cell type. The pitfall family covers
all "wrong element family name" typos a community user might
make.
"""
from __future__ import annotations

import sys
import traceback

from mpi4py import MPI

from dolfinx.fem import functionspace
from dolfinx.mesh import CellType, create_unit_square


def main() -> int:
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4, CellType.triangle)
    try:
        functionspace(mesh, ("NotAFamily", 1))
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: dolfinx accepted unknown element family",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
