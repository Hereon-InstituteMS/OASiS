"""Tier-2: FEniCSx (dolfinx) rejects unknown element family.

basix raises ValueError("Unknown element family: ...") when the
family string in the FunctionSpace constructor is a typo or an
unsupported family for the cell type. The pitfall family covers
all "wrong element family name" typos a community user might
make.

Mutation control: T2_MUTATE=1 spells a family basix DOES register
('Lagrange'), so the space is built, nothing raises, and the fixture
prints its dimension instead of a traceback — both expected strings
('ValueError', 'Unknown element family') disappear.
"""
from __future__ import annotations

import os
import sys
import traceback

from mpi4py import MPI

from dolfinx.fem import functionspace
from dolfinx.mesh import CellType, create_unit_square

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4, CellType.triangle)
    family = "Lagrange" if MUTATE else "NotAFamily"
    print(f"requested_family={family}")
    try:
        V = functionspace(mesh, (family, 1))
    except Exception:
        traceback.print_exc()
        return 1
    dim = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    print(f"family_accepted=True space_dim={dim}")
    if MUTATE:
        return 0
    print("ERROR: dolfinx accepted unknown element family",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
