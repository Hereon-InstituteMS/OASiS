"""Tier-2 for fenics stokes#2: basix.ufl.element does support quadrilateral
cells for a Q2/Q1 Taylor-Hood pair — provided the cell you pass is the mesh's
own cell.

The fixture builds the quadrilateral pair with cell=msh.basix_cell() and shows
it works, then passes a TRIANGLE element to the same quadrilateral mesh and
prints what comes back.

Mutation control: T2_MUTATE=1 passes the matching cell in both places, so
nothing is rejected.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix  # noqa: E402
import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"



def main() -> int:
    msh = dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 4, 4, cell_type=dolfinx.mesh.CellType.quadrilateral)
    gdim = msh.geometry.dim
    print(f"mesh_cell={msh.basix_cell().name}")
    Q2 = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(gdim,))
    Q1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([Q2, Q1]))
    n = W.dofmap.index_map.size_global * W.dofmap.index_map_bs
    print(f"quad_taylor_hood_built=True ndofs={n}")

    wrong_cell = (msh.basix_cell() if MUTATE
                  else basix.CellType.triangle)
    msg = ""
    try:
        T1 = basix.ufl.element("Lagrange", wrong_cell, 1)
        dolfinx.fem.functionspace(msh, T1)
        print("mismatched_cell_raised=False")
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        print(f"mismatched_cell_raised=True {msg}")
    if n > 0 and msg:
        print("VERDICT=quads_supported_but_the_cell_must_match_the_mesh")
        return 0
    print("VERDICT=cell_mismatch_accepted")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
