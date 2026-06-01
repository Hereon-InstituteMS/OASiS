"""Tier-2: dolfinx XDMFFile.write_function rejects P2 on P1 mesh.

Pitfall (fenics linear_elasticity#3): XDMFFile requires the
Function degree to match the underlying mesh degree. Writing
a P2 Function on a P1 mesh (the most common combination) is
rejected with:

  RuntimeError: Degree of output Function must be same as mesh
  degree. Maybe the Function needs to be interpolated?

The fix is to interpolate to a matching-degree space, or use
VTKFile / VTXWriter.
"""
from __future__ import annotations

import sys
import traceback

import dolfinx
import dolfinx.io
from mpi4py import MPI


def main() -> int:
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V_P2 = dolfinx.fem.functionspace(mesh, ('Lagrange', 2))
    u = dolfinx.fem.Function(V_P2)
    u.x.array[:] = 1.0
    path = "/tmp/_tier2_xdmf_degree_mismatch.xdmf"
    try:
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, path, "w") as f:
            f.write_mesh(mesh)
            f.write_function(u)
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: XDMFFile.write_function accepted P2 on P1 mesh "
          "(catalog claim wrong)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
