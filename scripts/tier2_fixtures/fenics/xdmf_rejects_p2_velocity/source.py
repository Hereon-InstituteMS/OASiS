"""Tier-2 for fenics navier_stokes#9: XDMFFile.write_function only accepts a
Function whose degree matches the mesh geometry degree, so a P2 velocity cannot
be written directly.

Mutation control: T2_MUTATE=1 interpolates into a matching P1 vector space
first, which is one of the two documented cures.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"



def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    gdim = msh.geometry.dim
    print(f"mesh_geometry_degree={msh.geometry.cmap.degree}")
    V2 = dolfinx.fem.functionspace(msh, ("Lagrange", 2, (gdim,)))
    u = dolfinx.fem.Function(V2)
    u.interpolate(lambda x: np.vstack([x[1], -x[0]]))
    if MUTATE:
        V1 = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
        u1 = dolfinx.fem.Function(V1)
        u1.interpolate(u)
        u = u1
    print(f"function_degree={u.function_space.ufl_element().degree}")
    path = os.path.join(tempfile.mkdtemp(prefix="xdmf_ns_"), "u.xdmf")
    msg = ""
    with dolfinx.io.XDMFFile(msh.comm, path, "w") as xf:
        xf.write_mesh(msh)
        try:
            xf.write_function(u)
            print("write_function_raised=False")
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"write_function_raised=True {msg}")
    if msg:
        print("VERDICT=p2_velocity_cannot_go_straight_into_xdmf")
        return 0
    print("VERDICT=xdmf_accepted_p2")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
