"""Tier-2 for fenics linear_elasticity#7: XDMFFile.write_function rejects a
Function whose degree does not match the mesh degree.

Wrong variant: write a P2 Function on a P1 mesh. The claim also denies an older
wording ('XDMF mesh must be P1'), so the fixture prints the message it actually
received and checks that older text is absent.

Mutation control: T2_MUTATE=1 interpolates into a P1 space first, which is the
documented cure; the write succeeds.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    print(f"mesh_geometry_degree={msh.geometry.cmap.degree}")
    V2 = dolfinx.fem.functionspace(msh, ("Lagrange", 2))
    u = dolfinx.fem.Function(V2)
    u.interpolate(lambda x: x[0] * x[0])
    if MUTATE:
        V1 = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
        u1 = dolfinx.fem.Function(V1)
        u1.interpolate(u)
        u = u1

    path = os.path.join(tempfile.mkdtemp(prefix="xdmf_t2_"), "u.xdmf")
    msg = ""
    with dolfinx.io.XDMFFile(msh.comm, path, "w") as xf:
        xf.write_mesh(msh)
        try:
            xf.write_function(u)
            print("write_function_raised=False")
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"write_function_raised=True {msg}")
    print(f"old_wording_present={'XDMF mesh must be P1' in msg}")
    if msg:
        print("VERDICT=xdmf_requires_matching_degree")
        return 0
    print("VERDICT=xdmf_accepted_mismatched_degree")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
