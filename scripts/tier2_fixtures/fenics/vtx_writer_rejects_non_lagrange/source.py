"""Tier-2 for fenics poisson#2: VTXWriter rejects non-Lagrange families, at
CONSTRUCTION time.

The claim carries an exact-text assertion and an explicit denial of two older
strings, so the fixture prints the message it actually got and checks the
denial too.

Mutation control: T2_MUTATE=1 writes a Lagrange Function instead of an
N1curl one; construction succeeds.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix.ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    if MUTATE:
        el = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    else:
        el = basix.ufl.element("N1curl", msh.basix_cell(), 1)
    V = dolfinx.fem.functionspace(msh, el)
    u = dolfinx.fem.Function(V)
    print(f"family={el.family_name}")

    out = os.path.join(tempfile.mkdtemp(prefix="vtx_t2_"), "u.bp")
    stage = ""
    msg = ""
    try:
        w = dolfinx.io.VTXWriter(msh.comm, out, [u])
        stage = "construction_ok"
        try:
            w.write(0.0)
            w.close()
            stage = "write_ok"
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            stage = "raised_at_write"
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        stage = "raised_at_construction"

    print(f"stage={stage}")
    print(f"message={msg}")
    old1 = "Cannot interpolate function to the VTX output basis"
    old2 = "ADIOS2 VTX only supports Lagrange elements"
    print(f"old_string_1_present={old1 in msg}")
    print(f"old_string_2_present={old2 in msg}")
    if stage == "raised_at_construction":
        print("VERDICT=vtx_rejects_non_lagrange_at_construction")
        return 0
    print("VERDICT=vtx_accepted_non_lagrange")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
