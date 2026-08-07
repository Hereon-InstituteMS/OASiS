"""Tier-2 for fenics heat#8: mesh.exterior_facet_indices needs the facet-to-cell
connectivity to exist, and says so.

Nothing else may touch the mesh first — a prior locate_entities_boundary builds
that connectivity as a side effect and the error never appears.

Mutation control: T2_MUTATE=1 calls create_connectivity(tdim-1, tdim) first.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"



def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    tdim = msh.topology.dim
    if MUTATE:
        msh.topology.create_connectivity(tdim - 1, tdim)
    msg = ""
    try:
        n = len(dolfinx.mesh.exterior_facet_indices(msh.topology))
        print(f"raised=False n_facets={n}")
    except RuntimeError as exc:
        msg = str(exc)
        print(f"raised=True msg={msg}")
    # After create_connectivity the same call must work.
    msh.topology.create_connectivity(tdim - 1, tdim)
    n2 = len(dolfinx.mesh.exterior_facet_indices(msh.topology))
    print(f"after_create_connectivity_n_facets={n2}")
    print(f"cure_works={n2 == 16}")
    if msg:
        print("VERDICT=exterior_facet_indices_is_eager")
        return 0
    print("VERDICT=exterior_facet_indices_is_lazy")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
