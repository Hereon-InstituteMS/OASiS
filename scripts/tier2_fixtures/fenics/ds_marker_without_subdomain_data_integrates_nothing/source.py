"""Tier-2 for fenics heat#7: ufl.ds(marker) with no subdomain_data attached
integrates over nothing and returns 0.0, with no error and no warning.

Mutation control: T2_MUTATE=1 attaches the meshtags via ufl.Measure, and the
same integral returns the real boundary length.
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
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    left = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    tags = dolfinx.mesh.meshtags(msh, fdim, np.sort(left),
                                np.full(len(left), 3, dtype=np.int32))
    one = dolfinx.fem.Constant(msh, 1.0)

    ds_tagged = ufl.Measure("ds", domain=msh, subdomain_data=tags)
    measure = ds_tagged if MUTATE else ufl.ds
    bare = float(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(one * measure(3))))
    tagged = float(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(one * ds_tagged(3))))
    print(f"bare_ds_marker_value={bare:.6e}")
    print(f"tagged_ds_marker_value={tagged:.6f}")
    print(f"bare_is_exactly_zero={bare == 0.0}")
    print(f"tagged_is_boundary_length={abs(tagged - 1.0) < 1e-12}")
    if bare == 0.0 and abs(tagged - 1.0) < 1e-12:
        print("VERDICT=untagged_ds_marker_silently_integrates_nothing")
        return 0
    print("VERDICT=untagged_ds_marker_did_something")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
