"""Tier-2 for fenics poisson#0: which boundary idiom still needs
create_connectivity on dolfinx 0.10.0.

The claim has two halves that pull in opposite directions, which is the point of
the entry:

  * locate_entities_boundary / locate_dofs_topological build the connectivity
    lazily, so omitting mesh.topology.create_connectivity(fdim, tdim) does NOT
    raise any more (it did pre-0.7);
  * dolfinx.mesh.exterior_facet_indices(mesh.topology) is still EAGER and
    raises on a fresh mesh.

ORDER IS LOAD-BEARING, and getting it wrong hides the second half entirely: the
first draft of this fixture ran the lazy path first, that call built the
facet-to-cell connectivity as a side effect, and exterior_facet_indices then
returned 16 facets without complaint. Each half therefore gets its OWN fresh
mesh here.

Mutation control: T2_MUTATE=1 calls create_connectivity(fdim, tdim) on the
eager mesh first; exterior_facet_indices then succeeds and the pathology is
gone.
"""
from __future__ import annotations

import os

import numpy as np
from mpi4py import MPI

import dolfinx

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    # Half one: the eager idiom, on a mesh nothing has touched yet.
    eager_msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    tdim = eager_msh.topology.dim
    fdim = tdim - 1
    if MUTATE:
        eager_msh.topology.create_connectivity(fdim, tdim)
    raised = False
    msg = ""
    try:
        idx = dolfinx.mesh.exterior_facet_indices(eager_msh.topology)
        print(f"eager_exterior_facet_indices_raised=False n={len(idx)}")
    except RuntimeError as exc:
        raised = True
        msg = str(exc)
        print(f"eager_exterior_facet_indices_raised=True msg={msg}")

    # Half two: the lazy idiom, on a SEPARATE mesh so the probe above cannot
    # have built its connectivity.
    lazy_msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(lazy_msh, ("Lagrange", 1))
    facets = dolfinx.mesh.locate_entities_boundary(
        lazy_msh, fdim, lambda x: np.full(x.shape[1], True))
    dofs = dolfinx.fem.locate_dofs_topological(V, fdim, facets)
    print(f"lazy_locate_dofs_raised=False n_dofs={len(dofs)}")
    print(f"lazy_path_ok={len(dofs) > 0}")

    if raised:
        print("VERDICT=eager_needs_connectivity_lazy_does_not")
        return 0
    print("VERDICT=eager_path_also_lazy")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
