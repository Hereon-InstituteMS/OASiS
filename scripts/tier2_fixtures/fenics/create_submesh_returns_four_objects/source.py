"""Tier-2 for fenics stokes_darcy#1: dolfinx.mesh.create_submesh has the
signature create_submesh(msh, dim, entities) and returns FOUR objects in 0.10 -
(Mesh, EntityMap, EntityMap, numpy int32 array), i.e. the sub mesh, the entity
map, the vertex map and the geometry node map.

Wrong variant: the three-value unpacking that older code and older tutorials use,
`sub, cell_map, vertex_map = mesh.create_submesh(...)`. Right variant: unpack
four.

Observed on dolfinx 0.10.0: the three-value unpacking raises
"ValueError: too many values to unpack (expected 3)". The two middle returns are
dolfinx.mesh.EntityMap objects whose public members are dim, sub_topology,
sub_topology_to_topology and topology; they are NOT index arrays, so
`cell_map[0]` raises "TypeError: 'EntityMap' object is not subscriptable".
The annotated return type reads
tuple[Mesh, EntityMap, EntityMap, npt.NDArray[np.int32]].

Mutation control: T2_MUTATE=1 unpacks four values (the correct call), so the
ValueError text and the unpack-failure token never appear.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import inspect  # noqa: E402

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dolfinx import mesh  # noqa: E402


def main() -> int:
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    tdim = msh.topology.dim
    cells = mesh.locate_entities(msh, tdim, lambda x: x[1] <= 0.5 + 1e-12)

    sig = str(inspect.signature(mesh.create_submesh))
    print(f"create_submesh_signature={sig}")

    raised = ""
    if MUTATE:
        # Correct call for dolfinx 0.10: four returns.
        sub, cell_map, vertex_map, geom_map = mesh.create_submesh(
            msh, tdim, cells)
        print("mutation=unpacking_four_values_as_0_10_requires")
    else:
        try:
            sub, cell_map, vertex_map = mesh.create_submesh(msh, tdim, cells)
            print("three_value_unpack_succeeded=True")
        except ValueError as exc:
            raised = f"{type(exc).__name__}: {exc}"
            print(f"three_value_unpack -> {raised}")
        sub, cell_map, vertex_map, geom_map = mesh.create_submesh(
            msh, tdim, cells)

    out = (sub, cell_map, vertex_map, geom_map)
    types = [type(o).__name__ for o in out]
    print(f"n_returns={len(out)} types={types}")
    members = sorted(a for a in dir(cell_map) if not a.startswith("_"))
    print(f"entity_map_public_members={members}")

    sub_raised = ""
    try:
        cell_map[0]
        print("entity_map_is_subscriptable=True")
    except TypeError as exc:
        sub_raised = f"{type(exc).__name__}: {exc}"
        print(f"entity_map_indexing -> {sub_raised}")

    n_ok = len(out) == 4
    types_ok = types == ["Mesh", "EntityMap", "EntityMap", "ndarray"]
    members_ok = members == ["dim", "sub_topology", "sub_topology_to_topology",
                             "topology"]
    print(f"create_submesh_returns_four={n_ok}")
    print(f"return_types_are_mesh_entitymap_entitymap_ndarray={types_ok}")
    print(f"three_value_unpack_raised_valueerror={bool(raised)}")
    print(f"entity_maps_are_not_index_arrays={bool(sub_raised)}")
    print(f"entity_map_members_are_the_documented_four={members_ok}")
    if n_ok and types_ok and raised and sub_raised and members_ok:
        print("VERDICT=create_submesh_returns_four_and_three_unpack_fails")
        return 0
    print("VERDICT=create_submesh_still_returns_three")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
