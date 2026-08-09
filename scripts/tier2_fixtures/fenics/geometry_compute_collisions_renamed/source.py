"""Tier-2 for fenics contact#4: the DOLFINx 0.10 geometry search functions are
not named the way older tutorials name them. The gap search a contact model
needs is compute_collisions_POINTS; plain compute_collisions is gone.

Wrong variant: dolfinx.geometry.compute_collisions(tree, points), the name used
by every pre-0.7 tutorial. It fails with
"AttributeError: module 'dolfinx.geometry' has no attribute
'compute_collisions'".

What DOES exist is checked too, because the claim is a rename and not a
removal: bb_tree is present under exactly that name, with the signature
bb_tree(mesh, dim, *, padding=0.0, entities=None) — the entities keyword being
what restricts the tree to the contact facets — and the ten working names of
the release are all present. The fixture then runs a real query through the
correct name and confirms it returns candidate cells for a point inside the
mesh.

Mutation control: T2_MUTATE=1 calls compute_collisions_points, the correct
name, so nothing is raised.
"""
from __future__ import annotations

import inspect
import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.geometry  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

WORKING = ["bb_tree", "BoundingBoxTree", "create_midpoint_tree",
           "compute_collisions_points", "compute_collisions_trees",
           "compute_colliding_cells", "compute_closest_entity",
           "squared_distance", "compute_distance_gjk",
           "determine_point_ownership"]


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)

    # bb_tree exists under that name, and takes the entities keyword.
    sig = str(inspect.signature(dolfinx.geometry.bb_tree))
    print(f"bb_tree_signature={sig}")
    print(f"bb_tree_takes_entities_kw={'entities' in sig}")
    tree = dolfinx.geometry.bb_tree(msh, tdim - 1, entities=facets)
    print(f"bb_tree_returns_BoundingBoxTree="
          f"{isinstance(tree, dolfinx.geometry.BoundingBoxTree)}")

    missing = [n for n in WORKING if not hasattr(dolfinx.geometry, n)]
    print(f"documented_working_names_missing={missing}")
    print(f"all_documented_working_names_present={not missing}")

    name = "compute_collisions_points" if MUTATE else "compute_collisions"
    pts = np.array([[0.5, 0.5, 0.0]], dtype=np.float64)
    raised = ""
    hits = None
    try:
        fn = getattr(dolfinx.geometry, name)
        cell_tree = dolfinx.geometry.bb_tree(msh, tdim)
        hits = fn(cell_tree, pts)
    except AttributeError as exc:
        raised = f"AttributeError: {exc}"
    print(f"old_name_raised={bool(raised)}")
    if raised:
        print(f"raised_text={raised}")

    # The correct name really does the job.
    cell_tree = dolfinx.geometry.bb_tree(msh, tdim)
    cand = dolfinx.geometry.compute_collisions_points(cell_tree, pts)
    n_cand = int(len(cand.links(0)))
    works = n_cand > 0
    print(f"compute_collisions_points_returns_candidates={works}")

    if raised and works and not missing:
        print("VERDICT=compute_collisions_gone_use_compute_collisions_points")
        return 0
    print("VERDICT=old_geometry_name_still_present")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
