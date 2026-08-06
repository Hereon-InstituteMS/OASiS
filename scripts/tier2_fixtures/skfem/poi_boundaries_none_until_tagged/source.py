"""Tier-2: m.boundaries is None until with_boundaries is REASSIGNED.

Claim: skfem poisson#2 -- basis.get_dofs(name) only works once the name is
registered via m = m.with_boundaries({...}); m.boundaries is None on a freshly
constructed mesh and get_dofs('left') raises ValueError("Boundary 'left' not
found.").

Wrong variant: calling m.with_boundaries(...) without rebinding m. The method
returns a new mesh and leaves the original untagged, so the very next
get_dofs('left') still fails -- this fixture proves the non-mutation directly
rather than only quoting the error.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import Basis, ElementQuad1, MeshQuad


def main() -> int:
    ok = True
    m = MeshQuad.init_tensor(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    print(f"fresh_boundaries_is_none={m.boundaries is None}")
    if m.boundaries is not None:
        print(f"FAIL: a fresh tensor mesh already carries boundaries "
              f"{m.boundaries!r}", file=sys.stderr)
        ok = False

    # --- WRONG variant (a): query an untagged mesh ----------------------
    basis = Basis(m, ElementQuad1())
    raised = ""
    try:
        basis.get_dofs("left")
    except ValueError as exc:
        raised = str(exc)
    print(f"untagged_get_dofs_msg={raised!r}")
    if "not found" not in raised:
        print(f"FAIL: expected a 'not found' ValueError, got {raised!r}",
              file=sys.stderr)
        ok = False

    # --- WRONG variant (b): call with_boundaries without rebinding ------
    tagged = m.with_boundaries({"left": lambda x: x[0] < 1e-10})
    print(f"with_boundaries_returns_new_mesh={tagged is not m}")
    print(f"original_still_untagged={m.boundaries is None}")
    if tagged is m or m.boundaries is not None:
        print("FAIL: with_boundaries mutated the original mesh, so the "
              "reassignment trap does not exist", file=sys.stderr)
        ok = False

    still_raises = ""
    try:
        Basis(m, ElementQuad1()).get_dofs("left")
    except ValueError as exc:
        still_raises = str(exc)
    print(f"unrebound_still_raises={bool(still_raises)}")
    if not still_raises:
        print("FAIL: the un-rebound mesh resolved 'left' anyway",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant --------------------------------------------------
    good = Basis(tagged, ElementQuad1())
    n_left = len(good.get_dofs("left").flatten())
    print(f"left_edge_dof_count={n_left}")
    print(f"tagged_boundaries_keys={sorted(tagged.boundaries)}")
    if n_left != 5:
        print(f"FAIL: the left edge of a 5x5 ElementQuad1 mesh has {n_left} "
              f"DOFs, expected 5", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
