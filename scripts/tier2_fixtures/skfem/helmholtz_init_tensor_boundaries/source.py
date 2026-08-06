"""Tier-2: MeshQuad.init_tensor attaches no named boundaries; two distinct errors.

Claim: skfem helmholtz#8 — ``MeshQuad.init_tensor`` (and most other ``init_*``)
does NOT attach named boundaries, so ``ib.get_dofs('left')`` raises
``ValueError("Boundary 'left' not found.")`` while the legacy subscript form
``ib.get_dofs()['left']`` raises ``TypeError: 'DofsView' object is not
subscriptable``.  The fix is
``MeshQuad.init_tensor(...).with_boundaries({'left': lambda x: x[0] < 1e-10, ...})``
followed by ``ib.get_dofs('left').flatten()``.

Wrong variant: both call patterns are executed on an untagged
``MeshQuad.init_tensor(9x9)`` / ``ElementQuad1`` basis and the two exception
types plus the library's own wording are printed; then the documented fix is
run and its DOF indices checked against the exact left-edge node count.

FALSIFIED sub-claim, pinned here so a regression is caught: the claim also says
"Same constraint applies after .to_meshtri() — boundaries must be reattached on
the triangulated mesh."  On skfem 12.0.1 ``.to_meshtri()`` PRESERVES the four
tags and ``get_dofs('left')`` keeps working, so no reattachment is needed.

Observed on skfem 12.0.1 (2026-08-06).
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import Basis, ElementQuad1, ElementTriP1, MeshQuad

NX = 8  # 9x9 tensor grid -> 9 nodes on the left edge


def main() -> int:
    ok = True
    xs = np.linspace(0.0, 1.0, NX + 1)

    # --- WRONG variant: no with_boundaries() ---------------------------------
    m_raw = MeshQuad.init_tensor(xs, xs)
    ib_raw = Basis(m_raw, ElementQuad1())
    print(f"mesh_class={type(m_raw).__name__}")
    print(f"init_tensor_boundaries_is_none={m_raw.boundaries is None}")

    try:
        ib_raw.get_dofs("left")
        name_exc, name_msg = "", ""
    except Exception as exc:                     # noqa: BLE001 - want the type
        name_exc, name_msg = type(exc).__name__, str(exc)
    print(f"named_lookup_exc={name_exc}")
    print(f"named_lookup_msg={name_msg}")

    try:
        ib_raw.get_dofs()["left"]
        sub_exc, sub_msg = "", ""
    except Exception as exc:                     # noqa: BLE001 - want the type
        sub_exc, sub_msg = type(exc).__name__, str(exc)
    print(f"subscript_exc={sub_exc}")
    print(f"subscript_msg={sub_msg}")
    print(f"dofsview_class={type(ib_raw.get_dofs()).__name__}")

    two_distinct = (name_exc == "ValueError" and sub_exc == "TypeError")
    print(f"two_distinct_errors={two_distinct}")
    if not two_distinct:
        print("FAIL: the untagged mesh did not produce the documented pair "
              f"(named={name_exc!r}, subscript={sub_exc!r})", file=sys.stderr)
        ok = False
    if "not found" not in name_msg:
        print(f"FAIL: ValueError wording changed: {name_msg!r}",
              file=sys.stderr)
        ok = False
    if "not subscriptable" not in sub_msg:
        print(f"FAIL: TypeError wording changed: {sub_msg!r}", file=sys.stderr)
        ok = False

    # --- RIGHT variant: the canonical incantation ----------------------------
    m = MeshQuad.init_tensor(xs, xs).with_boundaries({
        "left": lambda x: x[0] < 1e-10,
        "right": lambda x: x[0] > 1.0 - 1e-10,
        "bottom": lambda x: x[1] < 1e-10,
        "top": lambda x: x[1] > 1.0 - 1e-10,
    })
    ib = Basis(m, ElementQuad1())
    print(f"tagged_boundaries={','.join(sorted(m.boundaries))}")
    left = ib.get_dofs("left").flatten()
    print(f"n_left_dofs={left.size}")
    good = (left.size == NX + 1
            and bool(np.all(m.p[0][left] < 1e-10)))
    print(f"fixed_lookup_ok={good}")
    if not good:
        print("FAIL: with_boundaries + get_dofs('left').flatten() did not "
              "return the left-edge nodes", file=sys.stderr)
        ok = False

    # --- The to_meshtri() half of the claim ---------------------------------
    mt = m.to_meshtri()
    keeps = mt.boundaries is not None and "left" in mt.boundaries
    print(f"to_meshtri_class={type(mt).__name__}")
    print(f"to_meshtri_preserves_boundaries={keeps}")
    if keeps:
        n_tri_left = Basis(mt, ElementTriP1()).get_dofs("left").flatten().size
        print(f"n_left_dofs_after_to_meshtri={n_tri_left}")
        if n_tri_left != NX + 1:
            print("FAIL: to_meshtri kept the tag but lost left-edge DOFs",
                  file=sys.stderr)
            ok = False
    else:
        print("FAIL: to_meshtri dropped the boundary tags; skfem 12.0.1 "
              "preserved them", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
