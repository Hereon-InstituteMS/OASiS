"""Tier-2: DofsView is not a mapping, but len() works so it looks like one.

Claim: skfem heat#4 -- basis.get_dofs() returns a DofsView that is NOT
subscriptable by string; ib.get_dofs()['left'] raises TypeError with 'DofsView'
and 'not subscriptable' in the message. The correct API passes the tag name
positionally.

This fixture goes past the single subscript probe and maps every dict-like
operation an agent might reach for, because len() succeeding is exactly what
makes the object look like a mapping and sends people down this path.

Mutation control: T2_MUTATE=1 applies the documented fix at every probe site --
the mapping-style accesses on the DofsView are replaced by the supported form
(the tag passed positionally to basis.get_dofs, the tag dict on the mesh, the
DofsView flattened).  Nothing raises any more, so the recorded error text is
gone and "'DofsView' object is not subscriptable", "no attribute 'keys'" and
"not iterable" disappear from the output.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from skfem import Basis, ElementQuad1, MeshQuad

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    ok = True
    m = MeshQuad.init_tensor(
        np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5),
    ).with_boundaries({
        "left": lambda x: x[0] < 1e-10,
        "right": lambda x: x[0] > 1.0 - 1e-10,
    })
    basis = Basis(m, ElementQuad1())
    dv = basis.get_dofs()
    print(f"dofsview_type_name={type(dv).__name__}")
    print(f"dofsview_module={type(dv).__module__}")
    if type(dv).__name__ != "DofsView":
        print(f"FAIL: get_dofs() returned {type(dv).__name__}", file=sys.stderr)
        ok = False

    # --- WRONG variants: every mapping-style access ---------------------
    # Under mutation each probe is rewritten in the documented correct form:
    # the tag goes to get_dofs positionally, the tag dict lives on the mesh,
    # and the DOF numbers come out of flatten().
    if not MUTATE:
        probes = {
            "subscript": lambda: dv["left"],
            "keys": lambda: dv.keys(),
            "get": lambda: dv.get("left"),
            "contains": lambda: "left" in dv,
            "iter": lambda: list(iter(dv)),
        }
    else:
        probes = {
            "subscript": lambda: basis.get_dofs("left"),
            "keys": lambda: basis.get_dofs("left").nodal.keys(),
            "get": lambda: basis.get_dofs("left"),
            "contains": lambda: "left" in m.boundaries,
            "iter": lambda: list(dv.flatten()),
        }
    messages = {}
    for label, fn in probes.items():
        msg = ""
        try:
            fn()
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
        messages[label] = msg
        print(f"{label}_raises={bool(msg)}")
        print(f"{label}_msg={msg!r}")
        if not msg:
            print(f"FAIL: DofsView.{label} succeeded, so it now behaves like a "
                  f"mapping and the claim is stale", file=sys.stderr)
            ok = False

    if "not subscriptable" not in messages["subscript"]:
        print(f"FAIL: subscript message changed: "
              f"{messages['subscript']!r}", file=sys.stderr)
        ok = False
    if "no attribute 'keys'" not in messages["keys"]:
        print(f"FAIL: keys message changed: {messages['keys']!r}",
              file=sys.stderr)
        ok = False

    # len() works -- this is the trap's bait.
    n = len(dv)
    matches = n == len(dv.flatten())
    print(f"len_value={n}")
    print(f"len_works_and_matches_flatten={matches}")
    if not matches:
        print(f"FAIL: len(dv)={n} but flatten() has {len(dv.flatten())}",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant --------------------------------------------------
    left = basis.get_dofs("left")
    print(f"positional_call_returns_dofs={len(left.flatten()) == 5}")
    print(f"nodal_keys={list(left.nodal)}")
    if len(left.flatten()) != 5:
        print(f"FAIL: positional get_dofs('left') gave "
              f"{len(left.flatten())} DOFs, expected 5", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
