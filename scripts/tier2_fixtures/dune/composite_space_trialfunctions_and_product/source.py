"""Tier-2: the multi-field space, and the dolfinx unpacking idiom that
silently binds the wrong thing.

  stokes#0          ufl.TrialFunctions(W) does NOT split a dune-fem
                    composite space: it returns a 1-tuple whose single
                    entry has the FULL shape, so (u, p) = ... binds u to
                    the whole vector and the error surfaces elsewhere.
  mixed_methods#2   the same defect stated for the flux/potential pair.
  mixed_methods#4   the factory is product() or composite(); there is no
                    dune.fem.space.product_space, and product and
                    composite were measured to produce the same object
                    for the same arguments.

No weak form is built, so nothing here compiles a scheme.

Verified by execution against dune-fem 2.12.0.2.
"""
from __future__ import annotations

import importlib
import sys
import warnings

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                           # noqa: E402
import dune.fem.space as dspace                                # noqa: E402
from ufl import (TrialFunction, TrialFunctions,                 # noqa: E402
                 TestFunctions, as_vector)


def main() -> int:
    fail: list[str] = []
    gridView = structuredGrid([0, 0], [1, 1], [4, 4])

    # ── mixed_methods#4: the factory names ───────────────────────────
    print(f"has_product={hasattr(dspace, 'product')}")
    print(f"has_composite={hasattr(dspace, 'composite')}")
    print(f"has_product_space={hasattr(dspace, 'product_space')}")
    if hasattr(dspace, "product_space"):
        fail.append("dune.fem.space.product_space exists after all; the "
                    "catalog records it as absent/falsified")
    try:
        importlib.import_module("dune.fem.space.product_space")
        print("product_space_import_raises=False")
        fail.append("dune.fem.space.product_space is importable")
    except ImportError as exc:
        print(f"product_space_import_raises={type(exc).__name__}")

    legs = (dspace.lagrange(gridView, order=2, dimRange=2),
            dspace.lagrange(gridView, order=1))
    W_prod = dspace.product(*legs)
    W_comp = dspace.composite(*legs)
    same = (W_prod.dimRange == W_comp.dimRange
            and W_prod.size == W_comp.size)
    print(f"product_dimRange_size={W_prod.dimRange},{W_prod.size}")
    print(f"composite_dimRange_size={W_comp.dimRange},{W_comp.size}")
    print(f"product_and_composite_agree={same}")
    if not same:
        fail.append(f"product and composite no longer agree: "
                    f"{W_prod.dimRange},{W_prod.size} vs "
                    f"{W_comp.dimRange},{W_comp.size}")

    # ── stokes#0 / mixed_methods#2: the unpacking trap ───────────────
    W = W_comp
    trials = TrialFunctions(W)
    tests = TestFunctions(W)
    print(f"trialfunctions_len={len(trials)}")
    print(f"trialfunctions_shapes={[t.ufl_shape for t in trials]}")
    print(f"testfunctions_len={len(tests)}")
    if len(trials) != 1:
        fail.append(f"TrialFunctions(W) returned {len(trials)} entries; "
                    f"the claim is a 1-TUPLE, which is what makes the "
                    f"dolfinx idiom bind the wrong object")
    if trials[0].ufl_shape != (W.dimRange,):
        fail.append(f"the single entry has shape "
                    f"{trials[0].ufl_shape}, not ({W.dimRange},)")

    # This is the silent part: the unpacking SUCCEEDS for a 1-tuple
    # target, so nothing complains at the point of the mistake.
    try:
        (whole,) = TrialFunctions(W)
        print(f"one_tuple_unpacks_without_error={whole.ufl_shape}")
    except Exception as exc:                                 # noqa: BLE001
        print(f"one_tuple_unpacks_without_error=ERROR:"
              f"{type(exc).__name__}")
        fail.append(f"unpacking a 1-tuple raised {exc!r}")

    # …and the two-name unpacking the dolfinx idiom asks for fails with
    # a ValueError about counts, not about spaces.
    try:
        (u_bad, p_bad) = TrialFunctions(W)
        print("two_name_unpack_rejected=False")
        fail.append("(u, p) = TrialFunctions(W) succeeded; the claim is "
                    "that the tuple has only one entry")
    except ValueError as exc:
        print(f"two_name_unpack_rejected={type(exc).__name__}")
        print(f"two_name_unpack_message={' '.join(str(exc).split())[:120]}")

    # ── the working spelling: TrialFunction + slice ──────────────────
    t = TrialFunction(W)
    u = as_vector([t[0], t[1]])
    p = t[2]
    print(f"sliced_velocity_shape={u.ufl_shape}")
    print(f"sliced_pressure_shape={p.ufl_shape}")
    if u.ufl_shape != (2,) or p.ufl_shape != ():
        fail.append(f"slicing TrialFunction(W) gave shapes "
                    f"{u.ufl_shape} / {p.ufl_shape}; the documented "
                    f"workaround expects (2,) and ()")

    if not fail:
        print("dune_composite_space_unpacking_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
