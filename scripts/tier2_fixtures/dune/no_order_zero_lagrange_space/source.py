"""Tier-2: there is no order-0 Lagrange space (poisson#12).

The FEniCS habit of asking for a ('DG', 0) / order-0 space for a
piecewise-constant field dies here with a KeyError whose message is a
parameter-validation sentence, not a Python one. The replacements are
dglagrange(gridView, order=0) and finiteVolume(gridView), and the claim
is that both have exactly one dof per CELL.

Verified by execution against dune-fem 2.12.0.2.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                           # noqa: E402
from dune.fem.space import (lagrange, dglagrange,               # noqa: E402
                            finiteVolume)


def main() -> int:
    fail: list[str] = []
    gridView = structuredGrid([0, 0], [1, 1], [4, 4])
    n_cells = gridView.size(0)
    print(f"cells={n_cells}")

    try:
        lagrange(gridView, order=0)
        print("lagrange_order0_rejected=False")
        fail.append("lagrange(gridView, order=0) was ACCEPTED; the "
                    "order>=1 guard has gone")
    except Exception as exc:                                 # noqa: BLE001
        msg = " ".join(str(exc).split())
        print(f"lagrange_order0_rejected={type(exc).__name__}")
        print(f"lagrange_order0_message={msg[:200]}")
        if type(exc).__name__ != "KeyError":
            fail.append(f"the rejection is a {type(exc).__name__}, not "
                        f"the KeyError the claim names")
        for needle in ("order=0", "greater or equal to 1"):
            if needle not in msg:
                fail.append(f"rejection message no longer contains "
                            f"{needle!r}: {msg[:200]}")

    for label, space in (("dglagrange0", dglagrange(gridView, order=0)),
                         ("finiteVolume", finiteVolume(gridView))):
        print(f"{label}_size={space.size} one_dof_per_cell="
              f"{space.size == n_cells}")
        if space.size != n_cells:
            fail.append(f"{label} has {space.size} dofs on a "
                        f"{n_cells}-cell grid; the claim is one dof per "
                        f"cell, i.e. the order-0 replacement")

    if not fail:
        print("dune_no_order_zero_lagrange_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
