"""Tier-2: dune-fem-dg and dune-vem are NOT part of a dune-fem install.

The catalog used to advertise "Comprehensive DG methods via dune-fem-dg"
and "VEM (Virtual Element Method) support" as capabilities of this
backend. Neither module is importable from a conda-forge dune-fem
2.12.0.2 environment, so any plan built on dune-fem-dg's SSP Runge-Kutta
steppers, its Bassi-Rebay / CDG operators or its limiters, or on VEM
spaces, fails at the first import.

This fixture is deliberately cheap: no grid, no space, no JIT. It asserts
what is importable and what is not, so the claim cannot drift back in
without the gate turning red.

Verified by execution against dune-fem 2.12.0.2 on 2026-08-03.
"""
from __future__ import annotations

import importlib
import sys
import warnings

warnings.filterwarnings("ignore")


def importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:
        return False


def main() -> int:
    fail = []

    import dune.fem                                        # noqa: F401
    print(f"dune_fem_importable={importable('dune.fem')}")

    # present
    for mod, key in (("dune.alugrid", "dune_alugrid"),
                     ("dune.fem.view", "dune_fem_view"),
                     ("dune.fem.utility", "dune_fem_utility"),
                     ("dune.fem.operator", "dune_fem_operator")):
        ok = importable(mod)
        print(f"{key}_importable={ok}")
        if not ok:
            fail.append(f"{mod} unexpectedly absent")

    # absent
    for mod, key in (("dune.femdg", "dune_femdg"),
                     ("dune.fem.dg", "dune_fem_dg"),
                     ("dune.vem", "dune_vem"),
                     ("dune.fem.solver", "dune_fem_solver"),
                     ("dune.fem.parameter", "dune_fem_parameter_module")):
        ok = importable(mod)
        print(f"{key}_importable={ok}")
        if ok:
            fail.append(f"{mod} is importable — the catalog claim that "
                        f"it is absent is now wrong")

    # dune.fem.parameter IS reachable as an attribute even though the
    # module import fails — that distinction is the trap.
    has_attr = hasattr(dune.fem, "parameter")
    print(f"dune_fem_parameter_attribute={has_attr}")
    if not has_attr:
        fail.append("dune.fem.parameter attribute missing")

    for f in fail:
        print(f"FAIL: {f}")
    if fail:
        return 1
    print("dune_companion_module_inventory_verified=True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
