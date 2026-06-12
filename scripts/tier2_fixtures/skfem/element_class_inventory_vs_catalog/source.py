"""Tier-2: skfem element-class inventory vs catalog.

Asserts (a) every element class the catalog claims as real
IS present in the installed skfem; (b) every element class
the catalog flags as PHANTOM (catalog says it doesn't exist
under that name) actually is absent; and (c) the
under-exposed real classes the catalog could use are
present.

This is the Layer A regression gate for catalog ↔ library
naming alignment, derived from the skfem_source_scanner.py
run on skfem 12.0.1 (2026-06-01).

If skfem renames a class upstream (e.g. ElementTriRT0 →
ElementTriRaviartThomas in some future major), the fixture
flips signs and the test fails until either:
  * the catalog catches up, or
  * the alias is restored upstream.
"""
from __future__ import annotations

import sys

import skfem


# Catalog names skfem genuinely ships under (must all be
# present at runtime).
CATALOG_REAL = (
    "ElementDG",
    "ElementHex1",
    "ElementLineHermite",
    "ElementQuad1",
    "ElementQuadBFS",
    "ElementTetMini",
    "ElementTetP1",
    "ElementTetP2",
    "ElementTetRT0",
    "ElementTriArgyris",
    "ElementTriDG",
    "ElementTriMini",
    "ElementTriMorley",
    "ElementTriP0",
    "ElementTriP1",
    "ElementTriP2",
    "ElementTriRT0",
    "ElementTriRT1",
    "ElementVector",
)

# Catalog comments call these phantom (must NOT be added
# upstream OR catalog must be updated when they are).
CATALOG_PHANTOM = (
    "ElementTriRaviartThomas",
)

# Real classes skfem ships that the catalog generators do
# not yet name. Their presence lets future PRs widen
# coverage; their absence (future skfem rename) flips the
# fixture green-yellow.
UNDERCOVERED_REAL = (
    "ElementTriBDM1",
    "ElementTriCR",
    "ElementTriCCR",
    "ElementHex2",
    "ElementTetCR",
    "ElementTetCCR",
    "ElementTetRT1",
    "ElementWedge1",
    "ElementLineP1",
    "ElementLineP2",
    "ElementLineMini",
    "ElementGlobal",
    "ElementComposite",
)


def main() -> int:
    print(f"skfem_version={skfem.__version__}")

    real_missing = [e for e in CATALOG_REAL
                    if not hasattr(skfem, e)]
    phantom_present = [e for e in CATALOG_PHANTOM
                       if hasattr(skfem, e)]
    undercov_missing = [e for e in UNDERCOVERED_REAL
                        if not hasattr(skfem, e)]
    print(f"catalog_real_missing={real_missing}")
    print(f"catalog_phantom_present={phantom_present}")
    print(f"undercovered_real_now_missing={undercov_missing}")

    ok = (
        not real_missing
        and not phantom_present
        and not undercov_missing
    )
    if ok:
        return 0
    print("FAIL: skfem element-class inventory drift",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
