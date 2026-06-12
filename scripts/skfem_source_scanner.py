"""Layer A — scikit-fem source scanner.

Introspects the installed skfem Python package and compares
against the catalog's claimed APIs in:

  src/backends/skfem/generators/*.py
  src/backends/skfem/backend.py

For each catalog-claimed attribute (element class, mesh
constructor, top-level helper) this scanner reports
whether the attribute resolves at runtime in the current
env.

Layer A pattern: walk the source/library directly and emit
a 'claimed-but-absent' diff. Tier-2 fixtures then lock
specific findings as regression gates.
"""
from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path


# Catalog-claimed element classes from grep of
# src/backends/skfem/generators/*.py. Some are aliases
# the catalog refers to but the library doesn't ship
# under that name — those are the falsifications.
CATALOG_ELEMENTS = (
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
    "ElementTriRaviartThomas",   # phantom — known wrong, locked
    "ElementVector",
)

# Real element classes the catalog under-exposes. These
# exist in skfem 12.0+ and could be added to generators.
UNDERCOVERED_REAL_ELEMENTS = (
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

# Top-level helpers that catalog references
CATALOG_TOPLEVEL = (
    "BilinearForm",
    "LinearForm",
    "Functional",
    "Basis",
    "CellBasis",
    "InteriorBasis",
    "FacetBasis",
    "MeshTri",
    "MeshQuad",
    "MeshTet",
    "MeshHex",
    "condense",
    "enforce",
    "solve",
    "asm",
    "DofsView",
)

# Sub-module surfaces
CATALOG_SUBMODULES = (
    "skfem.models.poisson.laplace",
    "skfem.models.poisson.mass",
    "skfem.models.poisson.unit_load",
    "skfem.models.elasticity.linear_elasticity",
    "skfem.models.elasticity.lame_parameters",
    "skfem.io.meshio.from_meshio",
    "skfem.io.meshio.to_meshio",
)


def resolve(dotted: str) -> bool:
    """Resolve 'pkg.sub.attr' chain."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            mod = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        obj = mod
        for attr in parts[cut:]:
            if not hasattr(obj, attr):
                obj = None
                break
            obj = getattr(obj, attr)
        if obj is not None:
            return True
    return False


def main() -> int:
    import skfem
    version = skfem.__version__
    print(f"skfem_version={version}")

    elem_present = {
        e: hasattr(skfem, e) for e in CATALOG_ELEMENTS}
    elem_extra = {
        e: hasattr(skfem, e) for e in UNDERCOVERED_REAL_ELEMENTS}
    top_present = {
        t: hasattr(skfem, t) for t in CATALOG_TOPLEVEL}
    sub_present = {
        s: resolve(s) for s in CATALOG_SUBMODULES}

    print("--- Catalog elements ---")
    for k in sorted(elem_present):
        print(f"  {k}={elem_present[k]}")
    print("--- Real-but-under-exposed elements ---")
    for k in sorted(elem_extra):
        print(f"  {k}={elem_extra[k]}")
    print("--- Catalog top-level helpers ---")
    for k in sorted(top_present):
        print(f"  {k}={top_present[k]}")
    print("--- Catalog sub-module APIs ---")
    for k in sorted(sub_present):
        print(f"  {k}={sub_present[k]}")

    # condense signature provenance
    sig_condense = str(inspect.signature(skfem.condense))
    print(f"condense_signature={sig_condense}")
    sig_solve = str(inspect.signature(skfem.solve))
    print(f"solve_signature={sig_solve}")

    falsifications = {
        e: "claimed-but-absent" for e in CATALOG_ELEMENTS
        if not elem_present[e]
    }
    falsifications.update({
        t: "claimed-but-absent" for t in CATALOG_TOPLEVEL
        if not top_present[t]
    })
    falsifications.update({
        s: "claimed-but-absent" for s in CATALOG_SUBMODULES
        if not sub_present[s]
    })
    print(f"\nfalsifications={falsifications}")

    # Write structured output
    out_dir = Path(__file__).resolve().parent / "scan_results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "skfem_source_scan.json"
    out_path.write_text(json.dumps({
        "version": version,
        "elements_catalog": elem_present,
        "elements_undercovered_real": elem_extra,
        "toplevel": top_present,
        "submodules": sub_present,
        "condense_signature": sig_condense,
        "solve_signature": sig_solve,
        "falsifications": falsifications,
    }, indent=2))
    print(f"wrote={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
