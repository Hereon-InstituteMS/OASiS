"""Layer A — DUNE-fem source scanner.

DUNE's Python runtime is currently broken on this machine
by a conda/openmpi libibverbs ABI mismatch (task #15) —
'import dune.common' raises ImportError before any DUNE
function can be called. So this scanner walks the
installed DUNE source tree DIRECTLY, parsing 'def X(' /
'class X' declarations from Python source files.

Trade-off vs runtime introspection:
  + Works even when import is broken.
  + Robust against catalog claims for functions in
    physics the user hasn't run yet.
  - Misses dynamically-generated names (DUNE has some
    JIT-driven name aliases via dune.generator).
  - Asserts on the build-cmake / python tree, which is
    the AUTHORITATIVE source for what DUNE exposes after
    a clean build.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


DUNE_SRC_ROOTS = (
    Path("/home/hermann/Schreibtisch/dune-src/dune-fem"
         "/python/dune"),
    Path("/home/hermann/Schreibtisch/dune-src/dune-common"
         "/python/dune"),
    Path("/home/hermann/Schreibtisch/dune-src/dune-grid"
         "/python/dune"),
    Path("/home/hermann/Schreibtisch/dune-src/dune-istl"
         "/python/dune"),
)

# Catalog-claimed function names from grep of
# src/backends/dune/generators/*.py
CATALOG_FUNCTIONS = (
    "structuredGrid", "lagrange", "dglagrange",
    "raviartThomas", "galerkin",
    "DirichletBC", "gridFunction",
    "composite", "combined", "product",
    "dgonb", "dglegendre",
    # In case catalog wrote the WRONG casing:
    "raviartthomas",          # phantom — must be absent
)
PHANTOM = ("raviartthomas",)


def find_def_or_class(src: Path) -> set[str]:
    """Return all 'def X' and 'class X' names declared in
    any *.py under src."""
    names = set()
    pattern_def = re.compile(r"^\s*def\s+(\w+)\s*\(",
                             re.MULTILINE)
    pattern_class = re.compile(r"^\s*class\s+(\w+)\b",
                               re.MULTILINE)
    for p in src.rglob("*.py"):
        try:
            text = p.read_text()
        except Exception:
            continue
        names |= set(pattern_def.findall(text))
        names |= set(pattern_class.findall(text))
    return names


def main() -> int:
    available_names = set()
    roots_walked = []
    for root in DUNE_SRC_ROOTS:
        if root.is_dir():
            roots_walked.append(str(root))
            available_names |= find_def_or_class(root)
    print(f"roots_walked={roots_walked}")
    print(f"unique_def_class_names_count={len(available_names)}")

    catalog_missing = []
    for fn in CATALOG_FUNCTIONS:
        if fn in PHANTOM:
            continue
        if fn not in available_names:
            catalog_missing.append(fn)
    print(f"catalog_missing={catalog_missing}")

    phantom_present = [p for p in PHANTOM
                       if p in available_names]
    print(f"phantom_present={phantom_present}")

    # Provenance: list a few signature lines for the real
    # ones to confirm we are looking at the right symbol.
    fem_spaces = (Path("/home/hermann/Schreibtisch/dune-src"
                       "/dune-fem/python/dune/fem/space/"
                       "_spaces.py"))
    if fem_spaces.is_file():
        text = fem_spaces.read_text()
        rt_match = re.search(
            r"def\s+(raviartThomas|raviartthomas)\b"
            r"[^\n]*", text)
        if rt_match:
            print(f"_spaces.py_rt_def_line='{rt_match.group()}'")

    out_dir = Path(__file__).resolve().parent / "scan_results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "dune_source_scan.json"
    out.write_text(json.dumps({
        "roots_walked": roots_walked,
        "n_unique_names": len(available_names),
        "catalog_missing": catalog_missing,
        "phantom_present": phantom_present,
    }, indent=2))
    print(f"wrote={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
