"""Upstream-SOURCE coverage audit.

Counterpart to scripts/upstream_demo_audit.py — that tool walks
backend DEMO files. This one walks each backend's installed
SOURCE modules and reports every public symbol (class /
function) that is NOT mentioned in ANY catalog pitfall.

The output `data/upstream_source_coverage.json` is the precise,
line-by-line surface area the user-stated directive demands:
"completely loop over the entire backend source codes line by
line, test everything super carefully, encode our knowledge,
pitfalls etc."

Each cron tick can pick one uncovered symbol from the report,
verify its current behaviour in the live env, and either:
  - add a pitfall + falsification probe documenting how it
    fails when misused
  - confirm it's an internal / private / utility symbol that
    doesn't need user-facing documentation
  - update an existing pitfall to also cite it

Run:
  .venv/bin/python scripts/upstream_source_audit.py [--backend NAME]

Output schema (per backend):
  {
    "package": "skfem",
    "version": "12.0.1",
    "n_public_symbols": 412,
    "n_documented": 87,   # mentioned in at least one pitfall
    "n_uncovered": 325,
    "coverage_pct": 21.1,
    "uncovered": [
      {"module": "skfem.element.element_tri", "name": "ElementTriMorley", "kind": "class"},
      ...
    ],
  }
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import pkgutil
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


# Symbol prefixes / names that are infrastructure rather than
# user-facing API. Skip them from the "uncovered" tally.
_PRIVATE_PREFIX = ("_",)
_SKIP_NAMES: set[str] = {
    # Common Python stdlib / metaclass leakage
    "TYPE_CHECKING", "annotations", "TypeVar", "Generic",
    "ABC", "ABCMeta", "abstractmethod", "dataclass", "field",
    "Path", "List", "Dict", "Tuple", "Optional", "Union",
    "Any", "Callable", "Iterator", "Iterable", "Sequence",
    "Set", "FrozenSet",
    # numpy / scipy / json / sys aliases people import into modules
    "np", "numpy", "scipy", "json", "sys", "os", "math",
    "logging", "warnings",
}


def _is_user_facing(modname: str, name: str, obj) -> bool:
    """Heuristic: is this symbol part of the backend's public API,
    or an import / metaclass / private helper?"""
    if name.startswith(_PRIVATE_PREFIX):
        return False
    if name in _SKIP_NAMES:
        return False
    # Symbol must be DEFINED in this module (not re-exported from
    # numpy/scipy/etc).
    obj_mod = getattr(obj, "__module__", None)
    if obj_mod and not obj_mod.startswith(modname.split(".", 1)[0]):
        return False
    # Skip dunders / known stdlib classes.
    if name in ("__class__", "__init__", "__repr__"):
        return False
    return True


def _walk_package(pkg_name: str) -> list[dict]:
    """Walk every public class + function in `pkg_name` and its
    sub-modules. Returns list of {module, name, kind} dicts."""
    try:
        pkg = importlib.import_module(pkg_name)
    except ImportError as ex:
        print(f"WARN: cannot import {pkg_name}: {ex}",
              file=sys.stderr)
        return []
    found: list[dict] = []
    seen: set[str] = set()

    def visit(modname: str) -> None:
        if modname in seen:
            return
        seen.add(modname)
        try:
            mod = importlib.import_module(modname)
        except Exception:
            return
        for name in dir(mod):
            try:
                obj = getattr(mod, name)
            except Exception:
                continue
            if not _is_user_facing(modname, name, obj):
                continue
            if inspect.isclass(obj):
                kind = "class"
            elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
                kind = "function"
            else:
                continue
            found.append({"module": modname, "name": name,
                          "kind": kind})

        if hasattr(mod, "__path__"):
            for sub in pkgutil.iter_modules(mod.__path__):
                if sub.name.startswith("_"):
                    continue
                visit(f"{modname}.{sub.name}")

    visit(pkg_name)
    # Deduplicate. Two passes:
    #   1. By (name, kind) — drop re-exports of the same name.
    #   2. By object identity — drop aliases (where one symbol
    #      is just a re-binding of another, e.g. skfem's
    #      `BoundaryFacetBasis is FacetBasis`).
    by_name: dict[tuple[str, str], dict] = {}
    for entry in found:
        key = (entry["name"], entry["kind"])
        if key not in by_name:
            by_name[key] = entry

    # Resolve each entry back to its underlying object identity
    # and collapse aliases. Keep the canonical-named one
    # (shortest/lexicographically-first __qualname__ match).
    by_objid: dict[int, dict] = {}
    for entry in by_name.values():
        try:
            mod = importlib.import_module(entry["module"])
            obj = getattr(mod, entry["name"])
            oid = id(obj)
        except Exception:
            # Keep entry as-is if we can't resolve.
            by_objid[id(entry)] = entry
            continue
        existing = by_objid.get(oid)
        if existing is None:
            by_objid[oid] = entry
        else:
            # Prefer the canonical name — the one matching
            # __name__ if available; else lexicographic.
            canonical_name = getattr(obj, "__name__",
                                     entry["name"])
            if entry["name"] == canonical_name:
                by_objid[oid] = entry
            elif existing["name"] != canonical_name and \
                 entry["name"] < existing["name"]:
                by_objid[oid] = entry
    return list(by_objid.values())


def _collect_catalog_text() -> str:
    """Concatenate every catalog pitfall's text into a single
    lowercase blob. Symbols are then probed by substring."""
    from core.registry import load_all_backends, all_backends
    load_all_backends()
    chunks: list[str] = []
    for b in all_backends():
        for cap in b.supported_physics():
            k = b.get_knowledge(cap.name)
            if not isinstance(k, dict):
                continue
            for p in k.get("pitfalls", []):
                chunks.append(str(p))
            # Also pull description + weak_form prose.
            for field in ("description", "weak_form", "elements"):
                v = k.get(field)
                if isinstance(v, str):
                    chunks.append(v)
                elif isinstance(v, (list, tuple)):
                    chunks.extend(str(x) for x in v)
    return "\n".join(chunks)


# Backend → (pip-importable package name, conda env path or None)
_BACKEND_PACKAGES: dict[str, tuple[str, str | None]] = {
    "skfem":   ("skfem", None),  # in .venv
    "kratos":  ("KratosMultiphysics", None),  # in .venv
    "ngsolve": ("ngsolve", None),  # in .venv
    "fenics":  ("dolfinx",
                str(Path.home() / "miniconda3/envs/ofa-fenicsx")),
}


def _scan_one(backend: str, catalog_text: str) -> dict:
    pkg_name, env_path = _BACKEND_PACKAGES.get(backend, (None, None))
    if pkg_name is None:
        return {"package": None, "skipped": "no package mapping"}
    if env_path is not None:
        # Subprocess into the env to import the package.
        # For now, only scan packages importable in THIS python.
        try:
            importlib.import_module(pkg_name)
        except ImportError:
            return {"package": pkg_name,
                    "skipped": f"not importable here "
                               f"(need {env_path})"}
    try:
        symbols = _walk_package(pkg_name)
    except Exception as ex:
        return {"package": pkg_name, "error": str(ex)}

    try:
        pkg_obj = importlib.import_module(pkg_name)
        version = getattr(pkg_obj, "__version__", "unknown")
    except Exception:
        version = "unknown"

    blob_lower = catalog_text.lower()
    documented = []
    uncovered = []
    for sym in symbols:
        if sym["name"].lower() in blob_lower:
            documented.append(sym)
        else:
            uncovered.append(sym)

    return {
        "package": pkg_name,
        "version": version,
        "n_public_symbols": len(symbols),
        "n_documented": len(documented),
        "n_uncovered": len(uncovered),
        "coverage_pct": (round(100.0 * len(documented)
                               / max(1, len(symbols)), 1)),
        # Cap uncovered list at 200 to keep output manageable.
        "uncovered": uncovered[:200],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None,
                        help="restrict to one backend; default all")
    args = parser.parse_args()

    catalog = _collect_catalog_text()
    print(f"Catalog text: {len(catalog):,} chars across all "
          f"backends + physics", file=sys.stderr)

    out = {}
    backends = ([args.backend] if args.backend
                else list(_BACKEND_PACKAGES))
    for be in backends:
        scan = _scan_one(be, catalog)
        out[be] = scan

    output_path = _REPO / "data" / "upstream_source_coverage.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))

    print(f"\nUpstream source-symbol audit (written to "
          f"{output_path.name})\n")
    print(f"{'backend':<10} {'pkg':<22} {'symbols':>8} "
          f"{'docd':>5} {'gap':>5}  {'cov%':>6}  ver")
    print("-" * 80)
    for be, scan in out.items():
        if "skipped" in scan:
            print(f"{be:<10} {scan.get('package', '?'):<22} "
                  f"skipped: {scan['skipped']}")
            continue
        if "error" in scan:
            print(f"{be:<10} {scan.get('package', '?'):<22} "
                  f"ERROR: {scan['error']}")
            continue
        print(f"{be:<10} {scan['package']:<22} "
              f"{scan['n_public_symbols']:>8} "
              f"{scan['n_documented']:>5} {scan['n_uncovered']:>5}"
              f"  {scan['coverage_pct']:5.1f}%  {scan['version']}")

    print()
    print("Top 10 uncovered symbols per backend (target for next "
          "cron ticks):")
    for be, scan in out.items():
        if "uncovered" not in scan:
            continue
        if not scan["uncovered"]:
            continue
        print(f"\n  {be} ({scan['n_uncovered']} uncovered):")
        for sym in scan["uncovered"][:10]:
            print(f"    {sym['kind']:<8} "
                  f"{sym['module']}.{sym['name']}")
        if scan["n_uncovered"] > 10:
            print(f"    ... + {scan['n_uncovered'] - 10} more")

    return 0


if __name__ == "__main__":
    sys.exit(main())
