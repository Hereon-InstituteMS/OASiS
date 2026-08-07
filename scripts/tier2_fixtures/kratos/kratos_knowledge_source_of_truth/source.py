"""Tier-2: which Kratos knowledge catalog the backend actually consumes.

The claim is that backends.kratos.generators.KNOWLEDGE is the
ONLY per-physics catalog the backend uses, and that
data/kratos_knowledge does not export a unified
KRATOS_KNOWLEDGE dict, so importing it by that name fails
silently inside a try/except.

Mutation control: the dead `from data.kratos_knowledge import
KRATOS_KNOWLEDGE` branch this claim describes was already removed, so it
cannot be removed again. T2_MUTATE=1 REINSTATES it -- it appends that exact
import line to the backend source text the AST scan parses, i.e. it puts the
backend back into the state the claim describes. The scan is untouched and
still walks a real ast.parse of the real backend module; it must then report
backend_imports_data_kratos_knowledge=True and source_of_truth_mismatches=1.
That is what proves the printed False is a measurement of the backend's
import graph rather than a constant, and that a reintroduced dead branch is
actually caught.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
from pathlib import Path


MUTATE = os.environ.get("T2_MUTATE") == "1"
# Mutation control: the source line the AST scan must be able to see. Empty
# unless mutating, so the unmutated run parses the backend module verbatim.
_REINSTATED_DEAD_IMPORT = ""
if MUTATE:
    print("mutation=dead_data_kratos_knowledge_import_reinstated_in_scanned_source")
    _REINSTATED_DEAD_IMPORT = (
        "\nfrom data.kratos_knowledge import KRATOS_KNOWLEDGE\n")

_HERE = Path(__file__).resolve()
# In place the checkout is four levels up. The mutation harness stages a copy
# of this fixture into a scratch tree that has no such ancestor, so the search
# walks up looking for the catalog itself and only then falls back to
# $OASIS_REPO. If neither resolves, abort loudly: a fixture that cannot find
# the catalog it audits must never report a pass.
_REPO = next((p for p in _HERE.parents
              if (p / "src" / "backends" / "kratos" / "generators").is_dir()),
             None)
if _REPO is None:
    _REPO = Path(os.environ.get("OASIS_REPO") or "/nonexistent")
    if not (_REPO / "src" / "backends" / "kratos" / "generators").is_dir():
        print("FIXTURE_ABORT=no_oasis_checkout: set OASIS_REPO to the checkout "
              "whose Kratos catalog is under audit", file=sys.stderr)
        raise SystemExit(2)
sys.path.insert(0, str(_REPO / "src"))
# `data/` sits at the repo ROOT, not under src/. Without this the
# `import data.kratos_knowledge` below always landed in its except branch for
# a sys.path reason, so the half of the claim that is ABOUT what that module
# exports was never actually exercised: the fixture printed
# data_kratos_knowledge_importable=False and skipped the export check
# entirely. Appended rather than inserted, so repo-root directories cannot
# shadow site-packages.
sys.path.append(str(_REPO))


def main() -> int:
    bad = 0
    from backends.kratos.generators import KNOWLEDGE

    print(f"generators_KNOWLEDGE_is_a_dict={isinstance(KNOWLEDGE, dict)}")
    print(f"generators_KNOWLEDGE_key_count={len(KNOWLEDGE)}")
    if not isinstance(KNOWLEDGE, dict) or len(KNOWLEDGE) < 2:
        print("FAIL: the per-physics catalog is not a populated dict",
              file=sys.stderr)
        bad += 1

    # The claim: data/kratos_knowledge exports per-application constants and
    # NOT a unified KRATOS_KNOWLEDGE dict, so importing it by name fails.
    try:
        import data.kratos_knowledge as dk
        present = True
    except Exception as exc:
        present = False
        print(f"data_kratos_knowledge_importable=False")
        print(f"  message: {type(exc).__name__}: {str(exc)[:120]}")
    if present:
        print("data_kratos_knowledge_importable=True")
        has_unified = hasattr(dk, "KRATOS_KNOWLEDGE")
        print(f"exports_KRATOS_KNOWLEDGE={has_unified}")
        if has_unified:
            print("FAIL: data.kratos_knowledge DOES export a unified "
                  "KRATOS_KNOWLEDGE, contradicting the claim",
                  file=sys.stderr)
            bad += 1
        print(f"exports_KRATOS_APPLICATIONS="
              f"{hasattr(dk, 'KRATOS_APPLICATIONS')}")

    # The operative half either way: the generators catalog is the one the
    # backend actually consumes.
    from backends.kratos import backend as kb
    import ast

    tree = ast.parse(Path(kb.__file__).read_text() + _REINSTATED_DEAD_IMPORT)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    # A comment mentioning the old module is not an import of it; parse
    # rather than grep, or the check does not mean what it says.
    print(f"backend_imports_generators_KNOWLEDGE="
          f"{any(m.endswith('generators') for m in imported)}")
    dead = [m for m in imported if 'kratos_knowledge' in m]
    print(f"backend_imports_data_kratos_knowledge={bool(dead)}")
    if dead:
        print(f"FAIL: the backend still IMPORTS {dead}, which the claim says "
              f"was removed", file=sys.stderr)
        bad += 1

    print(f"source_of_truth_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
