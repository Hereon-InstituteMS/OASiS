"""Tier-2: which Kratos knowledge catalog the backend actually consumes.

The claim is that backends.kratos.generators.KNOWLEDGE is the
ONLY per-physics catalog the backend uses, and that
data/kratos_knowledge does not export a unified
KRATOS_KNOWLEDGE dict, so importing it by that name fails
silently inside a try/except.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
from pathlib import Path


_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[4]
sys.path.insert(0, str(_REPO / "src"))


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

    tree = ast.parse(Path(kb.__file__).read_text())
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
