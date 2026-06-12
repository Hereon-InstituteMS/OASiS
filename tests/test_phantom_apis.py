"""Regression: catalog Signal: clauses don't cite phantom APIs.

Wraps scripts/audit_phantom_apis.py as a pytest gate so the
Signal: catalog edits land alongside this check. Skips a
backend's audit when the live library is not importable in
the current test env (e.g. dolfinx is only in the ofa-fenicsx
conda env, not the repo .venv) — pytest runs in the repo .venv
so we audit skfem + ngsolve here. fenics is audited by hand /
in CI under the conda env (see scripts/audit_phantom_apis.py
docstring for the two-pass invocation).
"""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))


def _try_import(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


class TestNoUnexpectedPhantomAPIs(unittest.TestCase):
    """For each backend whose live library is importable in
    this env, assert the audit reports 0 unexpected phantoms.
    Backends whose libraries aren't importable are skipped —
    the audit is a multi-env gate, not a single-env gate."""

    def test_skfem_no_unexpected_phantoms(self) -> None:
        if not _try_import("skfem"):
            self.skipTest("skfem not in current env")
        from audit_phantom_apis import audit_backend, SKFEM_PATTERNS
        n, unexpected = audit_backend("skfem", SKFEM_PATTERNS)
        self.assertEqual(
            n, 0,
            f"unexpected skfem phantoms: {unexpected}. Fix the "
            "Signal: clause to use a real skfem idiom, or add "
            "the (module, attr) to INTENTIONAL_PHANTOMS in "
            "scripts/audit_phantom_apis.py with a rationale.")

    def test_ngsolve_no_unexpected_phantoms(self) -> None:
        if not _try_import("ngsolve"):
            self.skipTest("ngsolve not in current env")
        from audit_phantom_apis import audit_backend, NGSOLVE_PATTERNS
        n, unexpected = audit_backend("ngsolve", NGSOLVE_PATTERNS)
        self.assertEqual(
            n, 0,
            f"unexpected ngsolve phantoms: {unexpected}.")

    def test_fenics_no_unexpected_phantoms(self) -> None:
        if not _try_import("dolfinx"):
            self.skipTest("dolfinx not in current env")
        from audit_phantom_apis import audit_backend, FENICS_PATTERNS
        n, unexpected = audit_backend("fenics", FENICS_PATTERNS)
        self.assertEqual(
            n, 0,
            f"unexpected fenics phantoms: {unexpected}.")


if __name__ == "__main__":
    unittest.main()
