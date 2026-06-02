"""Regression: kratos has a single source of truth.

Documented in src/backends/kratos/backend.py:get_knowledge:

  Tier 1 — backends.kratos.generators.KNOWLEDGE — the ONLY
           per-physics catalog. data/kratos_knowledge.py exists
           but exports per-application constants (not a unified
           dict), so it cannot be consulted by name.

Two invariants:

  (a) Every kratos supported_physics gets a non-empty
      KNOWLEDGE entry. The auxiliary_overview alias is honoured.

  (b) No silent import-failure dead code: data/kratos_knowledge.py
      MUST NOT define a top-level `KRATOS_KNOWLEDGE` symbol unless
      backend.py is updated to consume it. A previous version of
      backend.py imported that name; the import always failed and
      the lookup was silently dead code. If a future developer
      adds `KRATOS_KNOWLEDGE` to kratos_knowledge.py to fix some
      apparent gap, this test fails as a heads-up to also wire
      it into backend.py — otherwise we are back to phantom code.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "data"))


class TestKratosSourceOfTruth(unittest.TestCase):
    def test_every_physics_returns_knowledge(self) -> None:
        from core.registry import get_backend, load_all_backends
        load_all_backends()
        backend = get_backend("kratos")
        assert backend is not None

        silent: list[str] = []
        for cap in backend.supported_physics():
            k = backend.get_knowledge(cap.name)
            if not isinstance(k, dict) or not k:
                silent.append(cap.name)
        self.assertEqual(
            silent, [],
            f"kratos physics with EMPTY get_knowledge(): {silent}. "
            "Either add to backends.kratos.generators.KNOWLEDGE or "
            "extend backend.py with an additional documented source.")

    def test_no_phantom_kratos_knowledge_export(self) -> None:
        """If kratos_knowledge.py grows a top-level
        KRATOS_KNOWLEDGE symbol, backend.py must be updated in
        the same change to actually consume it (and this test
        updated accordingly). Otherwise we get a silent dead-
        code import like the one removed 2026-06-02."""
        import kratos_knowledge  # type: ignore
        self.assertFalse(
            hasattr(kratos_knowledge, "KRATOS_KNOWLEDGE"),
            "data/kratos_knowledge.py now defines KRATOS_KNOWLEDGE. "
            "Update src/backends/kratos/backend.py to actually "
            "consume it and update this test accordingly. Until "
            "then the symbol is unreachable.")


if __name__ == "__main__":
    unittest.main()
