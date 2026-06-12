"""Test the catalog hot-reload mechanism.

Anchors the importlib-based reload story documented in
data/postmortems/mcp-catalog-staleness-runtime-isolation.json.

Two assertions:

  (1) importlib.reload on src/backends/kratos/generators/poisson.py
      really does re-bind KNOWLEDGE['poisson']['pitfalls'] to a
      fresh dict (proof that python-import-based catalog
      surfaces are hot-reloadable in principle).

  (2) The reload_catalog tool function in
      src/tools/consolidated.py contains the lines that drive
      the reload (regex-level smoke check — keeps the tool
      from being silently removed in a refactor).

These are unit-test-grade assertions, not Tier-2 fixtures —
the catalog itself is what is under test, not a numerical
solve.
"""
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "data"))


class TestReloadMechanism(unittest.TestCase):
    """Verify the mechanism the reload_catalog tool relies on."""

    def test_importlib_reload_refreshes_KNOWLEDGE(self) -> None:
        # Step 1: import the module and capture the dict identity.
        from backends.kratos.generators import poisson as p
        first_id = id(p.KNOWLEDGE)
        first_count = len(p.KNOWLEDGE["poisson"]["pitfalls"])

        # Step 2: reload — the module object stays the same but
        # its top-level attributes are re-bound to fresh dicts.
        importlib.reload(p)
        second_id = id(p.KNOWLEDGE)
        second_count = len(p.KNOWLEDGE["poisson"]["pitfalls"])

        # KNOWLEDGE is re-bound to a fresh dict (identity differs).
        # We do NOT assert the count changes — content depends on
        # on-disk state, which the test does not mutate. The
        # important invariant is that re-import produces a fresh
        # object the registered backend can re-attach to.
        self.assertNotEqual(
            first_id, second_id,
            "importlib.reload should produce a fresh KNOWLEDGE "
            "dict; got the same id back — module-level "
            "attributes may not be re-bound under reload, which "
            "would break the reload_catalog hot-reload story.")
        self.assertEqual(first_count, second_count,
                         "no on-disk change between the two "
                         "loads — counts must match")

    def test_reload_catalog_tool_is_registered(self) -> None:
        """Smoke-check that the reload_catalog tool exists in
        consolidated.py. A future refactor that deletes the
        block would silently re-introduce the staleness bug
        from data/postmortems/mcp-catalog-staleness-runtime-
        isolation.json — this test fails loudly when that
        happens.
        """
        path = _REPO / "src" / "tools" / "consolidated.py"
        text = path.read_text(encoding="utf-8")
        self.assertIn(
            "def reload_catalog(", text,
            "reload_catalog tool function missing from "
            "src/tools/consolidated.py")
        self.assertIn(
            "importlib.reload", text,
            "reload_catalog must call importlib.reload to be "
            "effective")
        self.assertIn(
            "load_all_backends", text,
            "reload_catalog must re-run load_all_backends to "
            "re-bind the registered backends")


if __name__ == "__main__":
    unittest.main()
