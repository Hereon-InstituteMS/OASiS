"""Regression: cross-backend collation pitfalls module + knowledge tool wiring.

Created 2026-06-03 as the first concrete cross-backend collation content
(task #220). Pins:
  1. The _cross.py module loads cleanly with no syntax / import errors.
  2. Each pitfall string carries the [Cross-Backend] tag — without it, the
     content drifts into single-backend territory and belongs elsewhere.
  3. Each pitfall names at least two backends in its body — collation
     pitfalls without a backend-pair are mis-categorized.
  4. The topic filter (e.g. 'units', 'mesh') narrows the dict correctly.
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


# Canonical backend names that pitfalls may reference. A pitfall must
# mention at least two of these by name in its text.
_KNOWN_BACKENDS = {
    "fenics", "dolfinx", "skfem", "ngsolve", "kratos", "dealii",
    "deal.ii", "fourc", "4c", "dune", "febio",
}


class TestCrossBackendPitfalls(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from backends._cross import (
            CROSS_BACKEND_PITFALLS,
            get_cross_backend_pitfalls,
        )
        cls.pitfalls = CROSS_BACKEND_PITFALLS
        cls.get = staticmethod(get_cross_backend_pitfalls)

    def test_module_has_entries(self) -> None:
        self.assertGreater(
            len(self.pitfalls), 0,
            "CROSS_BACKEND_PITFALLS is empty — collation work isn't doing anything",
        )

    def test_every_entry_has_required_fields(self) -> None:
        for key, entry in self.pitfalls.items():
            with self.subTest(key=key):
                self.assertIn("description", entry)
                self.assertIn("pitfalls", entry)
                self.assertIn("Signal", entry)
                self.assertIsInstance(entry["pitfalls"], list)
                self.assertGreater(len(entry["pitfalls"]), 0)

    def test_every_pitfall_has_cross_backend_tag(self) -> None:
        for key, entry in self.pitfalls.items():
            for i, p in enumerate(entry["pitfalls"]):
                with self.subTest(key=key, pitfall_idx=i):
                    self.assertIn(
                        "[Cross-Backend]", p,
                        f"Pitfall in {key} missing [Cross-Backend] tag — if it's "
                        "a single-backend issue, move it to src/backends/<be>/",
                    )

    def test_every_pitfall_names_at_least_two_backends(self) -> None:
        """A collation pitfall by definition needs >= 2 backends in the body."""
        for key, entry in self.pitfalls.items():
            for i, p in enumerate(entry["pitfalls"]):
                with self.subTest(key=key, pitfall_idx=i):
                    text = p.lower()
                    hits = sum(1 for be in _KNOWN_BACKENDS if be in text)
                    self.assertGreaterEqual(
                        hits, 2,
                        f"Pitfall in {key} mentions <2 backends — collation "
                        f"requires naming the delta-pair explicitly. Text "
                        f"snippet: {p[:200]!r}",
                    )

    def test_topic_filter_units(self) -> None:
        r = self.get("units")
        self.assertIn("units", r)
        self.assertEqual(len(r), 1)

    def test_topic_filter_mesh(self) -> None:
        # 'mesh' substring should match element_node_ordering AND
        # restart_checkpoint_compatibility (the latter mentions
        # 'mesh layout' in its description).
        r = self.get("mesh")
        self.assertIn("element_node_ordering", r)

    def test_no_topic_returns_everything(self) -> None:
        r = self.get(None)
        self.assertEqual(len(r), len(self.pitfalls))

    def test_unknown_topic_falls_back_to_full_set(self) -> None:
        """Unknown topic returns the full set rather than raising —
        an LLM that types a garbage filter should still see content."""
        r = self.get("zzzz_no_match_xyz")
        self.assertEqual(len(r), len(self.pitfalls))


if __name__ == "__main__":
    unittest.main()
