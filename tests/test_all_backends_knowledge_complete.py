"""Regression: every backend's supported_physics yields non-empty knowledge.

This is the universal contract complementing the per-backend
source-of-truth tests (test_{fenics,dealii,kratos}_source_of_truth):
no matter how a backend's get_knowledge resolves its lookup
(single-source, multi-tier fallback, or merge), the user MUST
get a non-empty dict back for every physics surfaced via
supported_physics().

A silent {} fallback is the worst possible UX:
  - discover('physics') lists the physics
  - knowledge(solver=..., physics=...) returns nothing
  - the LLM has no way to know whether the catalog is empty
    or its query is wrong

The kratos phantom-import bug (removed 2026-06-02) was the
canonical example: the deep_knowledge lookup silently
failed for years because `KRATOS_KNOWLEDGE` was never exported
from kratos_knowledge.py. This test would have caught it the
moment that import was added.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "data"))


# Backends to cover. All 8 must be in this list — if a new
# backend is added to the registry and not listed here, that
# itself is a coverage gap.
_BACKENDS = ["fenics", "dealii", "ngsolve", "skfem", "kratos",
             "fourc", "febio", "dune"]


class TestAllBackendsKnowledgeComplete(unittest.TestCase):
    def test_every_backend_every_physics_has_knowledge(self) -> None:
        from core.registry import get_backend, load_all_backends
        load_all_backends()

        bad: dict[str, list[str]] = {}
        for bk_name in _BACKENDS:
            backend = get_backend(bk_name)
            self.assertIsNotNone(
                backend, f"backend not registered: {bk_name}")
            assert backend is not None

            silent = []
            for cap in backend.supported_physics():
                k = backend.get_knowledge(cap.name)
                if not isinstance(k, dict) or not k:
                    silent.append(cap.name)
            if silent:
                bad[bk_name] = silent

        self.assertEqual(
            bad, {},
            "Silent get_knowledge() = {} fallbacks detected. "
            "Each (backend, physics) listed has supported_physics "
            "advertising the physics but get_knowledge returns an "
            "empty dict. Wire up KNOWLEDGE / deep_knowledge / data "
            "entries to surface real content, or remove the "
            "PhysicsCapability:\n"
            + "\n".join(f"  {b}: {ps}" for b, ps in bad.items()))

    def test_every_registered_backend_is_audited(self) -> None:
        """The _BACKENDS list above must be exhaustive — if a
        new backend is registered, it must be added here."""
        from core.registry import all_backends, load_all_backends
        load_all_backends()

        registered = {b.name() for b in all_backends()}
        audited = set(_BACKENDS)
        missing = registered - audited
        self.assertEqual(
            missing, set(),
            f"backends registered but not audited: {missing}. "
            "Add them to tests/test_all_backends_knowledge_complete."
            "_BACKENDS so the coverage gate keeps up.")


if __name__ == "__main__":
    unittest.main()
