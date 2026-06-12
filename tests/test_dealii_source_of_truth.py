"""Regression: source-of-truth ordering for dealii KNOWLEDGE.

Documented in src/backends/dealii/backend.py:get_knowledge:

  Tier 1 — data/dealii_knowledge.py:DEALII_KNOWLEDGE
           (currently never holds per-physics keys; present
           for future course-level catalog entries).
  Tier 2 — generator-embedded KNOWLEDGE in
           backends.dealii.generators — primary catalog (most
           dealii physics live here).
  Tier 3 — tools.deep_knowledge._DEALII_KNOWLEDGE — fallback
           ONLY for physics not covered above. Currently 3
           entries: advection_dg, contact, nonlinear_elasticity.

The audit pins both:
  (a) the exact set of physics in Tier 3, so any new addition
      that bypasses the generator catalog has to be explicit
      and reviewed
  (b) no physics silently falls through every tier and returns
      an empty / no-pitfalls dict
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "data"))


# Physics whose live get_knowledge() result comes from
# tools.deep_knowledge._DEALII_KNOWLEDGE (Tier 3). Everything
# else must come from the generator-embedded KNOWLEDGE.
_TIER3_DEALII_FALLBACK: set[str] = {
    "advection_dg",
    "contact",
    "nonlinear_elasticity",
}


class TestDealiiSourceOfTruth(unittest.TestCase):
    def test_tier3_set_matches_runtime(self) -> None:
        from core.registry import get_backend, load_all_backends
        from backends.dealii.generators import (
            get_knowledge as gen_get_knowledge,
        )
        from tools.deep_knowledge import _DEALII_KNOWLEDGE

        load_all_backends()
        backend = get_backend("dealii")
        assert backend is not None

        # Compute the runtime tier-3 set: physics whose generator
        # KNOWLEDGE is empty / has no pitfalls AND which ARE in
        # _DEALII_KNOWLEDGE.
        runtime_tier3: set[str] = set()
        for cap in backend.supported_physics():
            gen_k = gen_get_knowledge(cap.name)
            gen_has_pitfalls = (
                isinstance(gen_k, dict) and bool(gen_k.get("pitfalls"))
            )
            in_tools_deep = cap.name in _DEALII_KNOWLEDGE
            if not gen_has_pitfalls and in_tools_deep:
                runtime_tier3.add(cap.name)

        self.assertEqual(
            runtime_tier3,
            _TIER3_DEALII_FALLBACK,
            "dealii Tier-3 fallback set drifted. Fix "
            "backend.py docstring + this allow-list in "
            "lock-step.\n"
            f"  expected: {sorted(_TIER3_DEALII_FALLBACK)}\n"
            f"  actual:   {sorted(runtime_tier3)}\n"
            f"  diff:     {sorted(runtime_tier3 ^ _TIER3_DEALII_FALLBACK)}")

    def test_get_knowledge_returns_nonempty_for_every_physics(self) -> None:
        """No dealii physics may silently return {} from
        get_knowledge."""
        from core.registry import get_backend, load_all_backends

        load_all_backends()
        backend = get_backend("dealii")
        assert backend is not None

        silent_failures: list[str] = []
        for cap in backend.supported_physics():
            k = backend.get_knowledge(cap.name)
            if not isinstance(k, dict) or not k:
                silent_failures.append(cap.name)

        self.assertEqual(
            silent_failures, [],
            "dealii physics with EMPTY get_knowledge() result: "
            f"{silent_failures}. Add the physics to one of the "
            "three documented tiers (data/dealii_knowledge.py, "
            "generators KNOWLEDGE, or tools.deep_knowledge.")


if __name__ == "__main__":
    unittest.main()
