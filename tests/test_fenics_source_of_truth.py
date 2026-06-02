"""Regression: source-of-truth ordering for fenics KNOWLEDGE.

Documented in src/backends/fenics/backend.py (audit 2026-06-02):

  Tier 1 — src/tools/deep_knowledge.py is CANONICAL for any
           physics it covers. The matching KNOWLEDGE dict in
           src/backends/fenics/generators/<phys>.py (or in
           generators/advanced.py inside a multi-physics dict)
           is DEAD CODE and may have drifted from the canonical
           version.

  Tier 2 — Generator-level KNOWLEDGE is canonical ONLY for
           physics NOT covered by deep_knowledge.py. The 7
           fenics physics in this category are listed below.

This test pins both invariants:

  1. Every physics whose KNOWLEDGE actually surfaces to the LLM
     via get_knowledge() is exposed in supported_physics() (the
     orphan check, complementary to
     test_no_fenics_orphan_physics).

  2. The set of physics whose generator KNOWLEDGE is canonical
     (i.e. NOT shadowed by deep_knowledge.py) matches the
     documented allow-list in the backend.py docstring. If a
     new physics is added to deep_knowledge.py the corresponding
     generator KNOWLEDGE becomes dead code; if a new physics is
     added to a generator file that is NOT in deep_knowledge.py
     it becomes part of the canonical surface and must be
     listed here so reviewers know.

If either invariant breaks, fix backend.py + this allow-list
together so the runtime behaviour and the documentation stay
in sync.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


# Physics whose KNOWLEDGE in src/backends/fenics/generators/ is
# the canonical / live source. Anything NOT in this set is
# shadowed by deep_knowledge.py.
#
# (Note: the `elasticity` generator file exists too, but
# `elasticity` is NOT exposed as a supported_physics() — fenics
# exposes `linear_elasticity` which IS in deep_knowledge.py.
# The `elasticity` generator is shared-implementation code for
# the `linear_elasticity` template path. So it does not appear
# in this allow-list.)
_GENERATOR_CANONICAL: set[str] = {
    "mixed_poisson",
    "dg_methods",
    "multiphase",
    "time_dependent_heat",
    "nonlinear_pde",
    "magnetostatics",
}


class TestFenicsSourceOfTruth(unittest.TestCase):
    def test_generator_canonical_set_matches_runtime(self) -> None:
        """The set of physics where get_knowledge falls through to
        the generator KNOWLEDGE must equal _GENERATOR_CANONICAL.

        Computed by walking every fenics physics and checking
        whether get_deep_fenics_knowledge returns a non-empty
        dict (deep_knowledge wins → generator is dead) or not
        (generator wins → generator is canonical).
        """
        from core.registry import get_backend, load_all_backends
        from tools.deep_knowledge import get_deep_fenics_knowledge

        load_all_backends()
        backend = get_backend("fenics")
        assert backend is not None

        gen_canonical_actual: set[str] = set()
        for cap in backend.supported_physics():
            if not get_deep_fenics_knowledge(cap.name):
                gen_canonical_actual.add(cap.name)

        # Either set difference is a real bug:
        #   - extra in actual = a new generator physics that
        #     someone added without listing it here OR a physics
        #     was removed from deep_knowledge.py
        #   - missing from actual = a physics moved into
        #     deep_knowledge.py but the allow-list here was not
        #     pruned
        self.assertEqual(
            gen_canonical_actual,
            _GENERATOR_CANONICAL,
            "Generator-canonical fenics physics drifted from "
            "the documented allow-list. Fix backend.py "
            "docstring + tests/test_fenics_source_of_truth.py "
            "in lock-step.\n"
            f"  expected: {sorted(_GENERATOR_CANONICAL)}\n"
            f"  actual:   {sorted(gen_canonical_actual)}\n"
            f"  diff:     {sorted(gen_canonical_actual ^ _GENERATOR_CANONICAL)}")

    def test_get_knowledge_returns_nonempty_for_every_physics(self) -> None:
        """No fenics physics should silently return {} from
        get_knowledge. Either deep_knowledge.py covers it (Tier 1)
        or the generator KNOWLEDGE does (Tier 2).
        """
        from core.registry import get_backend, load_all_backends

        load_all_backends()
        backend = get_backend("fenics")
        assert backend is not None

        silent_failures: list[str] = []
        for cap in backend.supported_physics():
            k = backend.get_knowledge(cap.name)
            if not isinstance(k, dict) or not k:
                silent_failures.append(cap.name)

        self.assertEqual(
            silent_failures, [],
            "fenics physics with EMPTY get_knowledge() result: "
            f"{silent_failures}. Either add the physics to "
            "src/tools/deep_knowledge.py or restore the generator-"
            "level KNOWLEDGE dict.")


if __name__ == "__main__":
    unittest.main()
