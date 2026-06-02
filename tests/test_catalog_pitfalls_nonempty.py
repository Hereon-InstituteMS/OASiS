"""Regression: every (backend, physics) pair has at least one
pitfall in its knowledge dict.

Empty pitfalls lists mean the LLM gets no warning about the
known traps for that physics. Even a single, conservative
'[Integration] this template is a probe / stub' pitfall is
better than nothing.

Catches the regression where:
  - a new physics is added to supported_physics() but the
    KNOWLEDGE['pitfalls'] field is forgotten.
  - a catalog edit accidentally empties an existing pitfalls
    list.

Current state (audit 2026-06-02): 210 (backend, physics)
pairs across 8 backends, 1116 pitfalls total, min 1 pitfall
per physics (e.g. dealii::obstacle_problem, kratos::pfem_solid),
max 21 (fourc::fsi).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


class TestCatalogPitfallsNonempty(unittest.TestCase):
    def test_every_physics_has_at_least_one_pitfall(self) -> None:
        from core.registry import all_backends, load_all_backends
        load_all_backends()

        empty: list[tuple[str, str]] = []
        for backend in all_backends():
            bk = backend.name()
            for cap in backend.supported_physics():
                k = backend.get_knowledge(cap.name)
                pitfalls = (k.get("pitfalls", [])
                            if isinstance(k, dict) else [])
                if not isinstance(pitfalls, list) or not pitfalls:
                    empty.append((bk, cap.name))

        self.assertEqual(
            empty, [],
            "Empty pitfalls lists found — LLMs get no warning "
            "for these (backend, physics) pairs:\n"
            + "\n".join(f"  {bk}::{phys}" for bk, phys in empty)
            + "\nEvery physics surfaced via supported_physics() "
            "must carry at least one [Category]+Signal: pitfall "
            "entry. Add a conservative '[Integration] this "
            "template is a probe / stub' entry if no specific "
            "pitfall is yet known.")


if __name__ == "__main__":
    unittest.main()
