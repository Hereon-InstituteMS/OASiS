"""Regression: every physics advertised by `supported_physics()`
must have a non-empty `get_knowledge()` entry with at least the
`description` + `pitfalls` keys populated.

Caught (and prevented going forward by this test) 2026-06-02:
  The audit pass found 198 physics rows across 8 backends. Each
  one already returned a populated knowledge dict — but the
  contract was implicit. Without this test, a new physics row
  added with an empty/missing knowledge entry would silently
  produce a thin prepare_simulation response (knowledge section
  hidden by the no-such-key path, pitfalls section absent), and
  the LLM would have no way to tell that the physics was
  added without the supporting knowledge.

This test pins:
  1. b.get_knowledge(p.name) for every (backend, p in
     supported_physics()) returns a non-None dict.
  2. The dict has a non-empty `description` value.
  3. The dict has a non-empty `pitfalls` list.

When a real new physics ships with `description = ""` or no
pitfalls, fix the generator file in
src/backends/<be>/generators/<phys>.py, not this test.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


class TestKnowledgeDictCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, all_backends
        load_all_backends()
        cls.backends = all_backends()
        if not cls.backends:
            raise unittest.SkipTest("no backends registered")

    def test_every_physics_has_knowledge_entry(self) -> None:
        """get_knowledge(physics_name) must return a non-empty dict
        for every entry in supported_physics()."""
        failures = []
        for b in self.backends:
            for p in b.supported_physics():
                try:
                    k = b.get_knowledge(p.name)
                except Exception as e:
                    failures.append(
                        (b.name(), p.name,
                         f"raised: {type(e).__name__}: {e}"))
                    continue
                if k is None:
                    failures.append((b.name(), p.name, "returned None"))
                elif not isinstance(k, dict):
                    failures.append(
                        (b.name(), p.name,
                         f"not a dict, got {type(k).__name__}"))
                elif not k:
                    failures.append((b.name(), p.name, "empty dict"))
        if failures:
            lines = "\n".join(
                f"  {be}/{ph}: {err}" for be, ph, err in failures)
            self.fail(
                f"{len(failures)} physics row(s) without knowledge:\n"
                f"{lines}")

    def test_every_physics_has_description(self) -> None:
        """Knowledge dict must contain a non-empty `description`
        field. prepare_simulation surfaces this as the first
        block; absent / empty -> LLM has no idea what the
        physics actually is."""
        failures = []
        for b in self.backends:
            for p in b.supported_physics():
                k = b.get_knowledge(p.name)
                if not isinstance(k, dict):
                    continue  # covered by the other test
                desc = k.get("description")
                if not desc or not isinstance(desc, str) or not desc.strip():
                    failures.append((b.name(), p.name, repr(desc)))
        if failures:
            lines = "\n".join(
                f"  {be}/{ph}: description={d}" for be, ph, d in failures)
            self.fail(
                f"{len(failures)} physics row(s) missing description:\n"
                f"{lines}")

    def test_every_physics_has_pitfalls(self) -> None:
        """Knowledge dict must carry a non-empty `pitfalls` list.
        This is the post-execution-critic-actionable Table-1
        promoted content; an empty pitfalls list means the
        backend admits no known pitfalls for that physics,
        which is almost never true and is usually a missing-
        knowledge regression."""
        failures = []
        for b in self.backends:
            for p in b.supported_physics():
                k = b.get_knowledge(p.name)
                if not isinstance(k, dict):
                    continue
                pf = k.get("pitfalls")
                if pf is None:
                    failures.append((b.name(), p.name, "no key"))
                elif not isinstance(pf, list):
                    failures.append((b.name(), p.name,
                                     f"not a list, got {type(pf).__name__}"))
                elif len(pf) == 0:
                    failures.append((b.name(), p.name, "empty list"))
        if failures:
            lines = "\n".join(
                f"  {be}/{ph}: pitfalls {err}" for be, ph, err in failures)
            self.fail(
                f"{len(failures)} physics row(s) without pitfalls:\n"
                f"{lines}")


if __name__ == "__main__":
    unittest.main()
