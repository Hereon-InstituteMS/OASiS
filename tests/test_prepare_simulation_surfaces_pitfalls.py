"""Regression test for the prepare_simulation truncation bug.

Catalog audit 2026-06-01: get_knowledge() returns the FULL pitfall
list, but prepare_simulation in src/tools/consolidated.py used to
dump the whole dict as pretty-printed JSON and slice [:3000]. For
the larger physics blocks (ngsolve::hyperelasticity ~4.4 KB,
skfem::poisson ~4 KB, ngsolve::dg_methods ~2.4 KB with 10
pitfalls) the truncation hid every late-list pitfall from the
LLM-visible surface — meaning every Layer F fix landed in the
catalog but never reached prepare_simulation's response.

The fix renders pitfalls as a separate bulleted section AFTER
the JSON dump so the truncation does not eat them. This test
pins that contract: prepare_simulation's output must contain
every pitfall string verbatim for at least one of the worst-
truncation-hit physics blocks.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


class _StubMCP:
    """Minimal MCP stand-in that lets us register the tools and
    grab the underlying functions for direct unit-test use."""

    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def _deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return _deco

    def resource(self, *args, **kwargs):
        def _deco(fn):
            return fn
        return _deco

    def prompt(self, *args, **kwargs):
        def _deco(fn):
            return fn
        return _deco


def _prepare_simulation_fn():
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _StubMCP()
    register_consolidated_tools(mcp)
    return mcp.tools["prepare_simulation"]


class TestPrepareSimulationSurfacesAllPitfalls(unittest.TestCase):
    """Every pitfall in a worst-case-large KNOWLEDGE block must
    appear verbatim in the prepare_simulation response."""

    # Physics blocks that previously had > 3000 chars of total
    # KNOWLEDGE JSON, hiding pitfalls. Picked from the catalog
    # audit on 2026-06-01.
    WORST_CASES = [
        ("ngsolve", "hyperelasticity"),
        ("ngsolve", "dg_methods"),
        ("skfem",   "poisson"),
        ("skfem",   "mixed_poisson"),
    ]

    def test_all_pitfalls_present(self) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        prepare = _prepare_simulation_fn()
        for solver, physics in self.WORST_CASES:
            with self.subTest(solver=solver, physics=physics):
                backend = get_backend(solver)
                self.assertIsNotNone(backend, f"backend {solver} unavailable")
                k = backend.get_knowledge(physics)
                self.assertIsInstance(k, dict)
                pitfalls = k.get("pitfalls", [])
                self.assertTrue(pitfalls,
                                f"{solver}::{physics} has no pitfalls — "
                                "audit the KNOWLEDGE block")
                text = prepare(solver=solver, physics=physics)
                self.assertIsInstance(text, str)
                missing = [p for p in pitfalls if p not in text]
                if missing:
                    self.fail(
                        f"{solver}::{physics}: "
                        f"{len(missing)}/{len(pitfalls)} pitfalls "
                        f"missing from prepare_simulation output. "
                        f"First missing: {missing[0][:100]!r}. "
                        "Likely cause: the json.dumps([:3000]) "
                        "truncation in src/tools/consolidated.py "
                        "is back — render pitfalls separately.")


if __name__ == "__main__":
    unittest.main()
