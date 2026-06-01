"""Regression: prepare_simulation must NOT silently map empty
physics queries to the first supported physics.

Audit 2026-06-01: _fuzzy_match_physics checked
``if query_lower in p.name.lower()`` against every physics
name. An empty / whitespace-only query is a substring of every
string, so the very first physics in supported_physics matched.
prepare_simulation then built a full response for poisson when
the LLM had asked for nothing.

Fix: empty/whitespace input now short-circuits to the
"Empty physics query" path, surfacing the available-physics
list. This test pins that contract.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


class _StubMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco

    def resource(self, *args, **kwargs):
        def deco(fn):
            return fn
        return deco

    def prompt(self, *args, **kwargs):
        def deco(fn):
            return fn
        return deco


def _prepare_simulation_fn():
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _StubMCP()
    register_consolidated_tools(mcp)
    return mcp.tools["prepare_simulation"]


class TestEmptyPhysicsQuerySurfacesAvailableList(unittest.TestCase):
    """prepare_simulation must distinguish between an empty
    physics query and a real one."""

    EMPTY_INPUTS = ("", "   ", "\t", "\n")

    def test_empty_inputs(self) -> None:
        prepare = _prepare_simulation_fn()
        for q in self.EMPTY_INPUTS:
            with self.subTest(query=repr(q)):
                r = prepare(solver="fenics", physics=q)
                self.assertIsInstance(r, str)
                self.assertIn(
                    "Empty physics query", r,
                    f"prepare_simulation('fenics', {q!r}) "
                    "should flag empty query and surface "
                    "the available-physics list. Got: "
                    f"{r[:200]!r}")
                # Sanity: the available-list must be non-empty
                self.assertIn("poisson", r)


def _examples_fn():
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _StubMCP()
    register_consolidated_tools(mcp)
    return mcp.tools["examples"]


class TestEmptyKeywordExamplesGuarded(unittest.TestCase):
    """examples tool must NOT silently return arbitrary files
    when the keyword is empty (substring-of-everything bug)."""

    EMPTY_INPUTS = ("", "   ", "\t", "\n")

    def test_search_empty(self) -> None:
        ex = _examples_fn()
        for q in self.EMPTY_INPUTS:
            with self.subTest(query=repr(q), action="search"):
                r = ex(keyword=q, solver="fenics", action="search")
                self.assertIn(
                    "Empty keyword", r,
                    f"examples(keyword={q!r}, action='search') "
                    "should reject empty keyword, got: "
                    f"{r[:200]!r}")

    def test_template_empty(self) -> None:
        ex = _examples_fn()
        for q in self.EMPTY_INPUTS:
            with self.subTest(query=repr(q), action="template"):
                r = ex(keyword=q, solver="fenics", action="template")
                self.assertIn(
                    "Empty keyword", r,
                    f"examples(keyword={q!r}, action='template') "
                    "should reject empty keyword, got: "
                    f"{r[:200]!r}")


if __name__ == "__main__":
    unittest.main()
