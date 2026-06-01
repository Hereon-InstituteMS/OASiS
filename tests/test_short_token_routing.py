"""Regression: every MCP tool surface that accepts a physics-like
query MUST route short canonical shorthands through the same
canonical resolver — _fuzzy_match_physics — so the synonym map gets
consulted before any loose substring scan.

Caught 2026-06-02:
  Even after the prepare_simulation / _fuzzy_match_physics fix, three
  other sites still did raw substring matching on physics name +
  description and silently routed 'ns' / 'em' / 'pd' to the wrong
  physics:

    examples(keyword='ns', solver='fenics', action='search')
       -> template scan matched heat / thermal_structural /
          reaction_diffusion / multiphase / time_dependent_heat
          (all contain 'ns' somewhere); navier_stokes was never
          surfaced.

    examples(keyword='ns', solver='fenics', action='template')
       -> raw 'in p.name' match. 'ns' is NOT a substring of
          'navier_stokes' (no adjacent 'ns' there), so 'ns'
          fell through to the first physics whose name
          contained 'ns' as a substring (typically transient_*).

    discover(query='recommend', solver='ns')
       -> per-backend substring scan returned heat in fenics,
          eigenvalue in fenics, ... whichever match was first.

This test exercises the FastMCP-registered tool functions directly
to pin the LLM-facing contract: short canonical shorthands must
land at the canonical physics, not at a coincidental substring.
"""
from __future__ import annotations

import asyncio
import re
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _registered_tools():
    """Return the FastMCP tool dict after registering consolidated
    tools. Skips cleanly if FastMCP isn't installed."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise unittest.SkipTest(f"FastMCP not installed: {exc}")
    from core.registry import load_all_backends
    from tools.consolidated import register_consolidated_tools

    load_all_backends()
    mcp = FastMCP("test")
    register_consolidated_tools(mcp)
    return mcp._tool_manager._tools  # type: ignore[attr-defined]


class TestShortTokenRouting(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tools = _registered_tools()
        # Ensure an event loop exists for async tool entries
        # (visualize / transfer_field / examples is sync).
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

    # ── examples('search', keyword='ns', solver='fenics') ──

    def test_examples_search_ns_finds_navier_stokes(self) -> None:
        fn = self.tools["examples"].fn
        result = fn(keyword="ns", solver="fenics",
                    action="search", max_results=3)
        matches = re.findall(r"### Template: `(\S+)`", result)
        self.assertTrue(matches,
                        f"examples(search, ns) returned no template "
                        f"match. Full result: {result[:500]}")
        # The first template returned must be navier_stokes.
        self.assertTrue(
            matches[0].startswith("navier_stokes"),
            f"examples(search, ns) first template was "
            f"{matches[0]!r}; expected navier_stokes/* — short "
            f"shorthand 'ns' should route through the synonym map "
            f"to navier_stokes, not via a coincidental substring "
            f"to heat / transient / reaction_diffusion.")

    # ── examples('template', keyword='ns', solver='fenics') ──

    def test_examples_template_ns_returns_navier_stokes(self) -> None:
        fn = self.tools["examples"].fn
        result = fn(keyword="ns", solver="fenics", action="template")
        # The returned template should contain Navier-Stokes
        # vocabulary (taylor-hood / navier / stokes / pressure
        # / velocity in fenics's NS template).
        lc = result.lower()
        self.assertIn(
            "navier", lc,
            f"examples(template, ns) did not return a Navier-"
            f"Stokes template; got first 200 chars: {result[:200]}")

    # ── discover('recommend', physics='ns') ──

    def test_discover_recommend_ns_lists_ns_solvers(self) -> None:
        fn = self.tools["discover"].fn
        result = fn(query="recommend", solver="ns")
        # Every backend that supports navier_stokes (fenics /
        # dealii / ngsolve / skfem at time of writing) should
        # appear in the recommendation list, and the lines
        # should mention Navier-Stokes (case-insensitive).
        self.assertIn("Navier-Stokes", result + " ",
                      f"discover(recommend, ns) recommendation "
                      f"did not mention Navier-Stokes; got: "
                      f"{result[:500]}")
        # And critically: it must NOT recommend heat / eigenvalue
        # / unrelated physics that previously matched via the
        # substring trap.
        forbidden_first_lines = [
            "heat equation", "eigenvalue problem",
        ]
        for bad in forbidden_first_lines:
            self.assertNotIn(
                bad, result.lower(),
                f"discover(recommend, ns) returned {bad!r} — short "
                f"shorthand 'ns' should route to navier_stokes via "
                f"synonym map, not match a description substring.")

    # ── 'em' shorthand ──────────────────────────────────────

    def test_examples_search_em_finds_maxwell(self) -> None:
        fn = self.tools["examples"].fn
        result = fn(keyword="em", solver="fenics",
                    action="search", max_results=3)
        matches = re.findall(r"### Template: `(\S+)`", result)
        if matches:
            self.assertTrue(
                matches[0].startswith("maxwell"),
                f"examples(search, em) first template was "
                f"{matches[0]!r}; expected maxwell/* — 'em' "
                f"shorthand should route via synonym map.")


if __name__ == "__main__":
    unittest.main()
