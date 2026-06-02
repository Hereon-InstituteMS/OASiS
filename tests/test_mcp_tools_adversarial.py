"""Regression: MCP tools handle adversarial inputs gracefully.

Adversarial-fuzzing pass (task #40) — probe corner cases that an
LLM might emit by accident. Coverage:

  prepare_simulation:
    - empty / whitespace-only physics    — must NOT silently
                                            fuzzy-match the first
                                            physics (task #136 root
                                            cause).
    - unknown solver                     — clear "Unknown solver: X"
                                            error.
    - canonical aliases (Poisson, elasticity)
                                         — matched via fuzzy
                                            resolver with *Note:*
                                            breadcrumb.
  discover:
    - query='recommend' with empty solver/physics — task #150
                                            rejection contract.
  examples:
    - empty keyword                      — task #137 guard.

The gate calls registered handlers directly (NOT the MCP wire)
so the test passes regardless of MCP server staleness.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _get_tool(tool_name: str):
    """Re-register all consolidated tools on a fresh FastMCP
    instance and return the underlying callable for `tool_name`."""
    from mcp.server.fastmcp import FastMCP  # type: ignore
    from tools.consolidated import register_consolidated_tools
    from core.registry import load_all_backends
    load_all_backends()

    mcp = FastMCP("test")
    register_consolidated_tools(mcp)
    return mcp._tool_manager._tools[tool_name].fn


def _invoke_prepare_simulation(solver: str, physics: str) -> str:
    return _get_tool("prepare_simulation")(solver=solver,
                                            physics=physics)


class TestPrepareSimulationAdversarial(unittest.TestCase):
    def test_empty_physics_surfaces_available(self) -> None:
        result = _invoke_prepare_simulation("fenics", "")
        self.assertIn("Empty physics query", result,
                      "Empty physics must surface a usage hint, "
                      "NOT silently fuzzy-match the first physics. "
                      f"Got: {result[:200]}")
        self.assertIn("poisson", result.lower(),
                      "The error message must list available physics "
                      "so the LLM can self-correct.")

    def test_whitespace_physics_surfaces_available(self) -> None:
        result = _invoke_prepare_simulation("fenics", "   ")
        self.assertIn("Empty physics query", result,
                      "Whitespace-only physics must be rejected the "
                      "same way as truly empty input.")

    def test_unknown_solver_clean_error(self) -> None:
        result = _invoke_prepare_simulation("does_not_exist", "poisson")
        self.assertIn("Unknown solver", result,
                      "Unknown solver must produce a clean error, "
                      f"not crash. Got: {result[:200]}")

    def test_uppercase_physics_fuzzy_matches(self) -> None:
        result = _invoke_prepare_simulation("fenics", "Poisson")
        self.assertIn("matched to 'poisson'", result,
                      "Case-insensitive physics names must be fuzzy-"
                      "matched with a *Note:* breadcrumb so the LLM "
                      "sees what canonical name was used.")

    def test_alias_physics_fuzzy_matches(self) -> None:
        result = _invoke_prepare_simulation("fenics", "elasticity")
        self.assertIn("matched to 'linear_elasticity'", result,
                      "Known aliases (elasticity → linear_elasticity) "
                      "must surface a *Note:* breadcrumb.")


class TestDiscoverAdversarial(unittest.TestCase):
    """discover('recommend') with empty payload (task #150)."""

    def test_recommend_empty_physics_rejected(self) -> None:
        discover = _get_tool("discover")
        result = discover(query="recommend", solver="")
        self.assertIn("Empty physics", result,
                      "discover(query='recommend', solver='') must "
                      "be rejected — task #150. Got: "
                      f"{result[:200]}")


class TestExamplesAdversarial(unittest.TestCase):
    """examples with empty keyword (task #137)."""

    def test_empty_keyword_rejected(self) -> None:
        examples = _get_tool("examples")
        result = examples(keyword="", solver="fenics", action="search")
        # The guard wording may vary; the contract is that the
        # tool surfaces a usage hint or available-examples list
        # rather than silently returning random matches via the
        # fuzzy-resolver's '' substring match.
        lower = result.lower()
        self.assertTrue(
            ("usage" in lower) or ("provide" in lower)
            or ("specify" in lower) or ("empty" in lower)
            or ("keyword" in lower),
            "examples(keyword='') must guard against the empty "
            "substring match (task #137). Got: "
            f"{result[:200]}")


if __name__ == "__main__":
    unittest.main()
