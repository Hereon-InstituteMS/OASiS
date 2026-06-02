"""Regression: prepare_simulation handles adversarial inputs gracefully.

Adversarial-fuzzing pass (task #40) — probe corner cases that an
LLM might emit by accident:

  - empty / whitespace-only physics    — must NOT silently
                                          fuzzy-match the first
                                          physics (task #136 root
                                          cause). Must surface the
                                          available-physics list
                                          so the LLM can self-correct.
  - unknown solver                     — clear "Unknown solver: X"
                                          error.
  - canonical aliases (Poisson, elasticity)
                                       — must be matched via the
                                          fuzzy resolver with a
                                          *Note:* breadcrumb.

The gate calls the underlying prepare_simulation function (NOT the
MCP wire) so the test passes regardless of MCP server staleness.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _invoke_prepare_simulation(solver: str, physics: str) -> str:
    """Call the FUNCTION body inside register_workflow_tools.

    consolidated.py wraps the function with @mcp.tool() inside a
    nested closure, so a direct import doesn't expose it. Re-
    register the tools on a fresh FastMCP instance and pull the
    underlying callable from there.
    """
    from mcp.server.fastmcp import FastMCP  # type: ignore
    from tools.consolidated import register_consolidated_tools
    from core.registry import load_all_backends
    load_all_backends()

    mcp = FastMCP("test")
    register_consolidated_tools(mcp)
    # FastMCP stores registered tools on internal dict — grab via
    # the public `_tool_manager._tools` API path that mcp uses.
    tool_mgr = mcp._tool_manager
    tool = tool_mgr._tools["prepare_simulation"]
    return tool.fn(solver=solver, physics=physics)


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


if __name__ == "__main__":
    unittest.main()
