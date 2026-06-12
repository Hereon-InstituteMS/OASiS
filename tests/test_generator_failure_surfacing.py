"""Regression: when a catalog generator raises, prepare_simulation
and examples('search') MUST surface the failure to the LLM —
NOT silently return a "successful" response with no template.

Caught 2026-06-02:
  src/tools/consolidated.py wrapped backend.generate_input() in
  `except Exception: pass` at two LLM-facing sites:

    - prepare_simulation() line 1438
    - examples(action='search')      line 743

  When a generator raised (the Layer-F class of bug), the LLM
  saw a normal-looking response that simply lacked the
  ``## Template`` section. Neither the LLM nor the developer
  running the tool had any way to learn the generator had
  blown up.

This test patches one backend's generate_input to raise a
distinctive RuntimeError, then asserts that BOTH tool paths
expose a "Template generation FAILED" block with the exception
type/message — closing the silent-degradation hole.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_SENTINEL = "synthetic generator failure for test 2026-06-02"


def _registered_tools_and_backend(solver: str):
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise unittest.SkipTest(f"FastMCP not installed: {exc}")
    from core.registry import load_all_backends, get_backend
    from tools.consolidated import register_consolidated_tools

    load_all_backends()
    backend = get_backend(solver)
    if backend is None:
        raise unittest.SkipTest(f"backend {solver} not registered")
    mcp = FastMCP("test")
    register_consolidated_tools(mcp)
    return mcp._tool_manager._tools, backend  # type: ignore[attr-defined]


class TestGeneratorFailureSurfacing(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tools, cls.backend = _registered_tools_and_backend("fenics")
        cls._orig_gen = cls.backend.generate_input

        def _boom(physics, variant, params):
            raise RuntimeError(_SENTINEL)

        cls.backend.generate_input = _boom  # type: ignore[method-assign]

    @classmethod
    def tearDownClass(cls) -> None:
        cls.backend.generate_input = cls._orig_gen  # type: ignore[method-assign]

    # ── prepare_simulation: failure must be visible ─────────

    def test_prepare_simulation_surfaces_generator_failure(self) -> None:
        fn = self.tools["prepare_simulation"].fn
        result = fn(solver="fenics", physics="poisson")
        self.assertIn(
            "Template generation FAILED", result,
            "prepare_simulation silently swallowed a generator "
            "failure. The LLM cannot distinguish 'no template' "
            "from 'generator crashed'.")
        self.assertIn(
            _SENTINEL, result,
            "Failure block does not include the exception "
            "message — LLM has no diagnostic info.")
        self.assertIn(
            "RuntimeError", result,
            "Failure block does not include the exception type.")
        # Other sections must still render.
        self.assertIn(
            "## Knowledge", result,
            "Knowledge section vanished alongside the failed "
            "template; the failure block must NOT cause the "
            "whole response to short-circuit.")

    # ── examples('search'): failure must be visible ─────────

    def test_examples_search_surfaces_generator_failure(self) -> None:
        fn = self.tools["examples"].fn
        result = fn(keyword="poisson", solver="fenics",
                    action="search", max_results=3)
        self.assertIn(
            "Template generation FAILED", result,
            "examples('search') silently swallowed a generator "
            "failure.")
        self.assertIn(
            _SENTINEL, result,
            "Failure block in examples('search') does not "
            "include the exception message.")


if __name__ == "__main__":
    unittest.main()
