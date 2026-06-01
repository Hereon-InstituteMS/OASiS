"""Regression: knowledge('hardware') must cover EVERY registered
backend.

Caught 2026-06-02:
  The hardware-knowledge dict in src/tools/consolidated.py listed
  7 backends (FEniCSx, deal.II, 4C, NGSolve, scikit-fem, Kratos,
  DUNE-fem) but the MCP advertises 8 — FEBio was missing both
  from the dict and from the key_map. An LLM that asked
  knowledge('hardware', solver='febio') got back "No hardware
  info for febio", with no indication that the MCP otherwise
  fully supports FEBio. Same alignment-drift class as the
  discover('list') hide-unavailable hole.

This test enforces:
  1. Every registered backend has a hardware-knowledge entry
     (so the LLM never gets "No hardware info" for a backend
     the MCP otherwise lists).
  2. Every primary canonical name in the registry is reachable
     via the key_map normalization (so passing the
     short-form name like 'febio' resolves to the dict key).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _registered_tools():
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


class TestHardwareKnowledgeCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tools = _registered_tools()
        from core.registry import all_backends
        cls.backends = all_backends()
        if not cls.backends:
            raise unittest.SkipTest("no backends registered")

    def test_every_backend_has_hardware_entry(self) -> None:
        """knowledge('hardware', solver=<name>) must return a
        dict for every registered backend name — not a 'No
        hardware info' string."""
        fn = self.tools["knowledge"].fn
        missing = []
        for b in self.backends:
            name = b.name()
            result = fn(topic="hardware", solver=name)
            # The success path returns a JSON object literal
            # starting with `{`. The failure path returns a
            # plain string "No hardware info for ...".
            if not result.strip().startswith("{"):
                missing.append((name, result.strip()[:120]))
        if missing:
            lines = "\n".join(
                f"  - {n}: {r!r}" for n, r in missing)
            self.fail(
                f"knowledge('hardware') is missing entries for "
                f"{len(missing)} backend(s):\n{lines}\n\n"
                "Either add a hardware entry to the `hw` dict in "
                "src/tools/consolidated.py (knowledge() handler) "
                "OR add the backend's short name to the key_map "
                "in the same block so it resolves to an existing "
                "display-name key.")


if __name__ == "__main__":
    unittest.main()
