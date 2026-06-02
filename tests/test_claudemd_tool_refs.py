"""Regression: CLAUDE.md only references tools that the MCP
server actually registers.

Catches the drift surfaced 2026-06-02: CLAUDE.md mentioned
`get_example_inputs()` (an older tool name from before the
consolidated.py refactor). The current registered tool is
`examples()`. An LLM reading CLAUDE.md and then trying to
call `get_example_inputs()` would hit "no such tool" and
get stuck.

The gate parses CLAUDE.md, extracts every `tool_name(...)`
mention (backtick-quoted with parens), and asserts each
matches a registered MCP tool name.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _registered_mcp_tools() -> set[str]:
    """Return the set of @mcp.tool() names register_consolidated_tools
    sets up on a fresh FastMCP."""
    from mcp.server.fastmcp import FastMCP  # type: ignore
    from tools.consolidated import register_consolidated_tools
    from core.registry import load_all_backends
    load_all_backends()
    mcp = FastMCP("test")
    register_consolidated_tools(mcp)
    return set(mcp._tool_manager._tools.keys())


# Generic Python builtins / external references that the LLM
# instructions mention as plain prose, not as MCP tools. The
# regex pattern matches `name(...)` and could false-positive
# on any function call mention; this allow-list whitelists
# the legitimate non-MCP references.
_NON_MCP_ALLOWED = {
    "X",                     # placeholder in prose
    "APPROVE", "REJECT",     # critic verdict tokens
    "search",                # verb / generic
}


class TestClaudeMdToolRefs(unittest.TestCase):
    def test_referenced_tools_are_registered(self) -> None:
        claudemd = _REPO / "CLAUDE.md"
        self.assertTrue(claudemd.is_file(),
                        f"CLAUDE.md not found at {claudemd}")
        text = claudemd.read_text()

        # Extract every `name(...)` mention. The backticks
        # rule out prose-only callbacks like "the X function".
        # Allow the form `name(arg=...)` and `name()`.
        refs = set(re.findall(r"`([a-z_][a-z_0-9]*)\([^`]*\)`", text))

        # Filter out backtick references that are obviously
        # generic.
        refs = {r for r in refs if r not in _NON_MCP_ALLOWED}

        registered = _registered_mcp_tools()
        phantoms = refs - registered
        self.assertEqual(
            phantoms, set(),
            f"CLAUDE.md references tools that are NOT registered: "
            f"{sorted(phantoms)}. Registered MCP tools: "
            f"{sorted(registered)}. Either update CLAUDE.md to the "
            "current tool name OR re-register the missing tool in "
            "src/tools/consolidated.py.")


if __name__ == "__main__":
    unittest.main()
