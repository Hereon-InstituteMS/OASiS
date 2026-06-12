"""Regression: project instructions (CLAUDE.md, README.md) only
reference tools that the MCP server actually registers.

Catches the drift surfaced 2026-06-02: CLAUDE.md mentioned
`get_example_inputs()` (an older tool name from before the
consolidated.py refactor). The current registered tool is
`examples()`. An LLM reading CLAUDE.md and then trying to
call `get_example_inputs()` would hit "no such tool" and
get stuck.

Two reference styles are checked:
  - CLAUDE.md uses `tool(...)` form (backtick + parens) in
    instructions to sub-agents.
  - README.md uses bare `tool` form (backtick, no parens) in
    the tool-reference table.

Both are extracted and matched against the registered MCP
tool set; any phantom advertisement fails the gate.
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
    # README backtick-without-parens mentions of non-tool nouns:
    "oasis",        # MCP server name (not a tool)
}


class TestProjectInstructionsToolRefs(unittest.TestCase):
    def _check_file(self, path: Path, pattern: str,
                    description: str) -> None:
        self.assertTrue(path.is_file(), f"{description} not found at {path}")
        text = path.read_text()
        refs = set(re.findall(pattern, text))
        refs = {r for r in refs if r not in _NON_MCP_ALLOWED}
        registered = _registered_mcp_tools()
        phantoms = refs - registered
        self.assertEqual(
            phantoms, set(),
            f"{description} references tools that are NOT "
            f"registered: {sorted(phantoms)}. Registered MCP "
            f"tools: {sorted(registered)}. Either update "
            f"{description} to the current tool name OR re-register "
            "the missing tool in src/tools/consolidated.py.")

    def test_claude_md(self) -> None:
        """CLAUDE.md sub-agent instructions: `name(...)` form."""
        self._check_file(
            _REPO / "CLAUDE.md",
            r"`([a-z_][a-z_0-9]*)\([^`]*\)`",
            "CLAUDE.md")

    def test_readme_md(self) -> None:
        """README.md tool-reference table: bare `name` form.

        README backticks wrap many non-tool nouns (filenames,
        env vars, etc.); the regex extracts every backtick'd
        snake_case identifier, then we filter via heuristic: a
        registered MCP tool name is the canonical match set,
        anything else is implicitly trusted (only flag actual
        tool-name-shape phantoms)."""
        readme = _REPO / "README.md"
        self.assertTrue(readme.is_file(), f"README.md not found at {readme}")
        text = readme.read_text()
        refs = set(re.findall(r"`([a-z_][a-z_0-9]*)`", text))
        # README backticks many non-tool nouns; restrict to
        # snake-case identifiers that LOOK like tool names
        # (>= 6 chars, contain underscore OR match a known tool).
        registered = _registered_mcp_tools()
        tool_shaped = {r for r in refs
                       if r in registered
                       or (len(r) >= 6 and "_" in r
                           and r.endswith(("_simulation",
                                            "_solve",
                                            "_field",
                                            "_mesh",
                                            "_backends",
                                            "_catalog",
                                            "_insights",
                                            "_generator")))}
        tool_shaped -= _NON_MCP_ALLOWED
        phantoms = tool_shaped - registered
        self.assertEqual(
            phantoms, set(),
            f"README.md references tool-shaped names that are "
            f"NOT registered: {sorted(phantoms)}. Registered MCP "
            f"tools: {sorted(registered)}.")


if __name__ == "__main__":
    unittest.main()
