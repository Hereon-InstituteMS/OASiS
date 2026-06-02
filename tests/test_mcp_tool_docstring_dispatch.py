"""Regression: MCP-tool docstring enumerations match the actual
dispatch logic.

Generalised version of test_generate_mesh_docstring_alignment.py.
Several MCP tools dispatch on a string parameter via `if param ==
'X': ... elif param == 'Y': ...`. The docstring should enumerate
exactly the same X / Y / ... values. Drift causes:

  - Phantom (in docstring but no branch) → LLM tries an option
    that silently falls through to the default handler.
  - Unadvertised (in dispatch but not docstring) → LLM misses
    a real capability.

This test walks each @mcp.tool() function's body, extracts every
`if param == "X"` / `elif param == "X"` literal from the dispatch
chain, and asserts it equals the bullet-list set in the docstring
(`- "X" — ...`).

Pinned tools:
  • discover(query=...)            — list/physics/capabilities/recommend
  • examples(action=...)           — search/template/tutorials
  • developer(action=...)          — architecture/files/capabilities
  • knowledge(topic=...)           — physics/pitfalls/postmortems/
                                     materials/overview/coupling/tsi/
                                     precice/input_guide/solver_guidance/
                                     hardware
  • visualize(action=...)          — summary/list/plot/validate
  • session_insights(action=...)   — review/ingest/approve_all/reject_all/stats
  • transfer_field(target_format=...) — json/fenics/4c_neumann

If a tool's dispatch chain doesn't match this exact pattern (or
uses a literal dispatch dict), it's audited by a separate gate or
not yet covered.
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _find_tool_function(tree: ast.AST, name: str):
    """Match both sync `def` and `async def` tools."""
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name):
            return node
    raise AssertionError(f"tool function {name} not found")


def _docstring_bullets(func, param_name: str) -> set[str]:
    """Pull `- "value" — ...` bullets from the section of the
    docstring that follows the `<param_name>:` Args entry.

    Stops at the next Args entry (a top-level `name:` line) so
    bullets that belong to a different parameter aren't mixed in.
    """
    doc = ast.get_docstring(func) or ""
    lines = doc.splitlines()
    capture = False
    captured = []
    for line in lines:
        stripped = line.strip()
        if not capture:
            if stripped.startswith(f"{param_name}:"):
                capture = True
            continue
        # Stop at next Args entry: a line shaped `<name>:` at
        # the same or shallower indent than `param_name:`. The
        # bullet-list is indented deeper, so we just detect any
        # line that isn't a bullet or a continuation.
        if stripped and not stripped.startswith("-") and ":" in stripped:
            # Heuristic: if the first whitespace-token ends with
            # `:`, this is a new arg entry — stop.
            head = stripped.split()[0]
            if head.endswith(":") and not head.startswith('"'):
                break
        captured.append(line)
    text = "\n".join(captured)
    return set(re.findall(r'-\s+"([^"]+)"', text))


def _dispatch_string_constants(func, param_name: str) -> set[str]:
    """Walk the function body looking for `<param> == "X"`
    comparisons. Returns the set of X-values."""
    out: set[str] = set()
    for inner in ast.walk(func):
        if isinstance(inner, ast.Compare):
            left = inner.left
            if (isinstance(left, ast.Name) and left.id == param_name
                    and len(inner.ops) == 1
                    and isinstance(inner.ops[0], ast.Eq)
                    and len(inner.comparators) == 1):
                cmp = inner.comparators[0]
                if isinstance(cmp, ast.Constant) and isinstance(cmp.value, str):
                    out.add(cmp.value)
    return out


class TestMcpToolDocstringDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        src = (_REPO / "src" / "tools" / "consolidated.py").read_text()
        cls.tree = ast.parse(src)

    def _check(self, tool_name: str, param_name: str) -> None:
        func = _find_tool_function(self.tree, tool_name)
        documented = _docstring_bullets(func, param_name)
        dispatched = _dispatch_string_constants(func, param_name)
        self.assertEqual(
            documented, dispatched,
            f"{tool_name}({param_name}=...) docstring drifted from "
            "dispatch chain.\n"
            f"  documented: {sorted(documented)}\n"
            f"  dispatched: {sorted(dispatched)}\n"
            f"  diff:       {sorted(documented ^ dispatched)}\n"
            "Edit src/tools/consolidated.py docstring + dispatch "
            "in lock-step.")

    def test_discover_query(self) -> None:
        self._check("discover", "query")

    def test_examples_action(self) -> None:
        self._check("examples", "action")

    def test_developer_action(self) -> None:
        self._check("developer", "action")

    def test_knowledge_topic(self) -> None:
        self._check("knowledge", "topic")

    def test_visualize_action(self) -> None:
        self._check("visualize", "action")

    def test_session_insights_action(self) -> None:
        self._check("session_insights", "action")

    def test_transfer_field_target_format(self) -> None:
        self._check("transfer_field", "target_format")


if __name__ == "__main__":
    unittest.main()
