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


def _module_string_sets(tree) -> dict[str, set[str]]:
    """Module-level names bound to a set/frozenset of string literals.

    A dispatch branch may legitimately be written `topic in _SOME_ALIASES`
    rather than `topic == "x"`, when one branch answers to several spellings.
    Without this the drift check would report the branch as undispatched and
    push the author to either delete the aliases or stop documenting the
    topic — both worse than the thing the check exists to prevent."""
    out: dict[str, set[str]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        # frozenset({...}) / set({...}) wrapper, or a bare set literal.
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
                and value.func.id in ("frozenset", "set") and value.args):
            value = value.args[0]
        if isinstance(value, (ast.Set, ast.List, ast.Tuple)):
            items = {e.value for e in value.elts
                     if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            if items:
                out[target.id] = items
    return out


def _dispatch_string_constants(func, param_name: str,
                               string_sets: dict[str, set[str]] | None = None
                               ) -> set[str]:
    """Walk the function body looking for `<param> == "X"` comparisons and
    `<param> in <NAME>` membership tests against a module-level set of string
    literals. Returns the set of X-values either form can select."""
    string_sets = string_sets or {}
    out: set[str] = set()
    for inner in ast.walk(func):
        if not isinstance(inner, ast.Compare):
            continue
        left = inner.left
        if not (isinstance(left, ast.Name) and left.id == param_name
                and len(inner.ops) == 1 and len(inner.comparators) == 1):
            continue
        op, cmp = inner.ops[0], inner.comparators[0]
        if isinstance(op, ast.Eq):
            if isinstance(cmp, ast.Constant) and isinstance(cmp.value, str):
                out.add(cmp.value)
        elif isinstance(op, ast.In):
            if isinstance(cmp, ast.Name) and cmp.id in string_sets:
                out |= string_sets[cmp.id]
            elif isinstance(cmp, (ast.Set, ast.List, ast.Tuple)):
                out |= {e.value for e in cmp.elts
                        if isinstance(e, ast.Constant)
                        and isinstance(e.value, str)}
    return out


class TestMcpToolDocstringDispatch(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        src = (_REPO / "src" / "tools" / "consolidated.py").read_text()
        cls.tree = ast.parse(src)
        cls.string_sets = _module_string_sets(cls.tree)

    def _check(self, tool_name: str, param_name: str) -> None:
        func = _find_tool_function(self.tree, tool_name)
        documented = _docstring_bullets(func, param_name)
        dispatched = _dispatch_string_constants(func, param_name,
                                                self.string_sets)
        # BOTH directions still matter, but aliases are handled asymmetrically.
        #
        # A DOCUMENTED value with no branch is a lie to the caller — always a
        # failure. A DISPATCHED value with no docstring bullet is normally a
        # feature nobody can find, so it fails too; the exception is a member
        # of an alias set, where the docstring names the canonical spelling and
        # the synonyms exist so a caller who guesses a near-miss still lands on
        # the right branch. Requiring every synonym its own bullet would make
        # the docstring worse, not better.
        aliases = set().union(*self.string_sets.values()) if self.string_sets \
            else set()
        undispatched = documented - dispatched
        undocumented = dispatched - documented - aliases
        self.assertFalse(
            undispatched,
            f"{tool_name}({param_name}=...) documents a value the dispatch "
            f"chain does not handle.\n"
            f"  documented but not dispatched: {sorted(undispatched)}\n"
            "Edit src/tools/consolidated.py docstring + dispatch "
            "in lock-step.")
        self.assertFalse(
            undocumented,
            f"{tool_name}({param_name}=...) dispatches a value the docstring "
            f"never mentions — callers cannot discover it.\n"
            f"  dispatched but not documented: {sorted(undocumented)}\n"
            "Edit src/tools/consolidated.py docstring + dispatch "
            "in lock-step.")
        # The canonical spelling of an alias group must still be documented,
        # or the whole group is undiscoverable.
        for name, members in self.string_sets.items():
            if members & dispatched and not (members & documented):
                self.fail(
                    f"{tool_name}({param_name}=...) dispatches the alias group "
                    f"{name} but the docstring documents none of its "
                    f"spellings {sorted(members)} — the branch is "
                    f"unreachable by anyone reading the docs.")

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
