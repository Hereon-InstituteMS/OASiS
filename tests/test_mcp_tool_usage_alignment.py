"""Regression: every MCP tool's docstring + usage-hint + actual
dispatch branches must agree on the list of accepted actions /
topics.

Audit 2026-06-01 found TWO instances of the same drift class:

  session_insights:
    docstring listed {review, ingest, approve_all, reject_all,
                       stats}
    usage  listed {review, approve_all, reject_all, stats}
                  -- missing 'ingest'

  knowledge:
    docstring listed {physics, pitfalls, postmortems, materials,
                       coupling, tsi, precice, input_guide,
                       solver_guidance, hardware}
    usage  listed all except 'postmortems'

Both led to LLMs hitting an invalid action getting a usage hint
that pruned a real branch — they'd never learn the missing
action exists.

This test pins the contract by inspecting the source file
directly (regex-based, no MCP server needed):
  (1) every action/topic mentioned in the dispatch branches
      (`if action == "foo"` / `elif topic == "bar"`) appears in
      the corresponding usage-hint string.
  (2) every action/topic listed in the usage-hint string has a
      dispatch branch.

Hand-curated rules per tool because their dispatch shapes
differ (some use action=, some topic=, etc.).
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _read_consolidated() -> str:
    return (Path(__file__).resolve().parent.parent /
            "src" / "tools" / "consolidated.py").read_text()


def _branches(src: str, varname: str) -> set[str]:
    """Find every `(if|elif) <varname> == "<value>"` constant
    in the source. Picks out the action/topic dispatch."""
    pattern = re.compile(
        rf'(?:if|elif)\s+{varname}\s*==\s*"([^"]+)"')
    return set(pattern.findall(src))


class TestMcpToolUsageAlignment(unittest.TestCase):
    """Docstring/usage/dispatch must agree for each tool's
    action/topic enumeration."""

    def test_session_insights(self) -> None:
        src = _read_consolidated()
        # Carve out the session_insights function body.
        body = src.split("def session_insights")[1].split("# Storage for pending")[0]
        branches = _branches(body, "action")
        # Usage-hint string lives in this function — extract.
        usage_match = re.search(r'Actions:\s*([^"]+)', body)
        self.assertIsNotNone(
            usage_match,
            "session_insights has no 'Actions:' usage line.")
        # The captured CSV may end with a trailing literal '\n'
        # escape (the Actions: line is the last item before the
        # closing quote) — strip it before splitting on commas.
        actions_csv = (usage_match.group(1)
                       .replace("\\n", " ")
                       .replace("\n", " "))
        usage_actions = set(
            a.strip() for a in actions_csv.split(",")
            if a.strip())
        missing = branches - usage_actions
        self.assertFalse(
            missing,
            f"session_insights dispatch branches "
            f"{sorted(branches)} include {sorted(missing)} not "
            f"in usage hint {sorted(usage_actions)}. Update "
            "the Actions: line in src/tools/consolidated.py.")
        # Reverse direction: every advertised action must have
        # a branch.
        phantom = usage_actions - branches
        self.assertFalse(
            phantom,
            f"session_insights usage hint advertises "
            f"{sorted(phantom)} but no dispatch branch handles "
            "them.")

    def test_knowledge(self) -> None:
        src = _read_consolidated()
        body = src.split("def knowledge")[1].split("# 2. DISCOVER")[0]
        branches = _branches(body, "topic")
        # The usage-hint Topics list is split across multiple
        # adjacent string literals for line length. Find the
        # "Topics: ..." block by grabbing everything from
        # "Topics:" until the closing ).
        usage_block_match = re.search(
            r'Topics:[\s\S]+?\)', body)
        self.assertIsNotNone(
            usage_block_match,
            "knowledge has no 'Topics:' usage block.")
        usage_block = usage_block_match.group(0)
        # Strip the Python string-concat noise: each adjacent
        # string ends with `"\n` and the next starts with `"`.
        # The pattern '"\s+"' collapses that boundary; then
        # remove the remaining quotes / newlines / backslashes.
        flat = re.sub(r'"\s+"', "", usage_block)
        flat = flat.replace('"', "").replace("\\n", "")
        flat = flat.replace("\n", " ")
        topic_csv = flat.split("Topics:", 1)[1].split(")")[0]
        usage_topics = set(
            t.strip() for t in topic_csv.split(",")
            if t.strip())
        missing = branches - usage_topics
        self.assertFalse(
            missing,
            f"knowledge dispatch branches {sorted(branches)} "
            f"include {sorted(missing)} not in usage hint "
            f"{sorted(usage_topics)}.")


class TestNoPhantomMcpTools(unittest.TestCase):
    """The LLM-facing surfaces (server instructions + catalog
    template comments + tool docstrings) MUST NOT reference
    MCP tool names that no longer exist as callable handles.

    Caught 2026-06-01: server.py instructions still cited
    get_example_inputs(), get_solver_architecture(), and
    browse_solver_tests() — none of which are registered with
    the FastMCP server (only register_consolidated_tools is).
    LLMs reading the instructions and dutifully calling those
    names hit a 'no such tool' error.
    """

    # Tool names that USED to exist before the consolidation
    # pass but no longer do. Any user-facing reference to one
    # of these is a phantom and must be replaced with the live
    # consolidated-tool form.
    PHANTOM_TOOLS = (
        "get_example_inputs",
        "get_solver_architecture",
        "browse_solver_tests",
    )

    # Files that get rendered to the LLM directly. Skip the
    # legacy module sources (tools/examples_search.py,
    # tools/developer.py) where the functions still exist
    # internally — they're just no longer registered.
    LLM_VISIBLE_FILES = [
        _REPO / "src" / "server.py",
        _REPO / "src" / "tools" / "consolidated.py",
    ]
    # Plus every catalog generator (rendered via prepare_
    # simulation and examples('search')). The audit walks
    # this glob on the fly.

    def test_no_phantom_refs(self) -> None:
        offenders = []
        files = list(self.LLM_VISIBLE_FILES)
        for backend_root in (_REPO / "src" / "backends").iterdir():
            files.extend(backend_root.rglob("*.py"))
        for f in files:
            if not f.is_file():
                continue
            try:
                text = f.read_text()
            except Exception:
                continue
            for name in self.PHANTOM_TOOLS:
                # The phantom-tool functions are STILL defined
                # in src/tools/examples_search.py and
                # src/tools/developer.py — they're just no
                # longer registered. Skip those source files.
                if "tools/examples_search.py" in str(f) or \
                   "tools/developer.py" in str(f):
                    continue
                if name in text:
                    offenders.append((str(f.relative_to(_REPO)),
                                      name))
        if offenders:
            lines = "\n".join(f"  {p}: references {n}"
                              for p, n in offenders)
            self.fail(
                f"{len(offenders)} phantom-tool references in "
                f"LLM-visible surfaces:\n{lines}\n\n"
                "These names were retired in the tool-"
                "consolidation pass; the LLM that follows the "
                "instruction will hit 'no such tool'. Replace "
                "with examples(keyword, solver, action=...) / "
                "developer(action=..., solver=...).")


if __name__ == "__main__":
    unittest.main()
