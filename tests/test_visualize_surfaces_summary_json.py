"""Regression: visualize('summary') must surface results_summary.json.

Audit 2026-06-01: Layer F catalog templates (fenics / ngsolve /
skfem / kratos) write a per-run summary at results_summary.json
— max field values, dof counts, convergence metrics. The
visualize MCP tool's 'summary' action used to only read
VTU/VTK/VTP/XDMF/BP files, so a directory containing only
results_summary.json returned '[]' even though the template
produced exactly what the LLM wanted.

Fix: visualize('summary') now rglobs for results_summary.json
and prepends each file's contents to the response.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


class _StubMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco

    def resource(self, *args, **kwargs):
        def deco(fn):
            return fn
        return deco

    def prompt(self, *args, **kwargs):
        def deco(fn):
            return fn
        return deco


def _visualize_fn():
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _StubMCP()
    register_consolidated_tools(mcp)
    return mcp.tools["visualize"]


class TestVisualizeSummarySurfacesJSON(unittest.TestCase):
    """visualize('summary') must read results_summary.json
    in addition to VTU files."""

    def test_summary_json_alone(self) -> None:
        """A work_dir with ONLY results_summary.json must still
        produce a non-[] response containing the JSON content."""
        vis = _visualize_fn()
        with tempfile.TemporaryDirectory() as td:
            payload = {"max_value": 0.0625, "n_dofs": 4225,
                       "physics": "poisson_2d"}
            (Path(td) / "results_summary.json").write_text(
                json.dumps(payload))
            r = asyncio.run(vis(work_dir=td, action="summary"))
            self.assertIsInstance(r, str)
            self.assertNotEqual(
                r.strip(), "[]",
                "visualize('summary') returned '[]' even though "
                "results_summary.json was present. The summary "
                "reader must rglob for results_summary.json — "
                "every Layer F template writes one.")
            self.assertIn("max_value", r)
            self.assertIn("0.0625", r)
            self.assertIn("results_summary_json", r)

    def test_truly_empty_dir(self) -> None:
        """A work_dir with nothing readable still returns '[]'."""
        vis = _visualize_fn()
        with tempfile.TemporaryDirectory() as td:
            r = asyncio.run(vis(work_dir=td, action="summary"))
            self.assertEqual(
                r.strip(), "[]",
                "Empty directory must produce the empty-summary "
                "signal '[]'.")


if __name__ == "__main__":
    unittest.main()
