"""Regression: generate_mesh docstring lists only built-in
geometries that the dispatch table actually handles.

Task #157 (2026-06-01) removed a phantom "or custom" claim from
the generate_mesh tool docstring — generate_mesh does NOT have
a custom-passthrough path; passing an unknown geometry returns
an "Unknown geometry" error message. This gate prevents the
phantom claim from being re-introduced and prevents the inverse
problem (a new geometry added to the dispatch table without
being documented).

The gate parses the docstring's `geometry:` bullet list and
compares against the keys of the literal `generators` dict in
the same function body.
"""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


class TestGenerateMeshDocstring(unittest.TestCase):
    def test_docstring_geometries_match_dispatch(self) -> None:
        src = (_REPO / "src" / "tools" / "consolidated.py").read_text()
        tree = ast.parse(src)
        # Find the generate_mesh FunctionDef inside any nested
        # scope (it's defined as @mcp.tool() inside
        # register_workflow_tools).
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "generate_mesh":
                func = node
                break
        else:
            self.fail("generate_mesh not found in consolidated.py")

        # Extract the docstring's geometry-bullet list. The
        # canonical format is:
        #   - "l_domain"   — ...
        #   - "name"       — ...
        doc = ast.get_docstring(func) or ""
        documented = set(
            re.findall(r'-\s+"([^"]+)"', doc))

        # Extract dispatch keys: literal generators = { "X": ..., }.
        dispatch_keys: set[str] = set()
        for inner in ast.walk(func):
            if isinstance(inner, ast.Assign):
                for tgt in inner.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "generators":
                        val = inner.value
                        if isinstance(val, ast.Dict):
                            for key in val.keys:
                                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                                    dispatch_keys.add(key.value)
                        break

        self.assertEqual(
            documented, dispatch_keys,
            "generate_mesh docstring geometry list drifted from "
            "the literal `generators` dispatch dict.\n"
            f"  documented: {sorted(documented)}\n"
            f"  dispatched: {sorted(dispatch_keys)}\n"
            f"  diff:       {sorted(documented ^ dispatch_keys)}\n"
            "Phantom-geometry advertisement (in docstring but no "
            "dispatch) or undocumented-geometry support (dispatch "
            "but no docstring entry) both break the LLM's plan: "
            "the LLM either tries to use a non-existent name or "
            "misses a real one. Update docstring + dispatch in "
            "lock-step.")

    def test_no_phantom_custom_geometry(self) -> None:
        """Defense in depth: the documented set must not include
        'custom' (no passthrough) and must include the actual 3
        built-ins (l_domain, plate_with_hole, channel_cylinder)
        so a regression that removes the dispatch but leaves
        the docstring also fails this test."""
        src = (_REPO / "src" / "tools" / "consolidated.py").read_text()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "generate_mesh":
                func = node
                break
        else:
            self.fail("generate_mesh not found")
        doc = ast.get_docstring(func) or ""
        documented = set(re.findall(r'-\s+"([^"]+)"', doc))
        self.assertNotIn("custom", documented,
                         "generate_mesh docstring re-advertises 'custom' "
                         "geometry that has no dispatch — task #157 "
                         "regression.")
        self.assertGreaterEqual(
            documented & {"l_domain", "plate_with_hole",
                          "channel_cylinder"},
            {"l_domain", "plate_with_hole", "channel_cylinder"},
            "generate_mesh docstring is missing one of the canonical "
            "built-in geometries.")


if __name__ == "__main__":
    unittest.main()
