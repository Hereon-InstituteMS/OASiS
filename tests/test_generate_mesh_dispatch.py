"""Regression: generate_mesh tool dispatch must not crash on import.

Audit 2026-06-01: every invocation of the MCP generate_mesh tool
failed with `ImportError: cannot import name
'_generate_channel_cylinder_2d'` because the import line in
consolidated.py was misspelled — the actual function in
tools/mesh_generation.py is `_generate_channel_with_cylinder_2d`.
The ImportError short-circuited the dispatch dict for ALL three
advertised geometries, not just channel_cylinder.

This test pins the dispatch shape: the three advertised
geometries must be reachable (even if their underlying call
fails on a missing optional runtime dep like gmsh), and the
unknown-geometry path must surface the available list with the
correct names.
"""
from __future__ import annotations

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


def _generate_mesh_fn():
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _StubMCP()
    register_consolidated_tools(mcp)
    return mcp.tools["generate_mesh"]


class TestGenerateMeshDispatch(unittest.TestCase):
    """The three advertised geometries (l_domain, plate_with_hole,
    channel_cylinder) must each pass dispatch — i.e. NOT fail with
    an ImportError on a name that doesn't exist in
    tools.mesh_generation."""

    def test_known_geometries_reach_dispatch(self) -> None:
        gm = _generate_mesh_fn()
        with tempfile.TemporaryDirectory() as td:
            for geom in ("l_domain", "plate_with_hole",
                         "channel_cylinder"):
                with self.subTest(geometry=geom):
                    r = gm(geometry=geom, mesh_size=0.5,
                           output_dir=td)
                    self.assertNotIn(
                        "cannot import name", r,
                        f"generate_mesh({geom!r}) failed at "
                        "ImportError — the dispatch shape is "
                        "broken. tools.mesh_generation function "
                        "names must match the import line in "
                        "tools/consolidated.py exactly.")
                    self.assertNotIn(
                        "Unknown geometry", r,
                        f"generate_mesh({geom!r}) was rejected "
                        "as unknown — the dispatch dict is "
                        "missing this key.")

    def test_unknown_geometry_lists_available(self) -> None:
        gm = _generate_mesh_fn()
        with tempfile.TemporaryDirectory() as td:
            r = gm(geometry="bogus_xyz", mesh_size=0.1,
                   output_dir=td)
            self.assertIn("Unknown geometry", r)
            # The available list must contain all three advertised
            # geometries (the docstring promise).
            for name in ("l_domain", "plate_with_hole",
                         "channel_cylinder"):
                self.assertIn(name, r,
                              f"available list missing {name!r}")


class TestGenerateMeshEndToEnd(unittest.TestCase):
    """When gmsh is installed, generate_mesh must actually
    produce a non-empty .msh file for each advertised geometry."""

    def test_actual_mesh_files_written(self) -> None:
        try:
            import gmsh  # noqa: F401
        except ImportError:
            self.skipTest("gmsh not installed in this env — "
                          "the dispatch-only test still covers "
                          "the regression. Install with "
                          "`pip install gmsh` to exercise the "
                          "end-to-end path.")
        gm = _generate_mesh_fn()
        with tempfile.TemporaryDirectory() as td:
            for geom in ("l_domain", "plate_with_hole",
                         "channel_cylinder"):
                with self.subTest(geometry=geom):
                    r = gm(geometry=geom, mesh_size=0.3,
                           output_dir=td)
                    self.assertIn(
                        "mesh:", r,
                        f"generate_mesh({geom!r}) did not "
                        "return a 'mesh: ...' confirmation: "
                        f"{r[:200]!r}")
                    msh = Path(td) / f"{geom}.msh"
                    self.assertTrue(
                        msh.is_file(),
                        f"No .msh file at {msh}")
                    self.assertGreater(
                        msh.stat().st_size, 1000,
                        f"{msh} is suspiciously small "
                        f"({msh.stat().st_size} bytes) — "
                        "gmsh may have written an empty file.")


if __name__ == "__main__":
    unittest.main()
