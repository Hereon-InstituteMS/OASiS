"""Regression: the consolidated `transfer_field` MCP tool must honour
its advertised `target_format` parameter.

Caught 2026-06-02: the consolidated wrapper at
src/tools/consolidated.py:1375 advertised
``target_format='json'|'fenics'|'4c_neumann'`` and ``output_path`` in its
docstring/signature, but the implementation called
``extract_interface_from_vtu`` and threw both parameters away — every
invocation returned the same plain-markdown summary regardless of what
the LLM asked for, and no formatted output file was ever written.

Same drift class as the visualize('validate') case from the 2026-06-01
audit: the docstring promised an action / format the dispatcher did
not actually carry out.

This test exercises all three formats end-to-end on a synthetic VTU
and asserts:
  1. the correct file extension is written for each target_format
  2. the file is non-empty
  3. the unknown-format branch returns a usage hint string
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _make_synthetic_vtu(out: Path) -> None:
    """Write a small unstructured 3x3x3 grid with a 'temperature'
    point field to ``out``. PyVista is required; the test skips
    if it is unavailable so the suite still runs on minimal envs.
    """
    import pyvista as pv

    grid = pv.ImageData(dimensions=(3, 3, 3))
    ugrid = grid.cast_to_unstructured_grid()
    temp_vals = np.linspace(0, 100, ugrid.n_points)
    ugrid.point_data["temperature"] = temp_vals
    ugrid.save(str(out))


def _call_transfer_field(**kwargs) -> str:
    """Invoke the consolidated `transfer_field` tool directly by
    re-registering the MCP tools and pulling the registered
    function out of the FastMCP tool registry.

    Going through FastMCP rather than re-implementing the wrapper
    means the test exercises the SAME code path the live MCP
    server runs.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise unittest.SkipTest(f"FastMCP not installed: {exc}")
    from tools.consolidated import register_consolidated_tools

    mcp = FastMCP("test")
    register_consolidated_tools(mcp)
    # FastMCP stores tool functions on `mcp._tool_manager._tools`
    # (private but stable across the versions we pin). Each entry
    # is a Tool object whose .fn is the registered coroutine.
    tools = mcp._tool_manager._tools  # type: ignore[attr-defined]
    handle = tools["transfer_field"]
    coro = handle.fn(**kwargs)
    return asyncio.get_event_loop().run_until_complete(coro)


class TestTransferFieldTargetFormat(unittest.TestCase):

    def setUp(self) -> None:
        try:
            import pyvista  # noqa: F401
        except ImportError:
            self.skipTest("PyVista not installed")
        # Allow test runner to share an event loop.
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        import tempfile
        self._tmpdir = tempfile.mkdtemp(prefix="transfer_field_test_")
        self._vtu = Path(self._tmpdir) / "src.vtu"
        _make_synthetic_vtu(self._vtu)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _run(self, **kw):
        return _call_transfer_field(
            source_vtu=str(self._vtu),
            field_name="temperature",
            interface_coord=1.0,
            interface_axis=0,
            **kw,
        )

    def test_json_format_writes_json(self) -> None:
        out = Path(self._tmpdir) / "iface.json"
        result = self._run(target_format="json", output_path=str(out))
        self.assertIn("Field Transfer", result)
        self.assertTrue(out.exists(),
                        f"json output not written: {out}")
        self.assertGreater(out.stat().st_size, 0)

    def test_fenics_format_writes_py(self) -> None:
        out = Path(self._tmpdir) / "iface.json"  # extension rewritten
        result = self._run(target_format="fenics", output_path=str(out))
        py_out = out.with_suffix(".py")
        self.assertIn("Field Transfer", result)
        self.assertTrue(py_out.exists(),
                        f"fenics output not written: {py_out}")
        self.assertGreater(py_out.stat().st_size, 0)
        # The generated code must mention DirichletBC.
        text = py_out.read_text()
        self.assertIn("Dirichlet", text + " ")

    def test_4c_neumann_format_writes_yaml(self) -> None:
        out = Path(self._tmpdir) / "iface.json"  # extension rewritten
        result = self._run(target_format="4c_neumann",
                           output_path=str(out))
        yaml_out = out.with_suffix(".yaml")
        self.assertIn("Field Transfer", result)
        self.assertTrue(yaml_out.exists(),
                        f"4c_neumann output not written: {yaml_out}")
        self.assertGreater(yaml_out.stat().st_size, 0)

    def test_unknown_format_returns_usage_hint(self) -> None:
        result = self._run(target_format="bogus")
        self.assertIn("Unknown format", result)
        self.assertIn("json", result)
        self.assertIn("fenics", result)
        self.assertIn("4c_neumann", result)

    def test_default_output_path(self) -> None:
        # Empty output_path -> auto-generated next to source VTU.
        result = self._run(target_format="json", output_path="")
        self.assertIn("Field Transfer", result)
        auto = self._vtu.parent / "interface_temperature.json"
        self.assertTrue(auto.exists(),
                        f"auto output_path not honoured: {auto}")


if __name__ == "__main__":
    unittest.main()
