"""Regression: `reload_catalog()` must pick up BRAND-NEW physics
files added to a backend's generators/ directory after MCP-server
startup — not just edits to already-imported modules.

Context (mcp-catalog-staleness-runtime-isolation post-mortem,
2026-06-01): the MCP server imports each backend's per-physics
generator modules once at startup. When the user authors a NEW
physics — say `wave.py` or `adaptive_poisson.py` — and wires it
into `__init__.py` + `backend.py`, restarting the MCP server is
heavy. The intended workaround is `reload_catalog()`. This test
proves the workaround actually works in the new-file case (the
edit-only case is easier and is implicitly covered by the
existence of the tool).

Mechanism:
  1. fresh MCP context, snapshot skfem supported_physics
  2. write a tiny new generator file on disk
  3. inject 3 lines into skfem/generators/__init__.py +
     skfem/backend.py to register it
  4. call reload_catalog()
  5. assert: new physics name is reachable via
       `supported_physics()` AND `generate_input(...)` returns
       the template string
  6. tearDown: restore all touched files and drop the test
     module from sys.modules

If the tool starts only reloading already-imported modules
(common Python footgun — `importlib.reload` does not pick up
new submodules), this gate fires.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


_PROBE_FILENAME = "_reload_probe.py"
_PROBE_PHYSICS = "_reload_probe"
_PROBE_VARIANT = "2d"
_PROBE_KEY = f"{_PROBE_PHYSICS}_{_PROBE_VARIANT}"


def _probe_source() -> str:
    return (
        '"""Inert probe physics — created by '
        "test_reload_catalog_picks_up_new_files. "
        'Never check in.""" \n'
        "def _probe_2d(params):\n"
        '    return "# inert reload-probe template\\n"\n'
        f'GENERATORS = {{"{_PROBE_KEY}": _probe_2d}}\n'
        f'KNOWLEDGE = {{"{_PROBE_PHYSICS}": {{"description": '
        '"reload-catalog probe", '
        '"pitfalls": ['
        '"[Integration] reload-probe inert. '
        'Signal: never raises."]}}\n'
    )


class TestReloadCatalogPicksUpNewFiles(unittest.TestCase):
    def setUp(self) -> None:
        self._gens_dir = (_REPO / "src" / "backends" / "skfem"
                          / "generators")
        self._probe_path = self._gens_dir / _PROBE_FILENAME
        self._init_path = self._gens_dir / "__init__.py"
        self._backend_path = (_REPO / "src" / "backends" / "skfem"
                              / "backend.py")
        self._orig_init = self._init_path.read_text()
        self._orig_backend = self._backend_path.read_text()
        # Make sure prior runs didn't leave a probe behind.
        if self._probe_path.exists():
            self._probe_path.unlink()

    def tearDown(self) -> None:
        # Restore touched files unconditionally.
        if self._probe_path.exists():
            self._probe_path.unlink()
        self._init_path.write_text(self._orig_init)
        self._backend_path.write_text(self._orig_backend)
        # Drop the probe module from sys.modules so a re-run
        # doesn't see a stale binding.
        sys.modules.pop(
            f"backends.skfem.generators.{_PROBE_PHYSICS}", None)
        # Reload registry so the rest of the test session sees
        # the restored on-disk state.
        try:
            from core.registry import load_all_backends
            load_all_backends()
        except Exception:
            pass

    def test_reload_picks_up_new_physics(self) -> None:
        from mcp.server.fastmcp import FastMCP
        from tools.consolidated import register_consolidated_tools
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        mcp = FastMCP("reload-probe-test")
        register_consolidated_tools(mcp)
        reload_fn = mcp._tool_manager._tools["reload_catalog"].fn

        # ── Baseline.
        bk = get_backend("skfem")
        self.assertIsNotNone(bk, "skfem backend missing")
        names_pre = {p.name for p in bk.supported_physics()}
        self.assertNotIn(_PROBE_PHYSICS, names_pre,
            f"probe name {_PROBE_PHYSICS!r} already in catalog "
            "— pick a unique sentinel")

        # ── Write probe file + wire it in.
        self._probe_path.write_text(_probe_source())

        new_init = self._orig_init.replace(
            "from .adaptive_poisson import GENERATORS as _adapt_gen,",
            f"from .{_PROBE_PHYSICS} import "
            "GENERATORS as _probe_gen, KNOWLEDGE as _probe_kn\n"
            "from .adaptive_poisson import "
            "GENERATORS as _adapt_gen,"
        ).replace(
            "_nonlinear_gen, _wave_gen, _adapt_gen,",
            "_nonlinear_gen, _wave_gen, _adapt_gen, _probe_gen,"
        ).replace(
            "_nonlinear_kn, _wave_kn, _adapt_kn,",
            "_nonlinear_kn, _wave_kn, _adapt_kn, _probe_kn,"
        )
        self.assertNotEqual(new_init, self._orig_init,
            "wiring substitution into __init__.py failed — anchors "
            "may have drifted. Update this test if the skfem "
            "generators/__init__.py shape changed.")
        self._init_path.write_text(new_init)

        new_backend = self._orig_backend.replace(
            'name="adaptive_poisson",',
            f'name="{_PROBE_PHYSICS}",\n'
            '                description="reload-probe inert",\n'
            '                spatial_dims=[2],\n'
            '                element_types=["P1-tri"],\n'
            '                template_variants=["2d"],\n'
            '            ),\n'
            '            PhysicsCapability(\n'
            '                name="adaptive_poisson",',
        )
        self.assertNotEqual(new_backend, self._orig_backend,
            "wiring substitution into backend.py failed — anchors "
            "may have drifted. Update this test if the skfem "
            "backend.py PhysicsCapability list shape changed.")
        self._backend_path.write_text(new_backend)

        # ── Hot-reload.
        result = reload_fn()
        self.assertIn("reloaded", result,
            f"reload_catalog did not run cleanly: {result[:300]}")
        self.assertNotIn("FAILED", result,
            f"reload_catalog reported failure: {result[:300]}")

        # ── Assert: new physics is visible after hot-reload.
        bk = get_backend("skfem")
        names_post = {p.name for p in bk.supported_physics()}
        self.assertIn(_PROBE_PHYSICS, names_post,
            f"reload_catalog did NOT pick up the brand-new "
            f"physics {_PROBE_PHYSICS!r}. supported_physics still "
            f"shows {len(names_post)} entries: "
            f"{sorted(names_post)}. The hot-reload tool only "
            "picks up edits to already-imported modules — it "
            "needs to also discover and import NEW files. "
            "Without this fix, the user has to fully restart "
            "the MCP server every time a new physics ships.")

        # ── Assert: generate_input for the new physics works.
        src = bk.generate_input(_PROBE_PHYSICS, _PROBE_VARIANT, {})
        self.assertIn("inert reload-probe template", src,
            f"generate_input for the new physics returned "
            f"unexpected content: {src[:200]!r}")


if __name__ == "__main__":
    unittest.main()
