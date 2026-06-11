"""Tests for the guided backend-setup module + setup_backend MCP tool.

Created 2026-06-12 (task #227). Covers:
  1. Route catalog invariants — every backend has >= 1 route, every
     route covers linux, darwin entries exist as extension points
     (the user's Mac instance fills them in), pip/conda routes carry
     executable commands, source/binary routes don't pretend to.
  2. plan_setup() — structure, OS resolution, prefer= override,
     unknown-backend error.
  3. detect_backend()/setup_status() — runs against the live machine;
     asserts only stable invariants (skfem available in this venv,
     legacy config fallback resolves fourc's source tree).
  4. The legacy sources.json fallback (pre-rebrand
     ~/.config/open-fem-agent/ vs new ~/.config/oasis/) — the
     rebrand regression found 2026-06-12.
  5. MCP tool wiring — setup_backend registered, status/usage paths.

Deliberately NOT covered here: execute_setup's real pip/conda
installs (mutates the environment; exercised manually + via the
fresh-terminal prompts in TESTING_PROMPTS.md).
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from core.backend_setup import (  # noqa: E402
    SETUP_ROUTES, plan_setup, detect_backend, setup_status,
    render_status_markdown, _current_os,
)


class TestRouteCatalogInvariants(unittest.TestCase):

    def test_every_backend_has_at_least_one_route(self) -> None:
        self.assertEqual(
            sorted(SETUP_ROUTES),
            sorted(["skfem", "ngsolve", "kratos", "dune", "fenics",
                    "dealii", "fourc", "febio"]),
            "Route catalog must cover all 8 backends.")
        for be, routes in SETUP_ROUTES.items():
            self.assertGreater(len(routes), 0, be)

    def test_every_route_covers_linux(self) -> None:
        for be, routes in SETUP_ROUTES.items():
            for r in routes:
                with self.subTest(backend=be, kind=r["kind"]):
                    self.assertIn("linux", r["os_support"],
                                  f"{be}/{r['kind']} has no linux entry")

    def test_darwin_extension_points_exist(self) -> None:
        """Every backend must have at least one route with a darwin
        entry — that's the landing zone for the user's Mac compile
        notes. An absent darwin key would force a schema change later."""
        for be, routes in SETUP_ROUTES.items():
            with self.subTest(backend=be):
                self.assertTrue(
                    any("darwin" in r["os_support"] for r in routes),
                    f"{be} has no darwin extension point")

    def test_executable_routes_have_commands(self) -> None:
        """pip/conda routes execute inline — they need argv lists.
        source routes delegate to source_orchestrator and binary
        routes are manual — their commands list stays empty."""
        for be, routes in SETUP_ROUTES.items():
            for r in routes:
                with self.subTest(backend=be, kind=r["kind"]):
                    if r["kind"] in ("pip", "conda"):
                        self.assertTrue(r["commands"],
                                        f"{be}/{r['kind']} inline route "
                                        f"without commands")
                    else:
                        self.assertEqual(r["commands"], [],
                                         f"{be}/{r['kind']} should not "
                                         f"carry inline commands")

    def test_fourc_darwin_note_references_user_thread(self) -> None:
        """The 4C darwin entry is the specific extension point the
        user called out (his 4C-discussion-thread Mac settings)."""
        fourc_src = [r for r in SETUP_ROUTES["fourc"]
                     if r["kind"] == "source"][0]
        notes = " ".join(fourc_src["os_support"]["darwin"]["notes"])
        self.assertIn("EXTENSION POINT", notes)


class TestPlanSetup(unittest.TestCase):

    def test_unknown_backend_clean_error(self) -> None:
        p = plan_setup("ghost")
        self.assertIn("error", p)
        self.assertIn("Known", p["error"])

    def test_plan_structure(self) -> None:
        p = plan_setup("skfem")
        self.assertEqual(p["backend"], "skfem")
        self.assertEqual(p["os"], _current_os())
        self.assertIn("route", p)
        for key in ("kind", "description", "commands", "system_deps",
                    "notes", "verified_on_this_os", "typical_minutes"):
            self.assertIn(key, p["route"])

    def test_prefer_override(self) -> None:
        p = plan_setup("ngsolve", prefer="source")
        self.assertEqual(p["route"]["kind"], "source")
        # and without prefer the faster pip route wins
        p2 = plan_setup("ngsolve")
        self.assertEqual(p2["route"]["kind"], "pip")

    def test_fourc_on_linux_recommends_source(self) -> None:
        if _current_os() != "linux":
            self.skipTest("linux-only assertion")
        p = plan_setup("fourc")
        self.assertEqual(p["route"]["kind"], "source",
                         "4C has no binary distribution — source is "
                         "the only route.")
        self.assertTrue(p["route"]["verified_on_this_os"])


class TestDetectAndStatus(unittest.TestCase):

    def test_skfem_detected_available(self) -> None:
        """skfem is pip-installed in this repo's venv — if this fails
        the detection plumbing (registry check_availability) broke."""
        d = detect_backend("skfem")
        self.assertTrue(d["available"],
                        f"skfem should be available; details: "
                        f"{d['details']}")

    def test_legacy_config_fallback_resolves_fourc(self) -> None:
        """The rebrand moved _GLOBAL_CONFIG_PATH to ~/.config/oasis/
        but this machine's real config lives at the pre-rebrand
        ~/.config/open-fem-agent/. The fallback must surface fourc's
        source tree. (Regression found + fixed 2026-06-12.)"""
        legacy = (Path.home() / ".config" / "open-fem-agent"
                  / "sources.json")
        new = Path.home() / ".config" / "oasis" / "sources.json"
        if not legacy.exists() and not new.exists():
            self.skipTest("no sources.json on this machine")
        d = detect_backend("fourc")
        self.assertIsNotNone(
            d["source_tree"],
            "fourc source tree not resolved — the legacy-config "
            "fallback in source_config.load() regressed.")

    def test_status_table_renders(self) -> None:
        md = render_status_markdown()
        self.assertIn("| backend |", md)
        self.assertEqual(md.count("\n") + 1, 2 + len(SETUP_ROUTES),
                         "one row per backend plus 2 header lines")

    def test_status_rows_cover_all_backends(self) -> None:
        rows = setup_status()
        self.assertEqual(sorted(r["backend"] for r in rows),
                         sorted(SETUP_ROUTES))


class TestMcpToolWiring(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from mcp.server.fastmcp import FastMCP  # type: ignore
        from tools.consolidated import register_consolidated_tools
        from core.registry import load_all_backends
        load_all_backends()
        m = FastMCP("test")
        register_consolidated_tools(m)
        # staticmethod: otherwise self.tool(action=...) passes the
        # test instance as the first positional arg (action).
        cls.tool = staticmethod(m._tool_manager._tools["setup_backend"].fn)

    def test_status_action(self) -> None:
        out = self.tool(action="status")
        self.assertIn("| backend |", out)

    def test_plan_action_returns_json(self) -> None:
        out = self.tool(action="plan", solver="skfem")
        parsed = json.loads(out)
        self.assertEqual(parsed["backend"], "skfem")

    def test_missing_solver_usage_hint(self) -> None:
        out = self.tool(action="plan")
        self.assertIn("Usage", out)

    def test_unknown_action_usage_hint(self) -> None:
        out = self.tool(action="frobnicate", solver="skfem")
        self.assertIn("unknown action", out)


if __name__ == "__main__":
    unittest.main()
