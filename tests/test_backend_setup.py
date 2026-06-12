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
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

import core.backend_setup as backend_setup  # noqa: E402
from core.backend_setup import (  # noqa: E402
    SETUP_ROUTES, plan_setup, detect_backend, setup_status,
    render_status_markdown, _current_os, _verify_and_persist,
    execute_setup,
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


class TestSetupSessionRegressions(unittest.TestCase):
    """Regressions found in the 2026-06-12 fresh-Ubuntu test session
    (see the session FINDINGS.md): false 'verified' for a broken dune,
    false 'installed_unverified' + persistence for an absent febio,
    silently ignored route= override, wrong-route command selection,
    and the dune conda route claiming verification for a conda-forge
    package that does not exist."""

    def test_prefer_unavailable_route_is_explicit_error(self) -> None:
        """Finding 2: plan(dune, route='source') used to silently
        return the conda route. dune has only a conda route — asking
        for anything else must error, not substitute."""
        p = plan_setup("dune", prefer="source")
        self.assertIn("error", p)
        self.assertIn("source", p["error"])
        self.assertIn("conda", str(p["error"]))

    def test_dune_conda_route_not_marked_verified_on_linux(self) -> None:
        """Finding 1: dune-fem is not on conda-forge (confirmed
        2026-06-12 via api.anaconda.org) — the route must not claim
        verified_on_this_os until that changes."""
        dune_conda = [r for r in SETUP_ROUTES["dune"]
                      if r["kind"] == "conda"][0]
        self.assertFalse(dune_conda["os_support"]["linux"]["verified"])
        notes = " ".join(dune_conda["os_support"]["linux"]["notes"])
        self.assertIn("conda-forge", notes)

    def test_smoke_dune_honours_ok_false(self) -> None:
        """Finding 3: the dune smoke script exits 0 even on
        ImportError; the verdict is the JSON 'ok' flag. passed=True
        for {"ok": false} produced a false 'verified' status."""
        import core.smoke_tests as smoke_tests
        with mock.patch.object(
                smoke_tests, "_run_script",
                return_value=(True,
                              '{"ok": false, "error": "libibverbs boom"}',
                              "")):
            r = smoke_tests.smoke_dune()
        self.assertFalse(r.passed)
        self.assertIn("libibverbs boom", r.error or "")

    def test_smoke_dune_passes_on_ok_true(self) -> None:
        import core.smoke_tests as smoke_tests
        with mock.patch.object(smoke_tests, "_run_script",
                               return_value=(True, '{"ok": true}', "")):
            r = smoke_tests.smoke_dune()
        self.assertTrue(r.passed)

    def test_verify_no_smoke_test_unavailable_is_not_installed(self) -> None:
        """Finding 5: febio (no smoke test, not installed) used to be
        reported 'installed_unverified' and got a path persisted into
        sources.json. Unavailable + no smoke test must be
        'not_installed' with nothing persisted."""
        with mock.patch.object(
                backend_setup, "detect_backend",
                return_value={"backend": "febio", "available": False,
                              "source_tree": None, "build": None,
                              "details": "NOT_INSTALLED: no binary"}):
            out = _verify_and_persist("febio")
        self.assertEqual(out["status"], "not_installed")
        self.assertNotIn("persisted", out)

    def test_verify_no_smoke_test_available_persists_binary_env(self) -> None:
        """Finding 6: 'setup_backend persists the binary path' was
        untrue — FEBIO_BINARY was never read. When the backend is
        available and <BACKEND>_BINARY points at a real file, verify
        must persist it."""
        with tempfile.TemporaryDirectory() as td:
            fake_bin = Path(td) / "febio4"
            fake_bin.write_text("#!/bin/sh\n")
            cfg = Path(td) / "sources.json"
            with mock.patch.object(
                    backend_setup, "detect_backend",
                    return_value={"backend": "febio", "available": True,
                                  "source_tree": None, "build": None,
                                  "details": "AVAILABLE: binary"}), \
                 mock.patch.object(backend_setup, "_GLOBAL_CONFIG_PATH",
                                   cfg), \
                 mock.patch.dict("os.environ",
                                 {"FEBIO_BINARY": str(fake_bin)}):
                out = _verify_and_persist("febio")
            self.assertEqual(out["status"], "installed_unverified")
            self.assertIn("persisted", out)
            saved = json.loads(cfg.read_text())
            self.assertEqual(saved["backends"]["febio"]["binary"],
                             str(fake_bin))

    def test_first_persist_migrates_legacy_config(self) -> None:
        """Finding 9: load() reads the new global sources.json OR the
        legacy one, never both. _persist_backend_paths used to create
        the new file with only the persisted backend, silently
        shadowing every backend configured in the legacy file (this
        actually lost fourc/dealii on the test machine). The first
        write to the new path must migrate the legacy content."""
        with tempfile.TemporaryDirectory() as td:
            legacy = Path(td) / "legacy" / "sources.json"
            legacy.parent.mkdir()
            legacy.write_text(json.dumps({
                "scan_paths": ["~/Schreibtisch"],
                "backends": {
                    "fourc": {"source": "/src/4C", "build": "/src/4C/b"},
                    "dealii": {"source": "/src/dealii"},
                }}))
            new = Path(td) / "new" / "sources.json"
            with mock.patch.object(backend_setup, "_GLOBAL_CONFIG_PATH",
                                   new), \
                 mock.patch.object(backend_setup,
                                   "_LEGACY_GLOBAL_CONFIG_PATH", legacy):
                backend_setup._persist_backend_paths("dune",
                                                     source="/src/dune")
            saved = json.loads(new.read_text())
            self.assertEqual(saved["backends"]["dune"]["source"],
                             "/src/dune")
            self.assertEqual(saved["backends"]["fourc"]["build"],
                             "/src/4C/b", "legacy fourc entry lost")
            self.assertIn("dealii", saved["backends"])
            self.assertEqual(saved["scan_paths"], ["~/Schreibtisch"])

    def test_execute_setup_runs_chosen_route_commands(self) -> None:
        """Finding 7: with route_kind=None the commands came from
        routes[0] even when the plan chose a different route for this
        OS. The executed command must belong to the chosen kind."""
        osk = _current_os()
        fake_routes = [
            {"kind": "pip", "description": "wrong-OS pip route",
             "commands": [["pip", "install", "WRONG"]],
             "typical_minutes": 1,
             "os_support": {"someotheros": {"verified": True,
                                            "system_deps": [],
                                            "notes": []}}},
            {"kind": "conda", "description": "right-OS conda route",
             "commands": [["conda", "create", "-n", "RIGHT"]],
             "typical_minutes": 1,
             "os_support": {osk: {"verified": True, "system_deps": [],
                                  "notes": []}}},
        ]
        ran: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            ran.append(cmd)
            return mock.Mock(returncode=1, stderr="stop here", stdout="")

        with mock.patch.dict(SETUP_ROUTES, {"fakebe": fake_routes}), \
             mock.patch.object(
                 backend_setup, "detect_backend",
                 return_value={"backend": "fakebe", "available": False,
                               "source_tree": None, "build": None,
                               "details": ""}), \
             mock.patch.object(backend_setup.subprocess, "run", fake_run):
            execute_setup("fakebe")
        self.assertEqual(ran, [["conda", "create", "-n", "RIGHT"]],
                         "must run the OS-chosen conda route, not "
                         "routes[0]'s pip command")


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
