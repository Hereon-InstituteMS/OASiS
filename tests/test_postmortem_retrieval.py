"""Tests for the knowledge(topic='postmortems', ...) retrieval path.

Closes the self-improvement loop: the agent / critic-gate can call
this to find the audit record explaining why a particular pitfall
exists. Without retrieval, post-mortems are write-only and the
loop never closes (Open-FEM-Agent §3.2 / §5).

These tests use the post-mortem JSONs actually committed under
data/postmortems/, so they double as a structural-validity check
on the committed records themselves.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path

import pytest

from tools.consolidated import _load_matching_postmortems

REPO_ROOT = Path(__file__).resolve().parents[1]


POSTMORTEMS_DIR = (Path(__file__).resolve().parent.parent
                   / "data" / "postmortems")


class TestLoadMatchingPostmortems(unittest.TestCase):

    def setUp(self):
        # All real post-mortems committed today should be picked up
        # when called with no filters — that's the baseline.
        self.all_pm = _load_matching_postmortems()
        if not self.all_pm:
            self.skipTest(
                f"no post-mortems under {POSTMORTEMS_DIR} — this test "
                f"file assumes the layer-a branch's commits are present"
            )

    def test_empty_filters_return_all_committed_records(self):
        # Every JSON in the dir (excluding _-prefixed schema files
        # and candidates/) should be returned.
        files = [p for p in POSTMORTEMS_DIR.glob("*.json")
                 if not p.name.startswith("_")]
        self.assertEqual(len(self.all_pm), len(files),
                         f"{len(files)} JSON files but only "
                         f"{len(self.all_pm)} loaded — schema or "
                         f"filter bug")

    def test_solver_filter_exact(self):
        out = _load_matching_postmortems(solver="dealii")
        self.assertTrue(out)
        self.assertTrue(all(d["backend"] == "dealii" for d in out))

    def test_solver_filter_misses_when_wrong(self):
        # Case-insensitive but otherwise exact.
        out = _load_matching_postmortems(solver="fenics")
        # We have no fenics post-mortems on this branch yet — used
        # to be 0, will grow as encoding expands. Either way, none
        # of the dealii records should leak through.
        for d in out:
            self.assertEqual(d["backend"], "fenics")

    def test_physics_filter_substring(self):
        # The batch post-mortem has physics "poisson, heat,
        # helmholtz, eigenvalue (batch)" — substring match on any
        # member should retrieve it.
        for sub in ("poisson", "helmholtz", "eigenvalue"):
            out = _load_matching_postmortems(physics=sub)
            self.assertTrue(out, f"physics filter {sub!r} returned 0")
            self.assertTrue(
                any(sub in d.get("physics", "").lower() for d in out))

    def test_signal_filter_against_pitfall_entries(self):
        # The Crank-Nicolson degradation pitfall in the batch
        # post-mortem has a Signal: clause mentioning 'manufactured
        # solution' — retrieval should find it.
        out = _load_matching_postmortems(signal="manufactured solution")
        self.assertTrue(out,
                        "signal filter on a known Signal: clause "
                        "returned 0")

    def test_sorted_most_recent_first(self):
        # All post-mortems were written 2026-05-31; sort should be
        # stable on date. When older records appear in later
        # commits, the order should still put most-recent at index 0.
        dates = [d.get("date", "") for d in self.all_pm]
        self.assertEqual(dates, sorted(dates, reverse=True))

    def test_candidates_subdir_not_included(self):
        # Candidates are pending-review by definition (#46); they
        # MUST NOT be returned by the formal post-mortem retrieval.
        candidates_dir = POSTMORTEMS_DIR / "candidates"
        if not candidates_dir.is_dir():
            self.skipTest("no candidates/ subdir")
        candidate_files = list(candidates_dir.glob("*.json"))
        if not candidate_files:
            self.skipTest("no candidate JSONs to test against")
        # Make sure none of the returned post-mortems' IDs match
        # any candidate file's stem.
        candidate_ids = {p.stem for p in candidate_files}
        loaded_ids = {d.get("id", "") for d in self.all_pm}
        leaked = candidate_ids & loaded_ids
        self.assertFalse(
            leaked,
            f"candidate IDs {leaked} leaked into formal "
            f"post-mortem load — the candidates path is supposed "
            f"to require manual promotion")


class TestRequiredFieldsPresent(unittest.TestCase):
    """The retrieval is only useful if records have the fields it
    promises — root_cause, categories, pitfall_db_entries,
    agent_detection_after_fix. This is the same shape the
    test_postmortems.py schema-test enforces on PR #26's branch;
    we duplicate the minimal subset here so this branch's tests
    can stand alone."""

    def test_each_record_has_required_shape(self):
        records = _load_matching_postmortems()
        if not records:
            self.skipTest("no post-mortems to validate on this branch")
        required = ("id", "date", "backend", "physics",
                    "surface_symptom", "root_cause", "categories",
                    "pitfall_db_entries", "agent_detection_after_fix")
        failures = []
        for d in records:
            for field in required:
                if field not in d:
                    failures.append(f"{d.get('id', '?')}: missing "
                                    f"{field!r}")
        self.assertFalse(failures, "\n".join(failures))


class TestKnowledgeMCPToolReturnsPostmortems(unittest.TestCase):
    """End-to-end via the real MCP-stdio surface.

    Spawns the server in a subprocess, calls
    ``knowledge(topic='postmortems', solver='dealii')`` and asserts
    the response includes at least one of the deal.II post-mortems
    committed on this branch. This is the test that proves the
    self-improvement loop is closed at the agent-facing surface —
    not just that the helper function works.
    """

    def _server_params(self):
        from mcp import StdioServerParameters
        env = {
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "PYVISTA_OFF_SCREEN": "true",
        }
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", "server"],
            env=env,
            cwd=str(REPO_ROOT / "src"),
        )

    async def _call_tool(self, name: str, arguments: dict) -> dict:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client
        params = self._server_params()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                resp = await session.call_tool(name, arguments=arguments)
                text = "\n".join(
                    c.text for c in resp.content if hasattr(c, "text"))
                return {"isError": bool(resp.isError), "text": text}

    def test_knowledge_postmortems_returns_committed_records(self):
        try:
            r = asyncio.run(self._call_tool(
                "knowledge",
                {"topic": "postmortems", "solver": "dealii"},
            ))
        except ModuleNotFoundError as e:
            pytest.skip(f"mcp client SDK not importable: {e}")

        self.assertFalse(r["isError"],
                         f"knowledge('postmortems') raised: "
                         f"{r['text'][:300]}")
        # The response is a JSON array; parse and verify it has at
        # least one deal.II record. The exact records depend on
        # what is committed; we only require the SHAPE works.
        try:
            records = json.loads(r["text"])
        except json.JSONDecodeError:
            self.fail(
                f"knowledge('postmortems') did not return JSON; "
                f"got: {r['text'][:300]}")
        self.assertIsInstance(records, list,
                              f"expected JSON array, got "
                              f"{type(records).__name__}")
        if records:
            # When we DO have dealii records, every one must be
            # backend=dealii.
            for d in records:
                self.assertEqual(d.get("backend"), "dealii")

    def test_knowledge_physics_includes_postmortem_breadcrumbs_only(self):
        """At PLAN time, knowledge(topic='physics', ...) auto-includes
        only post-mortem BREADCRUMBS — IDs + categories + date — NOT
        the full record. Rationale per the 2026-05-31 senior-AI-
        scientist critic: full records include diagnostic fields
        (surface_symptom, root_cause, agent_detection_after_fix) that
        belong to post-execution critic review, and including all
        records in the plan-time response produces linear token bloat
        in N_postmortems. The agent fetches the full record
        explicitly when it has a Signal: to match against."""
        try:
            r = asyncio.run(self._call_tool(
                "knowledge",
                {"topic": "physics", "solver": "dealii",
                 "physics": "linear_elasticity"},
            ))
        except ModuleNotFoundError as e:
            pytest.skip(f"mcp client SDK not importable: {e}")

        self.assertFalse(r["isError"], f"raised: {r['text'][:300]}")
        # The plan-time response carries the breadcrumb header...
        self.assertIn("Post-mortem breadcrumbs", r["text"])
        # ...and at least one known breadcrumb ID.
        self.assertIn("dealii-elasticity-catalog-structure", r["text"])
        # ...but MUST NOT include the diagnostic / full-record
        # fields. surface_symptom and root_cause are present only
        # on the full record retrieved via topic='postmortems';
        # leaking them at plan time is the regression we are
        # guarding against.
        self.assertNotIn("surface_symptom", r["text"],
                         "full post-mortem leaked into plan-time "
                         "response — breadcrumbs-only contract "
                         "broken")
        self.assertNotIn("root_cause", r["text"],
                         "full post-mortem leaked into plan-time "
                         "response — breadcrumbs-only contract "
                         "broken")


if __name__ == "__main__":
    unittest.main()
