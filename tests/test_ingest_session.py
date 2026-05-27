"""Tests for the session-journal ingest path -- both the MCP-side
``session_insights('ingest', path=...)`` flow and the standalone CLI
``scripts/ingest_session.py``.

The fixture builds a small synthetic journal that exercises the
"error -> source-read -> retry -> success" pattern the analyzer is
designed to surface, and checks the resulting candidate is correctly
extracted, de-duplicated across multiple files, and saved to the
``community_knowledge/pending/`` staging dir when approved.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.session_analyzer import analyze_journal_file  # noqa: E402
from core.session_journal import SessionJournal  # noqa: E402


def _build_fixture_journal(session_id: str) -> SessionJournal:
    """Synthesise a journal that the analyzer should surface as a
    knowledge candidate.

    Pattern: a tool call fails with a specific error, the agent reads
    the relevant source file, retries with a fixed parameter, and
    succeeds.  This is the "error -> success" pattern enumerated in
    session_analyzer's module docstring.
    """
    j = SessionJournal(session_id=session_id, started_at=1_700_000_000.0)
    j.record("tool_call", "run_simulation", solver="fourc", physics="solid_mechanics")
    j.record(
        "tool_error", "run_simulation", solver="fourc", physics="solid_mechanics",
        error_message=(
            "FOUR_C_THROW: KINEM must be nonlinear for ElastHyper material"
        ),
    )
    j.record(
        "source_read", "developer", solver="fourc",
        details={"file": "src/mat/4C_mat_elasthyper.cpp"},
    )
    j.record(
        "parameter_override", "run_simulation",
        solver="fourc", physics="solid_mechanics",
        details={"KINEM": "nonlinear"},
    )
    j.record("tool_success", "run_simulation", solver="fourc", physics="solid_mechanics")
    return j


class TestIngestSingleJournal(unittest.TestCase):
    """Loading a single saved journal and running the analyzer."""

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            saved = _build_fixture_journal("fix-001").save(tmp_path)
            self.assertTrue(saved.exists())
            candidates = analyze_journal_file(saved)
            self.assertTrue(candidates, "fixture should produce >=1 candidate")
            c = candidates[0]
            self.assertEqual(c.solver, "fourc")
            self.assertEqual(c.physics, "solid_mechanics")
            # The fixture's FOUR_C_THROW error message should appear in
            # the candidate description -- exact format is up to the
            # analyzer, but the substring must survive.
            self.assertIn("FOUR_C_THROW", c.description)


class TestIngestBatchDedup(unittest.TestCase):
    """The CLI / MCP ingest flow should collapse identical candidates
    from multiple journals into one entry."""

    def test_three_identical_journals_collapse_to_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for i in range(3):
                _build_fixture_journal(f"fix-{i:03d}").save(tmp_path)
            files = sorted(tmp_path.glob("session_*.json"))
            self.assertEqual(len(files), 3)

            # Aggregate across files exactly the way ingest_session.py does.
            all_candidates = []
            for f in files:
                all_candidates.extend(analyze_journal_file(f))
            best = {}
            for c in all_candidates:
                key = (c.category, c.solver or "", c.title)
                if key not in best or c.confidence > best[key].confidence:
                    best[key] = c
            deduped = list(best.values())

            # At least one raw candidate per file is expected; dedup should
            # collapse identical ones.
            self.assertGreaterEqual(len(all_candidates), 3)
            self.assertLess(len(deduped), len(all_candidates))


class TestIngestCliApprove(unittest.TestCase):
    """The standalone CLI's --approve flag must drop a JSON file in
    ``community_knowledge/pending/`` so a contributor can PR it.
    """

    def test_approve_writes_pending_file(self):
        # Use a temp directory for both journals and (via env override
        # of the repo root) the pending output.  The CLI writes
        # relative to REPO_ROOT which is hard-coded at import; rather
        # than mocking that, we point at the live pending/ dir and
        # remove the artefact afterwards.
        pending_dir = REPO_ROOT / "data" / "community_knowledge" / "pending"
        before = set(pending_dir.glob("session_batch_*.json")) if pending_dir.exists() else set()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _build_fixture_journal("approve-fix-001").save(tmp_path)

            result = subprocess.run(
                [
                    sys.executable, str(REPO_ROOT / "scripts" / "ingest_session.py"),
                    str(tmp_path), "--approve",
                ],
                capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Saved", result.stdout)

        after = set(pending_dir.glob("session_batch_*.json"))
        new_files = sorted(after - before)
        self.assertEqual(len(new_files), 1, "exactly one batch file expected")

        try:
            entries = json.loads(new_files[0].read_text())
            self.assertIsInstance(entries, list)
            self.assertGreaterEqual(len(entries), 1)
            self.assertEqual(entries[0].get("solver"), "fourc")
        finally:
            for f in new_files:
                f.unlink(missing_ok=True)


class TestIngestSolverFilter(unittest.TestCase):
    """The CLI's ``--solver`` flag must restrict candidates to that
    solver only."""

    def test_filter_drops_other_solvers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Two journals, two different solvers; both yield candidates.
            j1 = _build_fixture_journal("filter-fourc-001")
            j1.save(tmp_path)
            j2 = SessionJournal(session_id="filter-skfem-001", started_at=1_700_000_000.0)
            j2.record("tool_call", "run_simulation", solver="skfem", physics="poisson")
            j2.record(
                "tool_error", "run_simulation", solver="skfem", physics="poisson",
                error_message="ImportError: cannot import name ElementTriP9",
            )
            j2.record("source_read", "developer", solver="skfem")
            j2.record(
                "parameter_override", "run_simulation",
                solver="skfem", physics="poisson",
                details={"element": "ElementTriP3"},
            )
            j2.record("tool_success", "run_simulation", solver="skfem", physics="poisson")
            j2.save(tmp_path)

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "ingest_session.py"),
                    str(tmp_path),
                    "--solver", "fourc",
                ],
                capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        # The fourc candidate must appear; the skfem candidate must not.
        self.assertIn("Solver: fourc", result.stdout)
        self.assertNotIn("Solver: skfem", result.stdout)


class TestIngestMixedBatchPartialFailure(unittest.TestCase):
    """A batch containing one malformed journal and one good one must
    surface the error for the bad file and still extract candidates
    from the good one -- not bail out on the first failure."""

    def test_one_bad_one_good(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Good fixture journal.
            _build_fixture_journal("mixed-good-001").save(tmp_path)
            # Truly malformed: not valid JSON.  `SessionJournal.load()`
            # tolerates missing optional fields, so we need an outright
            # parse failure to exercise the error path.
            (tmp_path / "session_mixed-bad-001.json").write_text(
                "{not even JSON"
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "ingest_session.py"),
                    str(tmp_path),
                ],
                capture_output=True, text=True, timeout=30,
            )
        # CLI itself succeeds (partial failure is reported, not raised).
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Errors:", result.stdout)
        self.assertIn("session_mixed-bad-001.json", result.stdout)
        # Still surfaces the good-journal candidate.
        self.assertIn("Solver: fourc", result.stdout)


class TestIngestCliDryRun(unittest.TestCase):
    """Dry-run mode (no --approve) must NOT touch the pending/ dir."""

    def test_dry_run_does_not_write(self):
        pending_dir = REPO_ROOT / "data" / "community_knowledge" / "pending"
        before = set(pending_dir.glob("session_batch_*.json")) if pending_dir.exists() else set()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _build_fixture_journal("dryrun-fix-001").save(tmp_path)
            result = subprocess.run(
                [sys.executable, str(REPO_ROOT / "scripts" / "ingest_session.py"), str(tmp_path)],
                capture_output=True, text=True, timeout=30,
            )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Dry run", result.stdout)

        after = set(pending_dir.glob("session_batch_*.json")) if pending_dir.exists() else set()
        self.assertEqual(before, after, "dry run must not write any files")


if __name__ == "__main__":
    unittest.main(verbosity=2)
