"""The recorded pass record must describe the fixtures that exist.

WHAT WENT WRONG
---------------
`scripts/scan_results/tier2_results.json` is the record of which fixtures ran
and passed. `tests/test_signal_verification.py` gates on it, and
`scripts/verify_signal_clauses.py` reads it.

Measured on one branch: **303 fixture directories on disk, 108 rows recorded.**
The file had not been rewritten as fixtures were added, so two thirds of the
suite had no entry — and a gate reading it was certifying a pass count that
described a tree from weeks earlier. Nothing failed; the number was simply about
a different suite than the one in the repository.

That is the same shape as every other defect this project keeps finding: not a
wrong answer, but a right-looking answer about the wrong thing. It is worse here
than most, because this file is what "the fixtures pass" MEANS when someone
asks for evidence.

A SECOND DEFECT IN THE SAME FILE
--------------------------------
Three keys are recorded more than once — `elasticity_mms_convergence` four
times, `poisson_mms_convergence` four, `stokes_mms_convergence` three. Different
BACKENDS reuse the same fixture NAME, and the results are keyed by name, so each
new backend's result overwrites the previous one's. Ten distinct runs collapse
to three rows, and whichever backend ran last is the one the record describes.

WHAT THIS GATE DOES
-------------------
It does not require the file to be regenerated on every commit — running 300
fixtures takes hours and several need solvers that are not on every machine. It
requires the file to be HONEST about what it covers: if it claims fewer fixtures
than exist, that must be visible rather than silently read as full coverage, and
its keys must distinguish fixtures that are genuinely different.

The tolerance is deliberately generous. The point is to catch a record that has
drifted far from the tree, not to demand it be perfectly current.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "scripts" / "scan_results" / "tier2_results.json"
FIXTURES = REPO / "scripts" / "tier2_fixtures"

# How far the record may lag the tree before it stops meaning anything. A record
# covering under half the suite is not "slightly stale", it is about a different
# suite.
MIN_COVERAGE_OF_TREE = 0.50


def _rows(payload) -> list:
    if isinstance(payload, dict):
        for key in ("results", "fixtures", "rows"):
            v = payload.get(key)
            if isinstance(v, list):
                return v
            if isinstance(v, dict):
                return list(v.values())
        return [v for v in payload.values() if isinstance(v, dict)]
    return payload if isinstance(payload, list) else []


def _fixture_dirs() -> list[Path]:
    if not FIXTURES.is_dir():
        return []
    return [d for be in FIXTURES.iterdir() if be.is_dir()
            for d in be.iterdir() if (d / "fixture.json").is_file()]


def test_results_file_covers_most_of_the_tree():
    """A record about a third of the suite must not read as a full pass."""
    if not RESULTS.is_file():
        pytest.skip("no tier2_results.json — nothing claims to be a record")
    on_disk = _fixture_dirs()
    if not on_disk:
        pytest.skip("no fixtures in this tree")

    rows = _rows(json.loads(RESULTS.read_text()))
    ratio = len(rows) / len(on_disk)

    assert ratio >= MIN_COVERAGE_OF_TREE, (
        f"tier2_results.json records {len(rows)} results but {len(on_disk)} "
        f"fixtures exist ({ratio:.0%} of the tree). A gate reads this file as "
        f"the pass record, so it is currently certifying a suite that no longer "
        f"matches the repository.\n\n"
        f"Regenerate it — `scripts/run_tier2_fixtures.py --write-results` — or, "
        f"if the fixtures cannot all run on this machine, record that explicitly "
        f"so nobody mistakes a partial record for a complete one. Note the "
        f"`--write-results` flag was itself parsed and never read for a while, "
        f"so confirm the file's timestamp actually changes.")


def test_results_keys_distinguish_different_fixtures():
    """Two different fixtures must not share one row.

    Measured: `elasticity_mms_convergence` appears four times, once per backend
    that happens to use that name. Keyed by name alone, each backend's result
    overwrites the last, so ten runs become three rows and the record describes
    whichever backend ran most recently.
    """
    if not RESULTS.is_file():
        pytest.skip("no tier2_results.json")
    rows = _rows(json.loads(RESULTS.read_text()))
    if not rows:
        pytest.skip("results file records nothing")

    keys = Counter()
    for r in rows:
        if not isinstance(r, dict):
            continue
        key = r.get("fixture_id") or r.get("id") or r.get("name")
        if key is None:
            key = f"{r.get('backend')}::{r.get('physics')}::{r.get('pitfall_index')}"
        keys[str(key)] += 1

    collisions = {k: n for k, n in keys.items() if n > 1}
    assert not collisions, (
        f"{len(collisions)} result keys appear more than once, so results "
        f"overwrite each other and the record describes only the last writer:\n"
        + "\n".join(f"  {k} x{n}" for k, n in list(collisions.items())[:10])
        + "\n\nDifferent backends legitimately reuse fixture NAMES — "
          "elasticity_mms_convergence exists under several. Key the record by "
          "backend AND name so each run keeps its own row.")
