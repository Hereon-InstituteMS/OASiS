"""Regression: per-backend `Signal:`-marker coverage on pitfalls
must not slip below the 2026-06-02 baseline.

WHY this matters
================
A pitfall's `Signal:` line is the critic-gate retrieval anchor.
When a simulation fails, the post-execution critic searches the
pitfall library for `Signal:` snippets that match the error
output and surfaces the matching pitfall + post-mortem record.
A pitfall without a `Signal:` line is invisible to that
retrieval path — the LLM may have a perfectly-written failure
diagnosis sitting in the catalog and never find it.

CURRENT BASELINE (2026-06-02)
=============================
A sweep across all 1068 catalog pitfalls in 198 physics rows
across 8 backends:

  kratos    : 147 / 147  (100.0%)   # Layer A/B promotion done
  dealii    :  96 / 138  ( 69.6%)
  skfem     :  51 / 103  ( 49.5%)
  ngsolve   :  64 / 135  ( 47.4%)
  febio     :   6 /  13  ( 46.2%)
  fenics    :  41 / 129  ( 31.8%)
  fourc     :  29 / 335  (  8.7%)   # heaviest gap
  dune      :   0 /  68  (  0.0%)   # complete miss

This test pins **percentage floors** per backend so:
  - new pitfalls added without Signal: markers degrade
    coverage and trip the test
  - existing Signal-less pitfalls that get rewritten WITH a
    Signal: line raise the floor naturally on re-record

If a new commit IMPROVES coverage, raise the floor in
SIGNAL_COVERAGE_MIN below to lock in the improvement.

This test is **inverted by design**: it accepts the current gaps
as known debt (fourc + dune especially) and prevents regression,
rather than pretending coverage is universally high.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


# Floor percentages locked in at the 2026-06-02 audit. Each
# floor is set ~1.5 percentage points below the measured value
# so a small reordering / re-counting noise does not trip the
# test, but a real regression (a new pitfall without Signal:
# pulling the average down) WILL.
SIGNAL_COVERAGE_MIN = {
    "kratos":  99.0,   # measured 100.0
    "dealii":  68.0,   # measured  69.6
    "skfem":   48.0,   # measured  49.5
    "ngsolve": 46.0,   # measured  47.4
    "febio":   45.0,   # measured  46.2
    "fenics":  50.0,   # measured  51.2 — CROSSED 50% (raised
                       #                 2026-06-02 from 44.0 after
                       #                 pass 3 on fenics:
                       #                 multiphase #1, #4, #6
                       #                 (interface-width
                       #                 oscillation, AC volume
                       #                 drift, level-set reinit)
                       #                 and deep_knowledge::
                       #                 thermal_structural #2, #3,
                       #                 #4, #5 (T_ref pre-strain,
                       #                 plane-strain/stress
                       #                 confusion, RBM null-space,
                       #                 Picard iteration). fenics
                       #                 31.8% -> 38.8% -> 45.7%
                       #                 -> 51.2% over three
                       #                 commits.)
    "fourc":   29.0,   # measured  30.1 — CROSSED 30% (raised
                       #                 2026-06-02 from 27.0 after
                       #                 a sixth pass: all 7
                       #                 xfem_fluid pitfalls
                       #                 received Signal: lines.
                       #                 fourc has gone 8.7% ->
                       #                 14.0% -> 18.2% -> 22.1%
                       #                 -> 25.4% -> 28.1% ->
                       #                 30.1% across seven
                       #                 commits — a 3.5x
                       #                 improvement.)
    "dune":     0.0,   # measured   0.0 — known total gap
}


def _pitfall_text(pit) -> str:
    if isinstance(pit, str):
        return pit
    if isinstance(pit, dict):
        return pit.get("text", "") or pit.get("description", "") or ""
    return str(pit)


class TestPitfallSignalCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, all_backends
        load_all_backends()
        cls.backends = all_backends()
        if not cls.backends:
            raise unittest.SkipTest("no backends registered")

    def test_signal_coverage_meets_floor(self) -> None:
        """No backend's Signal-marker coverage falls below its
        2026-06-02 floor. If you intentionally add Signal:
        markers (good!) and coverage climbs, RAISE the
        corresponding floor in SIGNAL_COVERAGE_MIN to lock the
        improvement in."""
        failures = []
        # Stable, sorted output for diagnostics.
        rows = []
        for b in self.backends:
            total = 0
            with_sig = 0
            for p in b.supported_physics():
                k = b.get_knowledge(p.name)
                if not isinstance(k, dict):
                    continue
                for pit in k.get("pitfalls", []):
                    total += 1
                    if "Signal:" in _pitfall_text(pit):
                        with_sig += 1
            if total == 0:
                continue
            pct = 100.0 * with_sig / total
            rows.append((b.name(), with_sig, total, pct))
            floor = SIGNAL_COVERAGE_MIN.get(b.name())
            if floor is None:
                failures.append(
                    (b.name(), with_sig, total, pct,
                     "no floor recorded — add one to "
                     "SIGNAL_COVERAGE_MIN at the 2026-06-02 "
                     "baseline value"))
                continue
            if pct < floor:
                failures.append(
                    (b.name(), with_sig, total, pct,
                     f"below {floor:.1f}% floor"))
        # Always render the per-backend table so failures and
        # green-builds both surface the current numbers.
        diagnostic = "\n".join(
            f"  {n:10s}: {s:4d}/{t:4d} ({p:5.1f}%)"
            for n, s, t, p in sorted(rows))
        if failures:
            fail_lines = "\n".join(
                f"  {n}: {s}/{t} ({p:.1f}%) -- {note}"
                for n, s, t, p, note in failures)
            self.fail(
                f"{len(failures)} backend(s) regressed below the "
                f"Signal:-coverage floor.\n\n"
                f"Current per-backend coverage:\n{diagnostic}\n\n"
                f"Regressions:\n{fail_lines}\n\n"
                "Either add a Signal: line to the new pitfall(s), "
                "or accept the new floor by editing "
                "SIGNAL_COVERAGE_MIN — but only AFTER confirming "
                "the new pitfall really has no observable signal.")


if __name__ == "__main__":
    unittest.main()
