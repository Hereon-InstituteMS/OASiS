"""Signal-clause verification regression test.

Establishes a floor for the quality of `Signal:` clauses in the
pitfall DB so the project cannot ship a worse claim than what is
shipped today. The harness in `scripts/verify_signal_clauses.py`
computes the metrics; this test asserts none of them regress.

This is the merge-gate the senior-AI-scientist critic (2026-05-31)
called for as the second-largest risk (Signal: clauses are an
unfalsifiable contract — make them at least falsifiable).

What this test does NOT do: claim every signal is real (Tier 2,
intentional-failure regression fixtures, is multi-week work). It
establishes Tier 0 (Signal references a real entity in the
canonical catalogs) and Tier 1 (Signal uses observable-symptom
vocabulary) as the falsifiability floor.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


class TestDealiiSignalFloor(unittest.TestCase):
    """deal.II pitfalls today — established floor as of 2026-05-31.

    These numbers come from running
    ``python scripts/verify_signal_clauses.py --backend dealii``
    after the canonical-element refactor (commit f748716). They
    are a FLOOR not a TARGET — encoding work should monotonically
    push them upward.
    """

    # When updating these numbers, do so ONLY upward — that means
    # the catalog has improved. A downward edit means a regression
    # snuck through and needs to be re-examined.
    MIN_N_PITFALLS = 96
    MIN_WITH_CATEGORY_PREFIX = 63
    MIN_WITH_SIGNAL_CLAUSE = 61
    MIN_TIER0_PASSED = 15
    MIN_TIER1_PASSED = 44
    MIN_TIER0_AND_1_PASSED = 12

    def setUp(self):
        from verify_signal_clauses import verify_backend
        self.results = verify_backend("dealii")

    def _count(self, predicate) -> int:
        return sum(1 for r in self.results if predicate(r))

    def test_total_pitfall_count_does_not_regress(self):
        n = len(self.results)
        self.assertGreaterEqual(
            n, self.MIN_N_PITFALLS,
            f"deal.II pitfall count dropped from "
            f"{self.MIN_N_PITFALLS} to {n} — a regression. "
            f"Either a pitfall got accidentally deleted or the "
            f"harvest path broke.")

    def test_category_prefix_coverage_does_not_regress(self):
        n = self._count(
            lambda r: r.pitfall_category != "(no-prefix)")
        self.assertGreaterEqual(
            n, self.MIN_WITH_CATEGORY_PREFIX,
            f"deal.II pitfalls with [Category] prefix dropped "
            f"from {self.MIN_WITH_CATEGORY_PREFIX} to {n}. "
            f"PR #26 discipline being violated.")

    def test_signal_clause_coverage_does_not_regress(self):
        n = self._count(lambda r: bool(r.signal_text))
        self.assertGreaterEqual(
            n, self.MIN_WITH_SIGNAL_CLAUSE,
            f"deal.II pitfalls with Signal: clause dropped "
            f"from {self.MIN_WITH_SIGNAL_CLAUSE} to {n}.")

    def test_tier0_floor_does_not_regress(self):
        n = self._count(lambda r: r.tier0_passed)
        self.assertGreaterEqual(
            n, self.MIN_TIER0_PASSED,
            f"deal.II pitfalls passing Tier 0 (Signal references "
            f"a canonical entity) dropped from "
            f"{self.MIN_TIER0_PASSED} to {n}.")

    def test_tier1_floor_does_not_regress(self):
        n = self._count(lambda r: r.tier1_passed)
        self.assertGreaterEqual(
            n, self.MIN_TIER1_PASSED,
            f"deal.II pitfalls passing Tier 1 (Signal uses "
            f"observable-symptom vocabulary) dropped from "
            f"{self.MIN_TIER1_PASSED} to {n}.")

    def test_tier0_and_1_floor_does_not_regress(self):
        n = self._count(
            lambda r: r.tier0_passed and r.tier1_passed)
        self.assertGreaterEqual(
            n, self.MIN_TIER0_AND_1_PASSED,
            f"deal.II pitfalls passing BOTH Tier 0 AND Tier 1 "
            f"dropped from {self.MIN_TIER0_AND_1_PASSED} to {n}. "
            f"This is the strictest floor.")


class TestHarnessSelfChecks(unittest.TestCase):
    """The harness itself must do what it claims."""

    def test_split_pitfall_handles_well_formed_entry(self):
        from verify_signal_clauses import _split_pitfall
        cat, sig = _split_pitfall(
            "[Numerical] SUPG stabilisation parameter ... "
            "Signal: spurious oscillations near boundary layers.")
        self.assertEqual(cat, "Numerical")
        self.assertIn("spurious oscillations", sig)

    def test_split_pitfall_handles_missing_prefix(self):
        from verify_signal_clauses import _split_pitfall
        cat, sig = _split_pitfall(
            "Use FE_Q for elasticity. Signal: rank-deficient stiffness.")
        self.assertIsNone(cat)
        self.assertIn("rank-deficient", sig)

    def test_split_pitfall_handles_missing_signal(self):
        from verify_signal_clauses import _split_pitfall
        cat, sig = _split_pitfall(
            "[Syntax] Element name needs node count suffix.")
        self.assertEqual(cat, "Syntax")
        self.assertIsNone(sig)


if __name__ == "__main__":
    unittest.main()
