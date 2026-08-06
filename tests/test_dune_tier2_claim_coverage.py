"""How much of the DUNE-fem pitfall catalog has an executed fixture.

The DUNE backend states 111 falsifiable claims — every entry of every
``pitfalls`` list, all of which carry a ``Signal:`` clause, plus the
``Signal*`` keys of the measured sections in ``_general``. A claim with
no fixture is a claim the project cannot defend, so this file measures
the fraction that has one and refuses to let it fall.

Measuring it needs a MAP, because one fixture may legitimately verify
several claims: DUNE JIT-compiles C++ for every distinct form, so
splitting claims that share a compiled module into separate fixtures
multiplies build time without adding evidence. A fixture therefore
declares a ``covers`` list in its ``fixture.json``::

    "covers": ["poisson:1", "poisson:2", "heat:2",
               "_general:assemble_measured.Signal"]

Fixtures without a ``covers`` key fall back to their own
``physics``/``pitfall_index`` pair, which is how the older fixtures are
counted. An index past the end of a pitfall list is a SYNTHETIC key
(the MMS convergence gate uses one) and counts for nothing here.

What this file does NOT do: judge whether a fixture is any good. That
is the runner's job — ``scripts/run_tier2_fixtures.py`` executes them
and ``test_signal_verification.py`` gates the pass count. This file
only answers "is there one at all", and keeps the answer honest by
rejecting a ``covers`` entry that names a claim which does not exist.
"""
from __future__ import annotations

import json
import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_FIXTURES = _REPO / "scripts" / "tier2_fixtures" / "dune"

# Raise this ONLY upward, and only with the fixtures that earned it in
# the same commit. 2026-08-06: 80 % floor set by the project owner.
MIN_COVERAGE_FRACTION = 0.80


def _claim_inventory() -> dict[str, str]:
    """Every falsifiable DUNE claim, keyed the way ``covers`` names it."""
    from backends.dune.generators import KNOWLEDGE           # noqa: E402

    claims: dict[str, str] = {}
    for physics, entry in KNOWLEDGE.items():
        pitfalls = entry.get("pitfalls")
        if isinstance(pitfalls, list):
            for i, text in enumerate(pitfalls):
                if isinstance(text, str) and "Signal:" in text:
                    claims[f"{physics}:{i}"] = text
    general = KNOWLEDGE.get("_general", {})
    for section, body in general.items():
        if not isinstance(body, dict):
            continue
        for key, text in body.items():
            # START_HERE points the reader AT the pitfall list; it is a
            # lookup instruction, not a claim about the code.
            if section == "START_HERE":
                continue
            if (isinstance(key, str) and key.startswith("Signal")
                    or (isinstance(text, str) and "Signal:" in text
                        and isinstance(key, str))):
                if isinstance(text, str) and "Signal:" in text:
                    claims[f"_general:{section}.{key}"] = text
    return claims


def _fixture_claims() -> tuple[dict[str, list[str]], list[str]]:
    """(claim -> fixtures naming it, fixtures with a synthetic key)."""
    covered: dict[str, list[str]] = {}
    synthetic: list[str] = []
    inventory = _claim_inventory()
    for d in sorted(p.parent for p in _FIXTURES.glob("*/fixture.json")):
        meta = json.loads((d / "fixture.json").read_text())
        declared = meta.get("covers")
        if declared is None:
            key = f"{meta.get('physics')}:{int(meta.get('pitfall_index', -1))}"
            declared = [key] if key in inventory else []
            if not declared:
                synthetic.append(f"{d.name} ({key})")
        for claim in declared:
            covered.setdefault(str(claim), []).append(d.name)
    return covered, synthetic


class TestDuneTier2ClaimCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = _claim_inventory()
        cls.covered, cls.synthetic = _fixture_claims()

    def test_every_covers_entry_names_a_real_claim(self) -> None:
        """A `covers` list is a coverage CLAIM; it has to be checkable."""
        bogus = {c: f for c, f in self.covered.items()
                 if c not in self.inventory}
        self.assertEqual(
            bogus, {},
            f"these fixtures claim to cover DUNE pitfalls that do not "
            f"exist: {bogus}. Pitfall indices are POSITIONAL — "
            f"inserting an entry earlier in a list re-points every "
            f"fixture after it — so a stale index here means the "
            f"coverage number is fiction. Re-read the list in "
            f"src/backends/dune/generators/ and fix the index.")

    def test_no_claim_is_counted_twice(self) -> None:
        dupes = {c: f for c, f in self.covered.items() if len(f) > 1}
        self.assertEqual(
            dupes, {},
            f"these claims are named by more than one fixture: {dupes}. "
            f"Double-counting inflates the coverage fraction; pick one "
            f"owner per claim.")

    def test_coverage_meets_the_floor(self) -> None:
        n_total = len(self.inventory)
        n_covered = len([c for c in self.covered if c in self.inventory])
        fraction = n_covered / n_total if n_total else 0.0
        uncovered = sorted(set(self.inventory) - set(self.covered))
        self.assertGreaterEqual(
            fraction, MIN_COVERAGE_FRACTION,
            f"DUNE tier-2 claim coverage is {n_covered}/{n_total} = "
            f"{fraction:.1%}, below the {MIN_COVERAGE_FRACTION:.0%} "
            f"floor. Still uncovered:\n  " + "\n  ".join(uncovered))

    def test_the_inventory_itself_did_not_shrink(self) -> None:
        """Coverage is a fraction, so shrinking the denominator is a
        way to 'improve' it without writing anything. 111 claims were
        counted on 2026-08-06; deleting claims to raise the percentage
        has to be a deliberate, visible edit."""
        self.assertGreaterEqual(
            len(self.inventory), 111,
            f"the DUNE claim inventory dropped to "
            f"{len(self.inventory)} from the 111 counted on "
            f"2026-08-06. If claims were retired because execution "
            f"falsified them, lower this number in the same commit and "
            f"say which ones; do not let the coverage fraction rise by "
            f"subtraction.")


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
