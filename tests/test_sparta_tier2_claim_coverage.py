"""How much of the SPARTA pitfall catalog has an executed fixture.

MEASURING THE DENOMINATOR HONESTLY IS THE HARD PART HERE, harder than for any
other backend, because SPARTA's ten cross-cutting pitfalls are attached to EVERY
physics row by ``generators/__init__.py``::

    for _phys in KNOWLEDGE.values():
        _phys["pitfalls"] = list(_phys.get("pitfalls", [])) + list(UNIVERSAL_PITFALLS)

Walk the knowledge dict and those ten strings are counted ten times each: a plain
walk returns 165 Signal-bearing strings where there are 75 distinct claims. The
same shape inflated a DUNE count from 111 to 127, and there it was one shared
sub-dict; here it is a factor of 2.2 on the whole denominator, and it inflates in
the direction that makes coverage look WORSE, so it would not have been caught by
the number looking too good.

So claims are keyed by IDENTITY, not by position:

  * ``<physics>:<index>`` for a pitfall that belongs to one physics row;
  * ``universal:<n>`` for the n-th entry of UNIVERSAL_PITFALLS, once.

A separate count worth recording because it is the number an earlier pass
published: ``grep -c "Signal:"`` over ``src/backends/sparta/**/*.py`` returns 79.
Four of those are PROSE — docstrings in ``_common.py``, ``__init__.py`` and
``backend.py`` that mention the ``Signal:`` convention rather than making a claim.
The catalog states 75 falsifiable claims, and 75 is what this file divides by.

Coverage is CLAIM-ATTRIBUTED, not fixture-directory-counted, for the reason the
DUNE pass established: one fixture can legitimately verify several claims when
they share a deck, and nine of the SPARTA fixtures do. A fixture declares what it
covers::

    "covers": ["surface_interaction:2", "surface_interaction:3",
               "surface_interaction:9"]

and the declaration is CHECKED — an entry naming a claim that does not exist is
rejected, and no claim may be claimed twice. Fixtures with no ``covers`` key fall
back to their own physics/pitfall_index pair, which is how the older ones count.
An index past the end of a list is SYNTHETIC and counts for nothing; the catalog
smoke-test fixture uses one deliberately.

What this file does NOT do is judge whether a fixture is any good. That is
``scripts/run_tier2_fixtures.py``, which executes them. This one answers "is
there one at all", and keeps the answer honest by rejecting bogus keys and by
flooring the denominator so coverage cannot rise by deleting claims.
"""
from __future__ import annotations

import json
import pathlib
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_FIXTURES = _REPO / "scripts" / "tier2_fixtures" / "sparta"

# Raise this ONLY upward, and only with the fixtures that earned it in the same
# commit. The project owner's target is 80 %. The constant tracks what is
# MEASURED, so every commit is green and the number here is never an aspiration.
#   2026-08-06  measured 0.17 (13 of 75) before this pass
#   2026-08-06  0.39 (29 of 75) after the covers-attribution pass
#   2026-08-06  0.53 (40 of 75) after the three zero-coverage-area fixtures
#   2026-08-06  0.70 (53 of 75) after the four cross-cutting fixtures
MIN_COVERAGE_FRACTION = 0.70

# Counted 2026-08-06 by identity, not by walking the dict. See the module
# docstring for why those differ by a factor of 2.2 on this backend.
CLAIMS_COUNTED = 75


def _claim_inventory() -> dict[str, str]:
    """Every falsifiable SPARTA claim, keyed the way ``covers`` names it."""
    from backends.sparta.generators import (  # noqa: E402
        KNOWLEDGE, UNIVERSAL_PITFALLS,
    )

    universal = {text: f"universal:{i}"
                 for i, text in enumerate(UNIVERSAL_PITFALLS)}
    claims: dict[str, str] = dict.fromkeys(universal.values(), "")
    for text, key in universal.items():
        claims[key] = text

    for physics, entry in sorted(KNOWLEDGE.items()):
        if physics.startswith("_"):
            continue
        for i, text in enumerate(entry.get("pitfalls", [])):
            if not isinstance(text, str) or "Signal:" not in text:
                continue
            if text in universal:
                continue          # already counted once, under universal:<n>
            claims[f"{physics}:{i}"] = text
    return claims


def _positional_keys() -> set[str]:
    """`<physics>:<index>` for every pitfall slot, universal ones included.

    Needed only so a legacy fixture keyed by position at a slot holding a
    universal pitfall can be translated to that pitfall's identity key instead
    of being reported as bogus.
    """
    from backends.sparta.generators import KNOWLEDGE  # noqa: E402
    out = set()
    for physics, entry in KNOWLEDGE.items():
        if physics.startswith("_"):
            continue
        for i in range(len(entry.get("pitfalls", []))):
            out.add(f"{physics}:{i}")
    return out


def _positional_to_identity() -> dict[str, str]:
    from backends.sparta.generators import (  # noqa: E402
        KNOWLEDGE, UNIVERSAL_PITFALLS,
    )
    universal = {text: f"universal:{i}"
                 for i, text in enumerate(UNIVERSAL_PITFALLS)}
    out: dict[str, str] = {}
    for physics, entry in KNOWLEDGE.items():
        if physics.startswith("_"):
            continue
        for i, text in enumerate(entry.get("pitfalls", [])):
            if isinstance(text, str) and text in universal:
                out[f"{physics}:{i}"] = universal[text]
    return out


def _fixture_claims() -> tuple[dict[str, list[str]], list[str]]:
    """(claim -> fixtures naming it, fixtures with a synthetic key)."""
    covered: dict[str, list[str]] = {}
    synthetic: list[str] = []
    inventory = _claim_inventory()
    translate = _positional_to_identity()
    for d in sorted(p.parent for p in _FIXTURES.glob("*/fixture.json")):
        meta = json.loads((d / "fixture.json").read_text())
        declared = meta.get("covers")
        if declared is None:
            key = f"{meta.get('physics')}:{int(meta.get('pitfall_index', -1))}"
            key = translate.get(key, key)
            declared = [key] if key in inventory else []
            if not declared:
                synthetic.append(f"{d.name} ({key})")
        elif not declared:
            synthetic.append(f"{d.name} (covers: [] — declared as not evidence)")
        for claim in declared:
            covered.setdefault(translate.get(str(claim), str(claim)),
                               []).append(d.name)
    return covered, synthetic


class TestSpartaTier2ClaimCoverage(unittest.TestCase):

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
            f"these fixtures claim to cover SPARTA pitfalls that do not "
            f"exist: {bogus}. Pitfall indices are POSITIONAL — inserting an "
            f"entry earlier in a list re-points every fixture after it — so a "
            f"stale index here means the coverage number is fiction. Re-read "
            f"the list in src/backends/sparta/generators/ and fix the index.")

    def test_no_claim_is_counted_twice(self) -> None:
        dupes = {c: f for c, f in self.covered.items() if len(f) > 1}
        self.assertEqual(
            dupes, {},
            f"these claims are named by more than one fixture: {dupes}. "
            f"Double-counting inflates the coverage fraction; pick one owner "
            f"per claim.")

    def test_the_ten_universal_pitfalls_are_counted_once_each(self) -> None:
        """The defect this whole keying scheme exists for.

        UNIVERSAL_PITFALLS is appended to all ten physics rows, so a positional
        walk sees each of those ten strings ten times. If the inventory ever
        grows past 75 by that route, the denominator has silently inflated and
        every coverage number computed from it is wrong.
        """
        from backends.sparta.generators import UNIVERSAL_PITFALLS  # noqa: E402
        universal_keys = [k for k in self.inventory
                          if k.startswith("universal:")]
        self.assertEqual(
            len(universal_keys), len(UNIVERSAL_PITFALLS),
            f"expected each of the {len(UNIVERSAL_PITFALLS)} universal "
            f"pitfalls once, got {len(universal_keys)} keys")
        positional = _positional_keys()
        self.assertGreater(
            len(positional), len(self.inventory),
            "a positional walk should see MORE slots than there are distinct "
            "claims on this backend; if it does not, the universal pitfalls "
            "are no longer being shared and this test's premise has changed")

    def test_coverage_meets_the_floor(self) -> None:
        n_total = len(self.inventory)
        n_covered = len([c for c in self.covered if c in self.inventory])
        fraction = n_covered / n_total if n_total else 0.0
        uncovered = sorted(set(self.inventory) - set(self.covered))
        self.assertGreaterEqual(
            fraction, MIN_COVERAGE_FRACTION,
            f"SPARTA tier-2 claim coverage is {n_covered}/{n_total} = "
            f"{fraction:.1%}, below the {MIN_COVERAGE_FRACTION:.0%} floor. "
            f"Still uncovered:\n  " + "\n  ".join(uncovered))

    def test_the_inventory_itself_did_not_shrink(self) -> None:
        """Coverage is a fraction, so shrinking the denominator is a way to
        'improve' it without writing anything. 75 distinct claims were counted
        on 2026-08-06; deleting claims to raise the percentage has to be a
        deliberate, visible edit."""
        self.assertGreaterEqual(
            len(self.inventory), CLAIMS_COUNTED,
            f"the SPARTA claim inventory dropped to {len(self.inventory)} "
            f"from the {CLAIMS_COUNTED} counted on 2026-08-06. If claims were "
            f"retired because execution falsified them, lower this number in "
            f"the same commit and say which ones; do not let the coverage "
            f"fraction rise by subtraction.")


if __name__ == "__main__":                                  # pragma: no cover
    unittest.main()
