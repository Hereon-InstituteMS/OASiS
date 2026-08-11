"""How much of the FEniCSx pitfall catalog has an executed fixture.

Every entry of every ``pitfalls`` list in
``backends.fenics.generators.*.KNOWLEDGE`` that carries a ``Signal:`` clause is
a falsifiable claim. A claim with no fixture is a claim the project cannot
defend, so this file measures the fraction that has one and refuses to let it
fall.

Measuring it needs a MAP rather than a directory count, because the two do not
agree in either direction. One fixture can legitimately verify several claims,
and — the direction that actually bit this backend — a fixture can sit in the
directory while pointing at the WRONG claim. Five of the fenics fixtures did:
``xdmf_degree_mismatch`` executed the XDMF degree-mismatch claim while keyed to
the near-incompressible locking claim, so a real claim looked defended and the
evidence was about something else entirely. Pitfall indices are POSITIONAL, so
inserting an entry earlier in a list silently re-points every fixture after it,
and the fixture keeps passing, because passing only means "the software printed
what I expected".

A fixture therefore declares what it covers::

    "covers": ["poisson::3", "poisson::4"]

Fixtures with no ``covers`` key fall back to their own
``physics``/``pitfall_index`` pair, which is how the older ones are counted. An
index past the end of a pitfall list is a SYNTHETIC key — the MMS convergence
gates and the catalog-shape checks use one — and counts for nothing here, which
is the point: a check that is not evidence for a stated claim must not be able
to inflate the number.

What this file does NOT do: judge whether a fixture is any good. That is the
runner's job (``scripts/run_tier2_fixtures.py`` executes them) and the mutation
driver's (``.t2check.py`` re-runs each with ``T2_MUTATE=1`` and requires an
expectation to be lost). This file only answers "is there one at all, and does
it point at the claim it tests".
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import unittest

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

BACKEND = "fenics"
_FIXTURES = _REPO / "scripts" / "tier2_fixtures" / BACKEND

# Raise this ONLY upward, and only with the fixtures that earned it in the same
# commit. The project owner's target is 80 %; the constant tracks what is
# actually MEASURED so every commit is green and the number here is never an
# aspiration.
#   2026-08-06  0.06 -> 0.69  (session fixtures, minus five mis-keyed ones
#                              whose claims went back to uncovered)
#   2026-08-06  0.69 -> 0.86  (31 more fixtures; 169 of 196 — past the target)
MIN_COVERAGE_FRACTION = 0.86

# Counted on 2026-08-06. Coverage is a fraction, so shrinking the denominator
# is a way to "improve" it without writing anything.
# Claim-bearing fixtures that predate the T2_MUTATE convention and would
# therefore pass with the pathology absent. Measured 2026-08-06; may only
# go DOWN.
MUTATION_CONTROL_DEBT = 6

CLAIM_INVENTORY_FLOOR = 196


def _claim_inventory() -> dict[str, str]:
    """Every falsifiable fenics claim, keyed the way ``covers`` names it.

    Harvested through the backend API, which is the same path the agent uses to
    serve the knowledge — so the denominator is what a user actually receives,
    not what a grep of the source happens to find. (Those two differ here: an
    AST sweep of the generator files counts 202, because some pitfall lists are
    not reachable from any supported_physics() entry.)
    """
    from core.registry import get_backend, load_all_backends       # noqa: E402

    try:
        load_all_backends()
    except Exception:                                     # pragma: no cover
        pass
    backend = get_backend("fenics")
    if backend is None:                                   # pragma: no cover
        raise unittest.SkipTest("fenics backend not registered")

    claims: dict[str, str] = {}
    for physics in backend.supported_physics():
        try:
            knowledge = backend.get_knowledge(physics.name)
        except Exception:                                 # pragma: no cover
            continue
        if not isinstance(knowledge, dict):
            continue
        for i, text in enumerate(knowledge.get("pitfalls", []) or []):
            if isinstance(text, str) and "Signal:" in text:
                claims[f"{physics.name}::{i}"] = text
    return claims


def _tracked_fixture_names() -> set[str]:
    """Fixture directories git tracks.

Only fixtures git TRACKS are counted. A fixture sitting untracked in the
working tree has not been committed, which on this project means it has not
been run through the mutation driver here — it is not evidence yet. Counting it
would let half-written work inflate the number, and a fixture caught mid-write
would turn this test red for reasons that have nothing to do with coverage.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO), "ls-files",
             f"scripts/tier2_fixtures/{BACKEND}"],
            capture_output=True, text=True, timeout=60).stdout.split()
    except (OSError, subprocess.SubprocessError):        # pragma: no cover
        return set()
    names = set()
    for path in out:
        parts = path.split("/")
        if len(parts) > 3:
            names.add(parts[3])
    return names


def _fixture_claims(inventory: dict[str, str]):
    """(claim -> fixtures naming it, fixtures whose key names no claim)."""
    covered: dict[str, list[str]] = {}
    synthetic: list[str] = []
    tracked = _tracked_fixture_names()
    for d in sorted(p.parent for p in _FIXTURES.glob("*/fixture.json")):
        if tracked and d.name not in tracked:
            continue
        meta = json.loads((d / "fixture.json").read_text())
        declared = meta.get("covers")
        if declared is None:
            key = f"{meta.get('physics')}::{meta.get('pitfall_index')}"
            declared = [key] if key in inventory else []
            if not declared:
                synthetic.append(f"{d.name} ({key})")
        elif not declared:
            key = f"{meta.get('physics')}::{meta.get('pitfall_index')}"
            synthetic.append(f"{d.name} ({key}, covers: [])")
        for claim in declared:
            covered.setdefault(str(claim), []).append(d.name)
    return covered, synthetic


class TestFenicsTier2ClaimCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.inventory = _claim_inventory()
        cls.covered, cls.synthetic = _fixture_claims(cls.inventory)

    def test_every_covers_entry_names_a_real_claim(self) -> None:
        """A `covers` list is a coverage CLAIM; it has to be checkable."""
        bogus = {c: f for c, f in self.covered.items()
                 if c not in self.inventory}
        self.assertEqual(
            bogus, {},
            f"these fixtures claim to cover fenics pitfalls that do not "
            f"exist: {bogus}. Pitfall indices are POSITIONAL — inserting an "
            f"entry earlier in a list re-points every fixture after it — so a "
            f"stale index here means the coverage number is fiction. Re-read "
            f"the list in src/backends/fenics/generators/ and fix the index.")

    def test_no_claim_is_counted_twice(self) -> None:
        dupes = {c: f for c, f in self.covered.items() if len(f) > 1}
        self.assertEqual(
            dupes, {},
            f"these claims are named by more than one fixture: {dupes}. "
            f"Double-counting inflates the coverage fraction; pick one owner "
            f"per claim and make the other synthetic or delete it.")

    def test_coverage_meets_the_floor(self) -> None:
        n_total = len(self.inventory)
        n_covered = len([c for c in self.covered if c in self.inventory])
        fraction = n_covered / n_total if n_total else 0.0
        uncovered = sorted(set(self.inventory) - set(self.covered))
        self.assertGreaterEqual(
            fraction, MIN_COVERAGE_FRACTION,
            f"fenics tier-2 claim coverage is {n_covered}/{n_total} = "
            f"{fraction:.1%}, below the {MIN_COVERAGE_FRACTION:.0%} floor. "
            f"Still uncovered:\n  " + "\n  ".join(uncovered))

    def test_the_inventory_itself_did_not_shrink(self) -> None:
        self.assertGreaterEqual(
            len(self.inventory), CLAIM_INVENTORY_FLOOR,
            f"the fenics claim inventory dropped to {len(self.inventory)} "
            f"from the {CLAIM_INVENTORY_FLOOR} counted on 2026-08-06. If "
            f"claims were retired because execution falsified them, lower "
            f"this floor in the same commit and say which ones; do not let "
            f"the coverage fraction rise by subtraction.")

    def test_mutation_control_debt_does_not_grow(self) -> None:
        """A fixture that counts toward coverage should FAIL when the pathology
        is removed. The ones below predate that convention: they pass, but they
        would pass with the pitfall absent too, so the claim they defend is
        defended weakly. This test does not demand they be fixed today — it
        pins the count so the debt cannot quietly grow while the coverage
        fraction rises. Lower the number as controls are added.

        The mutation control itself is `T2_MUTATE=1` in the fixture's source
        (or in the shared translation unit its cmd.sh names), which runs the
        CORRECT variant; the driver then requires an expectation to be lost.
        """
        import re
        tracked = _tracked_fixture_names()
        without = []
        for d in sorted(p.parent for p in _FIXTURES.glob("*/fixture.json")):
            if tracked and d.name not in tracked:
                continue
            meta = json.loads((d / "fixture.json").read_text())
            declared = meta.get("covers")
            if declared == []:
                continue                      # synthetic: defends no claim
            key = f"{meta.get('physics')}::{meta.get('pitfall_index')}"
            if declared is None and key not in self.inventory:
                continue                      # key names no claim: counts for 0
            text = ""
            for name in ("source.py", "cmd.sh"):
                p = d / name
                if p.exists():
                    text += p.read_text()
            shared = ""
            m = re.search(r"run\.sh\"?\s+(\w+)", text)
            if m:
                cc = _FIXTURES / "_shared" / f"{m.group(1)}.cc"
                if cc.exists():
                    shared = cc.read_text()
            if "T2_MUTATE" not in text + shared:
                without.append(d.name)
        self.assertLessEqual(
            len(without), MUTATION_CONTROL_DEBT,
            f"{len(without)} claim-bearing fixtures have no mutation control, "
            f"above the recorded debt of {MUTATION_CONTROL_DEBT}. Every NEW "
            f"fixture must carry one.\n  " + "\n  ".join(without))


if __name__ == "__main__":                                # pragma: no cover
    unittest.main()
