"""A fixture must ship the proof that it detects its own pathology.

WHAT A FIXTURE IS, AND WHY THIS IS THE WHOLE GAME
------------------------------------------------
A fixture is two files:

    fixture.json   backend, physics, pitfall_index (the claim it defends),
                   expect_in_output (strings the runner greps stdout for),
                   forbid_in_output (strings that must NOT appear, e.g.
                   "Traceback"), and _comment, the evidence record
    source.py      the executable — or cmd.sh for 4C, source.cpp + cmd.sh
                   for deal.II, plus any data files the deck needs

The runner executes the artefact and matches its stdout against those two
lists. That is the entire mechanism. Nothing in it can tell a fixture that
prints the right strings for the RIGHT reason from one that prints them for the
wrong reason — a script that hardcodes its expected output passes exactly as
well as one that measures.

The mutation control is what closes that hole: remove the pathology, and the
fixture must go red. Without it a fixture is a script that happens to pass.

THE STATE THAT MOTIVATED THIS, measured across all worktrees:

    fourc      354/373  95%   `_mutation` field in fixture.json
    fenics     183/200  92%   T2_MUTATE env hook inside the source
    dealii     104/119  87%   T2_MUTATE env hook
    coupling    18/18  100%   `_mutation` field
    skfem, kratos, ngsolve, febio, dune, sparta   0%   — 542 fixtures

Every one of those campaigns reported its fixtures mutation-proven, and their
reports name the specific mutation and the expectation it removed. The runs
almost certainly happened. But the evidence lived only in session transcripts,
so nobody could re-run it — which in this project is the same class of problem
as an unverified knowledge claim: probably true, and unauditable.

TWO CONVENTIONS, BOTH ACCEPTED
------------------------------
`_mutation` in fixture.json describes an edit somebody must reapply by hand.
A `T2_MUTATE` environment hook in the source IS the edit, already written and
already tested, so the proof re-runs with one command:

    T2_MUTATE=1 python scripts/tier2_fixtures/<backend>/<name>/source.py

The hook is the better of the two and was not imposed — FEniCSx and deal.II
arrived at it independently. It matters because three separate staging defects
silently voided mutation verdicts in a single day: a shared helper written as a
bare `_couplinglib.py` that the harness never staged, a missing
`_lib/preamble.sh` that made 174 fixtures report vacuous baselines, and a
`parents[N]` path resolution that could not find the checkout from /tmp. An
in-source hook has none of those failure modes.

WHAT THIS GATE CANNOT DO, stated plainly
----------------------------------------
It checks that a control EXISTS, not that it discriminates. It cannot, because
whether a mutation changes the verdict is a property of a run.

And no gate catches the worst case. `stokes_lbb_and_pressure_constraints` (DUNE)
PASSED while asserting a false statement: its inlet mask interpolated
`[x[0], x[1], 0]` and sliced the pressure leg, returning the constant-0 third
component, so every pressure dof tested as lying on the boundary. Coverage
counted it. Mutation evidence would NOT have caught it either — the mutation
changes the assertion either way, so an internally consistent, externally false
fixture is still killed by its own mutant. Only a person re-reading the probe
caught it. Pretending a gate could is the same error one level up.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "scripts" / "tier2_fixtures"

# Sources where an in-source hook can live.
_SOURCE_SUFFIXES = (".py", ".sh", ".cpp", ".cc", ".c")

BACKENDS = sorted(p.name for p in FIXTURES.iterdir() if p.is_dir()) \
    if FIXTURES.is_dir() else []


def _has_control(d: Path) -> bool:
    """Either convention counts: a `_mutation` field or a T2_MUTATE hook."""
    manifest = d / "fixture.json"
    try:
        spec = json.loads(manifest.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if spec.get("_mutation") or spec.get("mutations"):
        return True
    for f in d.iterdir():
        if f.is_file() and f.suffix in _SOURCE_SUFFIXES:
            try:
                if "T2_MUTATE" in f.read_text(errors="ignore"):
                    return True
            except OSError:
                continue
    return False


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_fixture_ships_its_mutation_control(backend):
    """Without it, a fixture is a script that happens to pass."""
    be = FIXTURES / backend
    if not be.is_dir():
        pytest.skip(f"{backend} has no fixtures")

    fixtures = [d for d in sorted(be.iterdir())
                if (d / "fixture.json").is_file()]
    if not fixtures:
        pytest.skip(f"{backend} has no fixtures in this tree")

    missing = [d.name for d in fixtures if not _has_control(d)]

    assert not missing, (
        f"{backend}: {len(missing)} of {len(fixtures)} fixtures ship no "
        f"mutation control, so nothing in the tree shows they detect the "
        f"pathology they claim:\n  " + "\n  ".join(missing[:15])
        + (f"\n  ... and {len(missing) - 15} more" if len(missing) > 15 else "")
        + "\n\nAdd a T2_MUTATE branch to the source — the pathology removed "
          "when the variable is set — so the proof re-runs with\n"
          "    T2_MUTATE=1 python scripts/tier2_fixtures/<backend>/<name>/source.py\n"
          "and record in `_comment` WHICH expectation disappears under it. "
          "A `_mutation` field in fixture.json is also accepted.\n\n"
          "If a fixture genuinely cannot be mutated, say so in `_mutation` "
          "with the reason — DUNE's helmholtz_indefinite_no_diagnostic is the "
          "honest example: its claim is that the solver's report is "
          "UNINFORMATIVE, which is true at every k, so removing the pathology "
          "would require the library to emit a diagnostic it does not have.")
