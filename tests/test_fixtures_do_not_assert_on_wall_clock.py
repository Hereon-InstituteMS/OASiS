"""A fixture must not pass or fail depending on how busy the machine is.

WHAT HAPPENED
-------------
A deal.II fixture asserted `assembly_fraction_in_claimed_60_to_80_band=false`.
On a quiet host, assembly was 32% of the loop, so the catalog's claimed band did
not reproduce and the author correctly pinned that. Under load a later run
landed INSIDE the band and the fixture went red — not because anything changed
in deal.II, but because other worktrees were compiling at the time.

That is a pinned measurement wearing a boolean's clothes. It looks like a
property of the software and is actually a property of the afternoon.

WHY IT MATTERS MORE THAN ORDINARY FLAKINESS
-------------------------------------------
This project's whole claim is that a green fixture is evidence a knowledge claim
holds. A fixture that goes red under load teaches the opposite lesson twice
over: first someone re-runs it and it passes, so red stops meaning "the claim
failed"; then the next genuine failure gets re-run too. The value of the suite
is that its verdicts are trustworthy, and one flaky verdict is worth more than
one fixture of damage.

WHAT IS AND IS NOT ALLOWED
--------------------------
Timing may be MEASURED and PRINTED — it is often the whole point of a
performance pitfall, and a reader wants the number. What it may not do is decide
the verdict. `expect_in_output` is the verdict, so a wall-clock comparison must
not appear there.

The distinction is between a property of the algorithm and a property of the
run. "The block solver touches 4x the non-zeros" is structural and holds on any
host. "The block solver is 1.3x slower" is not: it depends on cache, on core
count, on what else is running. Assert the first; print the second.

Measured when this was written: 16 such assertions across fenics (7), dealii
(4), ngsolve (4) and skfem (1) — including a contact-search fixture asserting
which of two algorithms wins at a few hundred facets, which is precisely the
regime where the answer flips between machines.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "scripts" / "tier2_fixtures"

# Verdict keys whose truth is decided by elapsed time. Deliberately narrow: an
# over-eager pattern here would flag physics claims that merely contain the word
# "time" (second-ORDER in time, time-step size, Crank-Nicolson), and a checker
# that cries wolf gets switched off. A first draft matched 133 expectations of
# which 117 were exactly that kind of false positive.
_WALL_CLOCK = re.compile(
    r"speedup|slower|faster|wall_?clock|elapsed"
    r"|_secs?=|_ms=|assembly_fraction|time_ratio|throughput|per_second",
    re.IGNORECASE)

# Things the pattern above would otherwise catch that are NOT about elapsed
# time: convergence order in time, time-step choices, and a timing key merely
# being present in a returned dict.
_NOT_WALL_CLOCK = re.compile(
    r"second[_ ]order|second_third|time[_ ]step|timestep|time_dependent"
    r"|crank|timing'|'timing",
    re.IGNORECASE)

BACKENDS = sorted(p.name for p in FIXTURES.iterdir() if p.is_dir()) \
    if FIXTURES.is_dir() else []


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_expectation_is_decided_by_elapsed_time(backend):
    """Timing may be printed; it may not decide pass or fail."""
    fx = FIXTURES / backend
    if not fx.is_dir():
        pytest.skip(f"{backend} has no fixtures")

    offenders = []
    for d in sorted(fx.iterdir()):
        manifest = d / "fixture.json"
        if not manifest.is_file():
            continue
        try:
            spec = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for exp in (spec.get("expect_in_output") or []):
            text = str(exp)
            if _WALL_CLOCK.search(text) and not _NOT_WALL_CLOCK.search(text):
                offenders.append(f"{d.name}: {text[:80]}")

    assert not offenders, (
        f"{backend}: {len(offenders)} expectations decide the verdict from "
        f"elapsed time, so the fixture's colour depends on what else the "
        f"machine is doing:\n  " + "\n  ".join(offenders[:15])
        + (f"\n  ... and {len(offenders) - 15} more"
           if len(offenders) > 15 else "")
        + "\n\nKeep measuring and printing the timing — a performance pitfall "
          "usually needs it, and a reader wants the number. Move the VERDICT "
          "onto something structural that holds on any host: operation counts, "
          "non-zero counts, iteration counts, memory footprint, or the "
          "asymptotic trend across refinements rather than a ratio at one "
          "size. A fixture that goes red under load teaches everyone that red "
          "means 're-run it', which costs more than the fixture is worth.")
