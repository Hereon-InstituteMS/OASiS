"""A fixture whose every expectation is a bare `key=` prefix asserts no value.

WHY THIS EXISTS
---------------
A mutation control proves a fixture's output DEPENDS on the pathology. It cannot
prove the assertion measures what its name says, and two independent passes put
that failure rate at roughly 5% of fixtures. The cheapest large slice of it is
mechanical: an expectation written as a bare prefix —

    "features_on="        matches features_on=<anything>
    "max_error="          matches max_error=<anything>

matches whatever value the run produced. It cannot tell the pathological value
from the correct one, so it is satisfied identically before and after the bug is
removed.

The worst instance found, `dealii/install_feature_flags_visible`: all five of its
expectations are bare prefixes. Its own `_comment` states MPI, P4EST, PETSC,
TRILINOS and SLEPC are undefined in that install and cites itself as the evidence
for a "this cannot be verified here" finding. Re-run, it measured all five as ON
and passed. Pointed at a second install with the flags inverted, it passes too.
It is the stated evidence for a claim it cannot distinguish from its opposite.

WHAT THIS CHECKS, AND WHAT IT DOES NOT
--------------------------------------
It fails a fixture where EVERY expectation is a bare prefix — one that asserts
no value anywhere, so all of its discrimination rests on `forbid_in_output`.
That is a real property and a cheap one to see.

It deliberately does NOT fail a fixture that merely CONTAINS bare prefixes. 192
of 10847 expectations corpus-wide are bare, and most sit beside a sibling that
does pin a value; printing a path or a version alongside a real assertion is
reasonable. Failing those would bury the eleven that matter.

It also cannot catch the harder half of the class — an expectation that pins a
value which happens to be true either way. `constrained_flux_matches_prescribed`
asserting `G == G`, or a boolean computed from an exception string that is empty
exactly when the co-asserted flag holds. Only a person reading the probe finds
those. This gate takes the mechanical slice so that attention can go to the rest.

THE BASELINE
------------
Ratchets. The eleven below are recorded so the gate fails on a NEW one
immediately, while the known set is worked down. Nine of the eleven are the MMS
convergence fixtures, whose expectations are `max_error=` / `observed_order=`
style — those are also deliberately unpinned to avoid leaking measured values
into the corpus, so fixing them means asserting a BOUND rather than a number.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "scripts" / "tier2_fixtures"

# Known value-blind fixtures, recorded 2026-08-07. Expected to shrink to empty.
# Each entry is a fixture whose expectations cannot distinguish any value from
# any other, so its verdict rests entirely on forbid_in_output.
BASELINE = {
    # MMS convergence: `max_error=` / `observed_order=` are left unpinned on
    # purpose, so that measured values do not leak into the corpus and become
    # answers an evaluated agent could quote. The fix is to assert a BOUND
    # (order within a tolerance of the theoretical rate) rather than a number.
    "fenics/elasticity_mms_convergence",
    "fenics/poisson_mms_convergence",
    "fenics/stokes_mms_convergence",
    "ngsolve/elasticity_mms_convergence",
    "ngsolve/poisson_mms_convergence",
    "ngsolve/stokes_mms_convergence",
    "skfem/elasticity_mms_convergence",
    "skfem/poisson_mms_convergence",
    "skfem/stokes_mms_convergence",
    # Blames a missing gmsh package; gmsh 4.15.1 imports fine on this host and
    # dolfinx 0.10 merely renamed the submodule to dolfinx.io.gmsh. All three
    # asserted strings are labels the script prints unconditionally.
    "fenics/gmshio_install_gap_diagnostic",
}


def _value_blind() -> list[str]:
    out = []
    for fj in sorted(FIXTURES.glob("*/*/fixture.json")):
        try:
            spec = json.loads(fj.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        exp = [str(e) for e in (spec.get("expect_in_output") or [])]
        if exp and all(e.rstrip().endswith("=") for e in exp):
            out.append(f"{fj.parent.parent.name}/{fj.parent.name}")
    return out


def test_no_new_fixture_asserts_only_bare_prefixes() -> None:
    new = sorted(set(_value_blind()) - BASELINE)
    assert not new, (
        f"{len(new)} fixture(s) assert only bare `key=` prefixes, so every "
        f"expectation matches whatever value the run produced and none can "
        f"distinguish the pathology from its absence:\n    "
        + "\n    ".join(new)
        + "\n\nPin a value, or a bound on one. If the value must stay unpinned "
          "to keep measured numbers out of the corpus, assert a bound instead "
          "(e.g. the observed order within a tolerance of the theoretical "
          "rate) rather than the bare key."
    )


def test_baseline_only_shrinks() -> None:
    """A fixture that has been fixed must leave the baseline.

    Otherwise the list becomes a graveyard, and a fixture that regresses to
    value-blind is silently re-permitted by its own stale entry.
    """
    fixed = sorted(BASELINE - set(_value_blind()))
    assert not fixed, (
        "these fixtures now assert values and must be removed from BASELINE, "
        "so it cannot re-permit them later:\n    " + "\n    ".join(fixed))
