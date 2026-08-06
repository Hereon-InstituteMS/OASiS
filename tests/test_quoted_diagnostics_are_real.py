"""A quoted error message must be one the software actually prints.

WHY THIS IS THE MOST IMPORTANT GATE IN THE SUITE
------------------------------------------------
Every verification pass that EXECUTED knowledge rather than reading it came back
with the same largest category of defect, on every backend:

    FEBio    25 invented error messages, 11 non-existent material names
    4C       a majority of quoted diagnostics not found in the source tree
             (see ON PRECISION below before quoting any rate — the first
             version of this screen reported 91% and the figure was wrong)
    NGSolve  "Newton did not converge after N iterations" — nowhere in
             ngsolve/netgen; the real text is "Warning: Newton might not
             converge! Error = "
    skfem    meshio "Cannot write complex array" — never emitted; the real
             failure is KeyError: dtype('complex128')
    Kratos   FREESTREAM_VELOCITY / MACH_INFINITY — neither name exists
    scipy    cg "raises RuntimeError: matrix not positive definite" — scipy
             raises nothing; it returns info=1000 and a garbage vector

A fabricated diagnostic is worse than a missing entry, because an agent uses the
quoted string to BUILD ITS GUARD. Told that cg raises, it writes an exception
handler and reads the absence of an exception as success — the guard passes and
the wrong answer flows through, wearing a checkmark. The knowledge does not
merely fail to help; it manufactures false confidence, which is the exact
failure this project exists to prevent.

`audit_phantom_apis.py` checks API ATTRIBUTES via hasattr. It says nothing about
message text, which is where the damage has actually been.

HOW THIS AVOIDS CRYING WOLF
---------------------------
Two earlier guards here were too eager and had to be retracted — a delegation
guard flagged `sorted`, `dedent` and `format` (about 20 false accusations), and
a disclosure guard failed five honest 4C entries over a phrasing difference. An
ignored gate protects nothing, so this one is deliberately conservative, and
each rule below exists because the naive version produced a false accusation on
the real corpus:

  * dependency trees are searched too, since genuine messages come from
    Trilinos, PETSc, UMFPACK, meshio and scipy as well as from the backend;
  * matching is on the longest STATIC fragment, because real messages
    interpolate at runtime;
  * quotes are paired left to right — an alternation regex straddled from one
    fragment's closing quote to the next one's opening quote and "found" prose;
  * RETRACTIONS ARE EXEMPT. When a pass falsifies a string, the honest fix
    records what the entry used to claim. Flagging that would punish exactly
    the behaviour we want;
  * a backend whose source cannot be found reports UNKNOWN, never ABSENT —
    "could not verify" and "verified absent" are different claims and only one
    is an accusation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import audit_quoted_diagnostics as audit  # noqa: E402

# Backends whose implementation this machine can actually search. Others are
# reported by the script as UNKNOWN and are not asserted on here.
CHECKABLE = ["fourc", "ngsolve", "skfem", "fenics", "kratos"]


@pytest.mark.parametrize("backend", CHECKABLE)
def test_quoted_diagnostics_exist_in_the_software(backend):
    """No entry may quote a diagnostic the software cannot emit."""
    result = audit.audit_backend(backend)

    if result["verdict"].startswith("UNKNOWN"):
        pytest.skip(
            f"{backend}: no searchable source on this machine "
            f"({', '.join(result['roots_missing']) or 'no roots configured'}). "
            f"That is 'not checked', NOT 'clean' — this backend needs the audit "
            f"run where its source is available.")

    absent = result["fragments_absent"]
    assert not absent, (
        f"{backend}: {len(absent)} quoted diagnostics could not be found in the "
        f"software that is supposed to emit them, out of "
        f"{result['fragments_present'] + len(absent)} checked. These are "
        f"CANDIDATES TO ADJUDICATE, not proven fabrications — see the note on "
        f"precision below.\n\n"
        + "\n".join(f"  {a['file']}\n      {a['fragment']}"
                    for a in absent[:25])
        + (f"\n  ... and {len(absent) - 25} more" if len(absent) > 25 else "")
        + "\n\nRun the software, capture what it ACTUALLY prints, and quote "
          "that. If the message turns out not to exist, say so in the entry — "
          "a documented retraction is exempt from this gate and is the right "
          "way to record a falsified claim.\n\n"
          "ON PRECISION, because this list must not be pasted into a report as "
          "a fabrication count. A hand adjudication of 57 NGSolve and skfem "
          "strings — grepping the sources AND the vendored .so files, then "
          "triggering the ones in doubt live — found 36 present, 13 already "
          "documented as absent, 5 genuinely fabricated, 2 paraphrases and 1 "
          "uncheckable. This screen flagged far more than 5. It catches real "
          "fabrications (the ProxyFunction 'neighbour' message was one), but it "
          "also flags messages from dependencies it cannot see, constants like "
          "PETSc's DIVERGED_* that live outside the searched trees, and text "
          "the extractor mangled. Treat every entry here as a lead to check by "
          "running the software, never as a verdict.")


def test_the_auditor_pairs_quotes_correctly():
    """Regression: the extractor must not straddle adjacent quoted fragments.

    The first version used an alternation regex with a minimum length inside
    the match. On the real entry

        writing 'SOLID QUAD4' for an ALE mesh raises
        'expected ALE element type' from 4C_ale_factory.cpp

    `SOLID QUAD4` is 11 characters, one below the minimum, so the regex skipped
    it and matched from that fragment's CLOSING quote to the next one's OPENING
    quote — yielding the prose "for an ALE mesh raises". Two failures in one:
    a fabricated accusation against prose, and the real assertion never checked.
    """
    text = ("writing 'SOLID QUAD4' for an ALE mesh raises "
            "'expected ALE element type' from 4C_ale_factory.cpp")
    assert audit.quoted_fragments(text) == ["expected ALE element type"]


def test_retractions_are_not_reported_as_fabrications():
    """A documented falsification must not be punished as a fabrication.

    Both directions matter, and both were wrong in the first implementation:
    a plain backward window also excused the REAL message that followed the
    retraction (the one string that must be checked), and a backward-only
    window missed cues that follow the quote.
    """
    both = ("older quote 'beam element type not supported in Exodus' is in no "
            "4C source file; the real message is "
            "'The cell data does not contain the key'")
    assert audit.quoted_fragments(both) == [
        "The cell data does not contain the key"]

    assert audit.quoted_fragments(
        "'zero pivot in Schur complement' does not exist in the source") == []
    assert audit.quoted_fragments(
        "the claimed 'no design nodes found' never appears") == []

    # And a plain assertion is still caught.
    assert audit.quoted_fragments(
        "Signal: the solver prints 'expected ALE element type' and aborts"
    ) == ["expected ALE element type"]


def test_auditor_reports_unknown_rather_than_clean_without_a_source():
    """Absence of evidence must not be reported as evidence of absence.

    If a backend's source cannot be found, every quoted string would trivially
    "not be found" in it. Reporting that as ABSENT would flood the gate with
    false accusations on any machine lacking a source tree, and the predictable
    response is to switch the gate off.
    """
    result = audit.audit_backend("a-backend-that-does-not-exist")
    assert result["verdict"].startswith("UNKNOWN")
    assert not result["fragments_absent"]
