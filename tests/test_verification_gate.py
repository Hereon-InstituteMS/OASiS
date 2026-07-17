"""Tests for the verification-gate verdict (_stamp_verification).

Faithful to the paper's V&V model (§3): OASiS verifies via numerical checks and
*attestation* — binding every reported number to run evidence — but does not
validate, and the pre-execution critic is OPTIONAL. So the trustworthy verdict is
driven by run evidence, NOT by whether the optional critic was run. These tests
pin exactly that, plus the anti-fabrication labelling and the eval ablation.
"""
import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, SRC)

from tools.consolidated import _stamp_verification  # noqa: E402


def test_evidence_ok_is_verified():
    r = _stamp_verification({}, evidence_ok=True, critic_approved=False)
    assert r["trustworthy_result"] is True
    assert r["verification"].startswith("VERIFIED")
    # Verification is not validation — the verdict must say so.
    assert "not validation" in r["verification"].lower()


def test_no_evidence_is_not_verified_and_flagged():
    r = _stamp_verification({}, evidence_ok=False,
                            reason="no output files", critic_approved=True)
    assert r["trustworthy_result"] is False
    assert r["verification"].startswith("NOT VERIFIED")
    assert "no output files" in r["verification"]
    # Must tell the agent not to report it as a result (anti-fabrication).
    assert "must NOT be reported as a result" in r["verification"]


def test_critic_does_not_drive_trust():
    """Per the paper the critic is optional; a run backed by evidence is
    verified whether or not the critic reviewed it. Attestation, not the
    critic, is what makes a result trustworthy."""
    with_critic = _stamp_verification({}, evidence_ok=True, critic_approved=True)
    without = _stamp_verification({}, evidence_ok=True, critic_approved=False)
    assert with_critic["trustworthy_result"] == without["trustworthy_result"] is True
    # ... but the critic status is still surfaced, and it differs.
    assert with_critic["critic_review"] == "approved"
    assert "not performed" in without["critic_review"]


def test_failed_checks_stay_unverified_even_with_critic_approved():
    """A critic-approved setup that then fails its numerical checks is still
    NOT verified — the critic cannot vouch for a run that didn't pass."""
    r = _stamp_verification({}, evidence_ok=False, critic_approved=True)
    assert r["trustworthy_result"] is False


def _critic_review_under_ablation() -> str:
    code = ("import sys; sys.path.insert(0, %r);"
            "from tools.consolidated import _stamp_verification;"
            "r=_stamp_verification({}, evidence_ok=True, critic_approved=False);"
            "print(r['critic_review']); print(r['trustworthy_result'])" % SRC)
    env = dict(os.environ, OFA_DISABLE_CRITIC="1")
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


def test_ablation_only_touches_the_critic_surface_not_trust():
    """OFA_DISABLE_CRITIC (the held-out eval's ablation) disables the optional
    critic surface but must NOT change the attestation-driven verdict."""
    review, trust = _critic_review_under_ablation().splitlines()
    assert review == "disabled for evaluation"
    assert trust == "True"   # evidence-backed run is still verified
