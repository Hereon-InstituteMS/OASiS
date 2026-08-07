"""Standing guards on the blind campaign's custody and blindness.

These check the campaign directory as it sits on disk, so a problem set that
drifts back into disclosing its answers, or a key that gets left in plaintext,
fails here rather than in review.

Two groups, with different preconditions:

  * custody checks run in the campaign's NORMAL state (keys sealed) and are
    always meaningful;
  * the leak-gate sweep needs to read the keys, so it is a pre-campaign audit
    step and skips when the keys are correctly sealed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from blind_eval import keyvault
from blind_eval.leakgate import scan

CAMPAIGN = Path("/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind")
KEYS = CAMPAIGN / "keys"
PROBLEMS = CAMPAIGN / "problems"
MANIFEST = Path(__file__).resolve().parents[1] / "data" / "blind_key_commitment.json"

needs_campaign = pytest.mark.skipif(
    not PROBLEMS.is_dir(), reason="campaign3_blind not present")


# ── custody ───────────────────────────────────────────────────────────
@needs_campaign
def test_hash_commitment_is_published_and_self_consistent():
    """The commitment must exist and not have been edited after the fact."""
    assert MANIFEST.is_file(), (
        "no key commitment in the repo. Publish one BEFORE the campaign: "
        "scripts/blind_keys.py commit")
    man = json.loads(MANIFEST.read_text())
    recomputed = hashlib.sha256(
        json.dumps(man["entries"], sort_keys=True).encode()).hexdigest()
    assert recomputed == man["manifest_sha256"], "commitment has been tampered with"
    assert man["entries"], "commitment covers no key files"
    assert man.get("generated_utc")


@needs_campaign
def test_no_plaintext_keys_are_left_on_disk():
    """Encryption is only a control if the plaintext is actually gone."""
    if keyvault.is_sealed(KEYS):
        pytest.skip("keys sealed — cannot enumerate (which is the point)")
    plain = [p for p in KEYS.rglob("*.json") if not p.name.endswith(".enc")]
    assert not plain, f"plaintext answer keys on disk: {[str(p) for p in plain]}"


@needs_campaign
def test_encrypted_keys_do_not_contain_the_answer_in_clear():
    if keyvault.is_sealed(KEYS):
        pytest.skip("keys sealed")
    for p in KEYS.rglob("*.enc"):
        blob = p.read_bytes()
        assert blob.startswith(keyvault.MAGIC)
        for marker in (b"exact_solution", b"sin(", b"exp(", b"pi**2"):
            assert marker not in blob, f"{p} leaks {marker!r} in clear"


def test_absence_of_keys_is_never_reported_as_sealed(tmp_path):
    """Regression guard for the pre-existing runner's check.

    ``run_blind.py``'s ``keys_are_sealed()`` returns True when the directory is
    missing or empty, so a deleted keys tree reads as sealed and the campaign
    starts with nothing to grade against.
    """
    assert keyvault.is_sealed(tmp_path / "absent") is False
    (tmp_path / "empty").mkdir()
    assert keyvault.is_sealed(tmp_path / "empty") is False


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores DAC permissions")
@needs_campaign
def test_seal_is_verified_by_execution_not_by_assertion():
    if not keyvault.is_sealed(KEYS):
        pytest.skip("keys are currently unsealed (grading/audit state)")
    v = keyvault.verify_unreadable(KEYS)
    assert v["sealed"] is True, v
    assert v["files_opened"] == 0
    assert "exact_solution" not in v.get("shell_cat", "")


# ── blindness of the problem set ──────────────────────────────────────
@needs_campaign
def test_every_problem_passes_the_leak_gate():
    """Pre-campaign audit: run it after building the problems, before encrypting.

    Coverage is asserted, not assumed.  Without the final assertion this test
    passes while checking nothing the moment the keys are encrypted or sealed —
    it finds no ``key.json``, iterates over an empty set, and reports success.
    A gate that reports PASS on zero inputs is worse than no gate, because it
    is believed.
    """
    if keyvault.is_sealed(KEYS):
        pytest.skip("keys sealed — run this as a pre-campaign audit, unsealed")
    n_problems = sum(1 for p in PROBLEMS.iterdir()
                     if (p / "task.txt").is_file())
    if n_problems and not any((KEYS / p.name / "key.json").is_file()
                              for p in PROBLEMS.iterdir()):
        pytest.skip("keys are encrypted — run the gate audit before "
                    "scripts/blind_keys.py encrypt, or decrypt to audit")

    leaking, checked = [], 0
    for pdir in sorted(PROBLEMS.iterdir()):
        task = pdir / "task.txt"
        kpath = KEYS / pdir.name / "key.json"
        if not (task.is_file() and kpath.is_file()):
            continue
        checked += 1
        rep = scan(task.read_text(), json.loads(kpath.read_text()), pdir.name)
        worst = [f for f in rep.findings
                 if f.severity in ("CRITICAL", "HIGH", "MEDIUM")]
        if worst:
            leaking.append((pdir.name, [f.rule for f in worst]))
    assert not leaking, f"problems disclose their solution: {leaking}"
    assert checked == n_problems, (
        f"gate audited {checked} of {n_problems} problems — the rest have no "
        f"readable key, so this test proved nothing about them")
    assert checked > 0


@needs_campaign
def test_task_texts_never_name_an_exact_solution():
    """Cheap lexical guard — runs with the keys sealed."""
    banned = ("exact solution", "u_exact", "manufactured", "analytical solution",
              "closed form solution", "reference solution")
    bad = []
    for task in sorted(PROBLEMS.rglob("task.txt")):
        low = task.read_text(errors="ignore").lower()
        for b in banned:
            if b in low:
                bad.append((task.parent.name, b))
    assert not bad, f"task text names a solution: {bad}"
