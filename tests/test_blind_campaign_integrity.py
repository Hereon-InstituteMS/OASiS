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
    state = keyvault.seal_state(KEYS)
    if state in ("SEALED", "ABSENT", "EMPTY"):
        pytest.skip(f"keys are {state} — cannot enumerate (which is the point)")
    # rglob over an unreadable subtree yields NOTHING, silently. In a PARTIAL
    # tree — top listable, subdirectories already 000, which is what a
    # concurrent seal looks like — this test would find no plaintext keys and
    # report PASS while every key on disk was plaintext. A custody control that
    # passes on zero inputs is worse than no control.
    if state == "PARTIAL":
        pytest.skip("keys are PARTIAL (a seal is in progress): rglob would "
                    "silently see nothing and this test would pass vacuously")
    reachable = [d for d in KEYS.iterdir() if d.is_dir()]
    assert reachable, "no key directories are reachable; nothing was checked"
    plain = [p for p in KEYS.rglob("*.json") if not p.name.endswith(".enc")]
    assert not plain, (
        f"plaintext answer keys on disk: {[str(p) for p in plain]}.\n"
        f"This is the state a freshly BUILT problem set is in, and the campaign "
        f"must not run from it. The lifecycle is: build -> "
        f"scripts/blind_keys.py commit -> scripts/blind_keys.py encrypt (types "
        f"the passphrase, which touches no disk) -> scripts/blind_keys.py seal. "
        f"run_blind.py's preflight refuses to start while this is true, so the "
        f"failure is the control working, not a broken test.")


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
    probs = [p for p in sorted(PROBLEMS.iterdir()) if (p / "task.txt").is_file()]
    n_problems = len(probs)
    if n_problems and not any((KEYS / p.name / "key.json").is_file()
                              for p in probs):
        pytest.skip("keys are encrypted — run the gate audit before "
                    "scripts/blind_keys.py encrypt, or decrypt to audit")

    # A campaign is routinely MIXED: rebuilding a subset leaves those keys in
    # plaintext while the rest stay encrypted. Coverage must still be asserted,
    # so every problem has to be accounted for as either audited or encrypted —
    # silently auditing a subset and reporting PASS is the failure mode this
    # assertion exists to prevent.
    # A concurrent `seal` leaves the top directory listable while the
    # per-problem subdirectories are already 000. Reading a key then raises
    # PermissionError, which used to surface as a FAILING LEAK-GATE TEST — a
    # gate failing on a race rather than on a finding, which is the worst
    # possible way for a leak gate to be wrong. Unreachable keys are counted,
    # not walked into.
    def _state(name):
        try:
            if (KEYS / name / "key.json").is_file():
                return "plain"
            if (KEYS / name / "key.json.enc").is_file():
                return "encrypted"
            return "missing"
        except PermissionError:
            return "unreadable"

    leaking, audited, encrypted, unreadable = [], [], [], []
    for pdir in probs:
        st = _state(pdir.name)
        if st == "unreadable":
            unreadable.append(pdir.name)
            continue
        if st != "plain":
            if st == "encrypted":
                encrypted.append(pdir.name)
            continue
        kpath = KEYS / pdir.name / "key.json"
        audited.append(pdir.name)
        rep = scan((pdir / "task.txt").read_text(),
                   json.loads(kpath.read_text()), pdir.name)
        worst = [f for f in rep.findings
                 if f.severity in ("CRITICAL", "HIGH", "MEDIUM")]
        if worst:
            leaking.append((pdir.name, [f.rule for f in worst]))
    assert not leaking, f"problems disclose their solution: {leaking}"
    if unreadable:
        pytest.skip(
            f"{len(unreadable)} key(s) became unreachable while the audit ran "
            f"(seal_state={keyvault.seal_state(KEYS)}): {unreadable[:4]}. This "
            f"is a concurrent seal, not a leak. Re-run with the keys unsealed "
            f"and nothing else touching them.")
    assert len(audited) + len(encrypted) == n_problems, (
        f"gate audited {audited} and found {encrypted} encrypted, but there "
        f"are {n_problems} problems — the remainder have no key at all, so "
        f"this test proved nothing about them")
    assert audited, (
        f"no problem had a readable key; all {len(encrypted)} are encrypted, "
        f"so the gate checked nothing")


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


@needs_campaign
def test_builder_sources_are_not_readable_during_runs():
    """The builders hold every hidden field as a literal.

    They sit two levels above the agent's working directory
    (``runs/<cell>/work``), and the agent has bash plus an absolute-path
    ``read_file``. ``cat ../../build_extra.py`` hands over eight of the eleven
    solutions, so sealing ``keys/`` while leaving these readable is custody
    theatre. DESIGN.md noted the exposure without acting on it.
    """
    if not keyvault.is_sealed(KEYS):
        pytest.skip("campaign is unsealed (grading/audit state)")
    readable = [p.name for p in (
        CAMPAIGN / "build_problems.py", CAMPAIGN / "build_extra.py",
        CAMPAIGN / "build_coupled.py") if p.exists() and os.access(p, os.R_OK)]
    assert not readable, (
        f"solution-bearing builder sources readable during a run: {readable}")


@needs_campaign
def test_no_agent_readable_file_carries_a_derivation_source():
    """Structural sweep: string matching alone would miss these.

    A builder writes ``x * (1 - x) * y * (1 - y) * sp.cos(...)`` while the key
    stores ``x*y*(1-x)*(1-y)*cos(2*pi*x)`` -- different bytes, same function --
    so a literal search over the tree finds nothing and reports safety.
    """
    if not keyvault.is_sealed(KEYS):
        pytest.skip("campaign is unsealed (grading/audit state)")
    markers = ("def problem_", "def coupled_", "diffusion_source",
               "elasticity_source", "_elastic_body_force")
    exposed = []
    for p in CAMPAIGN.rglob("*.py"):
        try:
            text = p.read_text(errors="ignore")
        except (PermissionError, OSError):
            continue                     # unreadable is the desired state
        if "sympy" in text and any(m in text for m in markers):
            exposed.append(str(p))
    assert not exposed, f"agent-readable derivation sources: {exposed}"


# ── the builder holds no answer only if the SEED is not in the repo ───
def test_no_live_draw_seed_appears_in_any_tracked_file():
    """A seed in git re-derives every hidden field, whatever the builder holds.

    The coupled builder was made solution-free so it could be version-controlled:
    the fields are drawn from a CSPRNG and only the seed reaches the sealed key.
    That property is destroyed the moment the seed itself is committed — and it
    was, in a verification script, which is how this test came to exist. The
    check is here rather than in review because a review that has to remember
    something is not a control.
    """
    import json as _json
    import subprocess as _sp
    from pathlib import Path as _P
    repo = _P(__file__).resolve().parents[1]
    keys = _P("/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys")
    if not keys.is_dir():
        pytest.skip("no key directory on this machine")
    seeds = []
    for kp in sorted(keys.rglob("key.json")):
        try:
            s = _json.loads(kp.read_text()).get("draw_seed")
        except (PermissionError, ValueError):
            continue
        if s is not None:
            seeds.append(str(s))
    if not seeds:
        pytest.skip("keys are sealed or encrypted; cannot check")
    tracked = _sp.run(["git", "ls-files"], cwd=repo, capture_output=True,
                      text=True).stdout.split()
    hits = []
    for rel in tracked:
        p = repo / rel
        if not p.is_file() or p.stat().st_size > 4_000_000:
            continue
        try:
            txt = p.read_text(errors="ignore")
        except OSError:
            continue
        for s in seeds:
            if s in txt:
                hits.append((rel, s))
    assert not hits, (
        f"a live draw seed appears in version-controlled file(s) {hits}: "
        f"anyone with the repository can re-derive the hidden fields by "
        f"re-running the builder, which is exactly what the solution-free "
        f"builder was for")
