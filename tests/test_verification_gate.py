"""Tests for the verification-gate verdict (_stamp_verification).

OASiS enforces verification IN SOFTWARE: it verifies via numerical checks on the
RUN (completed, produced output, finite / converged / balanced) but does not
validate. It does NOT currently bind a reported NUMBER to that run — an audit
showed an invented value attached to a real run still passes — so these tests
pin what the gate actually does, not what it was once described as doing.

The independent pre-execution critic is MANDATORY and UNCONDITIONAL: a result is
trustworthy ONLY when it passes the numerical checks AND a critic review of that
exact setup is on the server's record. `critic_approved=True` is a self-report
and verifies nothing by itself; there is no environment switch that lifts the
requirement. Enforced by verdict, never by error. These tests pin exactly that,
plus the anti-fabrication labelling.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, SRC)

from tools.consolidated import _stamp_verification  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_critic_registry(monkeypatch):
    """Each test gets its own review record.

    Without this, a review registered by one test verifies a LATER test's run:
    the registry is module-level, and several tests here share the same solver
    and the same one-line deck, so their digests collide. That was observed —
    an unreviewed-run test came back "reviewed (submitted review matches this
    setup)" and passed for entirely the wrong reason. Shared critic state
    manufactures exactly the false green this whole gate exists to prevent.
    """
    import tools.consolidated as _C
    from core.critic_gate import CriticRegistry
    monkeypatch.setattr(_C, "_CRITIC_REGISTRY", CriticRegistry())


def _review_on_record(solver: str, setup_text: str) -> None:
    """Register a critic review the way an agent must.

    The gate resolves the critic from the server's own record, not from the
    caller's `critic_approved` flag, so a test that wants a VERIFIED verdict
    has to put a review there first. Tests that skip this step are asserting
    the behaviour of an unreviewed run, which is the interesting case anyway.
    """
    from core.critic_gate import setup_digest
    from tools.consolidated import _CRITIC_REGISTRY
    _CRITIC_REGISTRY.submit_review(
        solver=solver, digest=setup_digest(solver, setup_text),
        findings="Checked the problem statement, boundary conditions, units "
                 "and discretisation against the stated physics; no issues "
                 "found that would invalidate the run.")


def test_evidence_and_critic_is_verified():
    # A verified result needs BOTH run evidence and the mandatory critic.
    _review_on_record("skfem", "deck")
    r = _stamp_verification({}, evidence_ok=True, critic_approved=True,
                            solver="skfem", setup_text="deck")
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


def test_critic_is_mandatory_for_trust():
    """The whole point of OASiS: verification is enforced in software. A run that
    passes every automated check but was NOT reviewed by the mandatory critic is
    still NOT verified — enforced by verdict, not by an error."""
    _review_on_record("skfem", "mandatory-critic-deck")
    with_critic = _stamp_verification({}, evidence_ok=True, critic_approved=True,
                                      solver="skfem",
                                      setup_text="mandatory-critic-deck")
    without = _stamp_verification({}, evidence_ok=True, critic_approved=False,
                                  solver="skfem", setup_text="unreviewed-deck")
    assert with_critic["trustworthy_result"] is True
    assert without["trustworthy_result"] is False        # critic is MANDATORY
    assert without["verification"].startswith("NOT VERIFIED")
    assert "MANDATORY" in without["verification"]
    # The note names WHERE the approval came from — a server-side record, not
    # the caller's assertion.
    assert "reviewed" in with_critic["critic_review"]
    # Two distinct ways to be unreviewed, and the note must tell them apart:
    # nothing reviewed for this solver at all …
    never = _stamp_verification({}, evidence_ok=True, critic_approved=False,
                                solver="dealii", setup_text="anything")
    assert never["trustworthy_result"] is False
    assert "no critic review is on record" in never["critic_review"]
    # … versus a review that exists but was for a different setup, which is the
    # review-a-clean-deck-then-run-another-one route.
    assert "changed after it was reviewed" in without["critic_review"]


def test_failed_checks_stay_unverified_even_with_critic_approved():
    """A critic-approved setup that then fails its numerical checks is still
    NOT verified — the critic cannot vouch for a run that didn't pass."""
    r = _stamp_verification({}, evidence_ok=False, critic_approved=True)
    assert r["trustworthy_result"] is False


def _trust_under_env(**extra_env) -> str:
    """What verdict does an evidence-backed but unreviewed run get, under an
    environment an evaluation harness might plausibly be running with?"""
    code = ("import sys; sys.path.insert(0, %r);"
            "from tools.consolidated import _stamp_verification;"
            "r=_stamp_verification({}, evidence_ok=True, critic_approved=True,"
            "                      solver='skfem', setup_text='deck');"
            "print(r['trustworthy_result'])" % SRC)
    env = dict(os.environ, **extra_env)
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip()


def test_no_environment_variable_lifts_the_mandatory_critic_requirement():
    """This test used to assert the OPPOSITE.

    OFA_DISABLE_CRITIC was an ablation switch that lifted the mandatory-critic
    requirement, so an unreviewed run came back VERIFIED. A mandatory gate with
    an environment-variable off-switch is not mandatory: a stray export, a
    harness default, or a copied shell script silently converts every verdict in
    a campaign, and nothing in the output says so. The switch is removed, and
    the ablation arm it existed for is not run — the design is OASiS or no
    OASiS. What is asserted here is that setting it now changes nothing.
    """
    assert _trust_under_env(OFA_DISABLE_CRITIC="1") == "False"
    assert _trust_under_env() == "False"


def test_a_tool_that_does_not_identify_its_setup_fails_closed():
    """A caller that cannot say what it ran cannot have had that thing
    reviewed, so the gate must refuse rather than wave it through."""
    r = _stamp_verification({}, evidence_ok=True, critic_approved=True)
    assert r["trustworthy_result"] is False
    assert "did not identify its setup" in r["critic_review"]


# ── Finiteness scan (attestation's numeric side) ─────────────────────────
import numpy as np          # noqa: E402
import meshio               # noqa: E402
from core.quality_checks import check_result_files_finite  # noqa: E402


def _write_vtu(path, values):
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], float)
    cells = [("triangle", np.array([[0, 1, 2], [1, 3, 2]]))]
    meshio.write_points_cells(str(path), pts, cells,
                              point_data={"u": np.asarray(values, float)})


def test_finite_output_passes_scan(tmp_path):
    f = tmp_path / "ok.vtu"; _write_vtu(f, [1.0, 2.0, 3.0, 4.0])
    assert check_result_files_finite([f]) == []


def test_nan_inf_output_is_flagged(tmp_path):
    f = tmp_path / "bad.vtu"; _write_vtu(f, [1.0, np.nan, np.inf, 4.0])
    w = check_result_files_finite([f])
    assert w and "non-finite" in w[0]


def test_scan_never_raises_on_garbage(tmp_path):
    bad = tmp_path / "junk.vtu"; bad.write_text("not a real vtu at all")
    # A best-effort scan must NEVER raise (incl. SystemExit). A file with a
    # SCANNABLE suffix that fails to parse is a CORRUPT result file (stress
    # audit F1): a hard, verdict-flipping finding — never a false NaN finding
    # and never a crash.
    w = check_result_files_finite([bad])
    assert all("non-finite" not in x for x in w)          # no false NaN finding
    assert any("unreadable/corrupt" in x for x in w)      # hard corrupt finding


def test_unscannable_format_keeps_honesty_note_only(tmp_path):
    # A genuinely UNSCANNABLE format (e.g. FEBio .xplt) is a coverage gap, not
    # corruption: only the informational honesty note fires, no hard finding —
    # so a legitimate FEBio run is not wrongly flipped to NOT VERIFIED.
    xplt = tmp_path / "result.xplt"; xplt.write_bytes(b"\x00\x01\x02")
    w = check_result_files_finite([xplt])
    assert any("not asserted" in x for x in w)            # honesty note
    assert all("unreadable/corrupt" not in x for x in w)  # no hard finding


# ── Headline-number scan: results_summary.json + stdout (P2) ─────────────
from core.quality_checks import check_summary_finite  # noqa: E402


def test_summary_json_infinity_is_flagged(tmp_path):
    # json.loads parses bare Infinity/NaN to floats; the walk must catch them.
    (tmp_path / "results_summary.json").write_text('{"max_value": Infinity, "mean": 0.5}')
    w = check_summary_finite(tmp_path)
    assert w and any("max_value" in x and "non-finite" in x for x in w)


def test_summary_clean_passes(tmp_path):
    (tmp_path / "results_summary.json").write_text('{"max_value": 0.0737, "mean": 0.03}')
    assert check_summary_finite(tmp_path, "solve ok, residual 1e-9") == []


def test_stdout_nonfinite_result_flagged(tmp_path):
    assert check_summary_finite(tmp_path, "max(u) = inf\nmean = nan")


def test_stdout_prose_nan_does_not_false_trigger(tmp_path):
    # 'information'/'nanometer'/a bare word must NOT trigger a false downgrade;
    # only a NaN/Inf presented as a numeric result counts.
    assert check_summary_finite(tmp_path, "computed 42 nanometers of information") == []
    # ... including prose around the anchor tokens (F2 hardening must stay
    # anchored: 'infinite domain' / 'this is fine' are not findings).
    assert check_summary_finite(tmp_path, "solved on an infinite domain, all is fine") == []


def test_stdout_arrow_and_copula_nonfinite_flagged(tmp_path):
    # Stress audit F2: arrow / copula headline forms evaded the =/: anchor.
    assert check_summary_finite(tmp_path, "max(u) -> nan")
    assert check_summary_finite(tmp_path, "residual is inf")


# ── Call-site wiring: the tools must actually apply the verdict ───────────
import asyncio            # noqa: E402
import json               # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest import mock  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402
import tools.consolidated as C  # noqa: E402


def _capture_tools():
    m = FastMCP("t")
    captured = {}
    orig = m.tool

    def cap(*a, **k):
        dec = orig(*a, **k)
        def wrap(fn):
            r = dec(fn); captured[fn.__name__] = fn; return r
        return wrap
    m.tool = cap
    C.register_consolidated_tools(m)
    return captured


class _Job:
    def __init__(self, wd, status="completed", error=None):
        self.job_id = "j"; self.status = status; self.work_dir = wd
        self.elapsed = 0.1; self.error = error


class _Backend:
    def __init__(self, files, status="completed", error=None):
        self._f, self._s, self._e = files, status, error
    def check_availability(self):
        return SimpleNamespace(value="available"), "ok"
    def validate_input(self, c):
        return []
    async def run(self, content, work_dir, np=1, timeout=None):
        return _Job(work_dir, self._s, self._e)
    def get_result_files(self, job):
        return self._f


def _run_sim(backend, **kw):
    caps = _capture_tools()
    with mock.patch.object(C, "get_backend", lambda s: backend):
        _loop = asyncio.new_event_loop()
        try:
            out = _loop.run_until_complete(
                caps["run_simulation"]("fenics", "print('x')", **kw))
        finally:
            _loop.close()
    return json.loads(out)


def test_run_simulation_verified_with_finite_output_and_critic(tmp_path):
    f = tmp_path / "out.vtu"; _write_vtu(f, [1.0, 2.0, 3.0, 4.0])
    _review_on_record("fenics", "print('x')")   # _run_sim's deck
    d = _run_sim(_Backend([f]), critic_approved=True)
    assert d["status"] == "completed"
    assert d["trustworthy_result"] is True
    assert d["verification"].startswith("VERIFIED")


def test_run_simulation_finite_output_without_critic_is_not_verified(tmp_path):
    # Enforced in software: a clean run WITHOUT the mandatory critic is not
    # trustworthy — but it still returns (no error).
    f = tmp_path / "out.vtu"; _write_vtu(f, [1.0, 2.0, 3.0, 4.0])
    d = _run_sim(_Backend([f]), critic_approved=False)
    assert d["status"] == "completed"           # the run still ran and returned
    assert d["trustworthy_result"] is False
    assert "MANDATORY" in d["verification"]


def test_run_simulation_nan_output_not_verified_even_if_critic_approved(tmp_path):
    f = tmp_path / "out.vtu"; _write_vtu(f, [np.nan, np.nan, np.nan, np.nan])
    d = _run_sim(_Backend([f]), critic_approved=True)
    assert d["trustworthy_result"] is False   # attestation beats critic
    assert d["verification"].startswith("NOT VERIFIED")
    assert any("non-finite" in v for v in d.get("validation", []))


def test_run_simulation_corrupt_output_not_verified_even_if_critic_approved(tmp_path):
    # Stress audit F1: a garbage/unreadable .vtu as the ONLY output used to be
    # stamped VERIFIED with the honesty note buried in 'validation'. A result
    # the gate cannot read is not verified run evidence — verdict must flip.
    f = tmp_path / "out.vtu"; f.write_text("%%% not a vtu at all")
    d = _run_sim(_Backend([f]), critic_approved=True)
    assert d["status"] == "completed"          # the run itself still returns
    assert d["trustworthy_result"] is False
    assert d["verification"].startswith("NOT VERIFIED")
    assert any("unreadable/corrupt" in v for v in d.get("validation", []))


def test_run_simulation_no_output_is_completed_unverified(tmp_path):
    d = _run_sim(_Backend([]), critic_approved=True)
    assert d["status"] == "completed_unverified"
    assert d["trustworthy_result"] is False
    # The claim is recorded and contradicted rather than silently accepted:
    # asserting approval with no review on record is itself worth reporting.
    assert "critic_approved=True" in d["critic_review"]


def test_run_simulation_error_is_not_verified(tmp_path):
    d = _run_sim(_Backend([], status="failed", error="boom"), critic_approved=True)
    assert d["trustworthy_result"] is False
    assert d["verification"].startswith("NOT VERIFIED")


def _run_precice(logs, converged=True, exchanged=True, exit_ok=True, **kw):
    """Stand in for run_precice_coupling.

    Its return contract changed: `converged` used to be a synonym for "every
    process exited 0" (which two scripts that never call preCICE also satisfy),
    and is now derived from preCICE's own per-window iteration record, alongside
    `exit_codes_ok` and `exchanged`. The stub mirrors the real shape so the
    assertions below still test the gate rather than the stub.
    """
    caps = _capture_tools()
    import core.precice_config as PC
    stub = lambda *a, **k: {"converged": converged,
                            "coupling_converged": converged,
                            "exchanged": exchanged,
                            "exit_codes_ok": exit_ok,
                            "evidence": [], "config_notes": [],
                            "returncodes": {"A": 0, "B": 0}, "logs": logs}
    import tempfile
    with mock.patch.object(PC, "check_precice_available", lambda: (True, "ok")), \
         mock.patch.object(PC, "run_precice_coupling", stub):
        _loop = asyncio.new_event_loop()
        try:
            out = _loop.run_until_complete(
                caps["couple_precice"]('[{"name":"A"},{"name":"B"}]', "[]", "[]",
                                       tempfile.mkdtemp(), **kw))
        finally:
            _loop.close()
    return json.loads(out)


def test_couple_precice_verified_on_clean_logs():
    from tools.consolidated import _coupling_setup_text
    _review_on_record("couple_precice", _coupling_setup_text(
        participants='[{"name":"A"},{"name":"B"}]', data="[]", exchanges="[]",
        scheme="serial-explicit", dimensions=2, max_time=10.0,
        time_window=1.0))
    d = _run_precice({"B": "coupling ok, residual 1e-9"}, critic_approved=True)
    assert d["trustworthy_result"] is True


def test_couple_precice_flags_nan_in_logs_despite_converged():
    d = _run_precice({"B": "residual nan detected"}, converged=True,
                     critic_approved=True)
    assert d["trustworthy_result"] is False   # log NaN downgrades, fails safe


def _reviewed_precice():
    from tools.consolidated import _coupling_setup_text
    _review_on_record("couple_precice", _coupling_setup_text(
        participants='[{"name":"A"},{"name":"B"}]', data="[]", exchanges="[]",
        scheme="serial-explicit", dimensions=2, max_time=10.0, time_window=1.0))


def test_couple_precice_clean_exit_without_exchange_is_not_verified():
    """Every participant exiting 0 is not a coupling.

    Two scripts that never call preCICE at all exit 0, and the verdict used to be
    driven by `converged`, which was itself just "all exited 0".
    """
    _reviewed_precice()
    d = _run_precice({"B": "done"}, exchanged=False, critic_approved=True)
    assert d["trustworthy_result"] is False
    assert any("NO EXCHANGE" in v for v in d.get("validation", []))


def test_couple_precice_nonconvergence_recorded_by_precice_is_not_verified():
    """An implicit scheme that exhausts max-iterations logs it and EXITS 0."""
    _reviewed_precice()
    d = _run_precice({"B": "done"}, converged=False, critic_approved=True)
    assert d["trustworthy_result"] is False
    assert any("NOT CONVERGED" in v for v in d.get("validation", []))


def test_couple_precice_unmeasured_convergence_is_reported_not_assumed():
    """An explicit scheme measures no convergence: say so, do not imply it."""
    _reviewed_precice()
    d = _run_precice({"B": "done"}, converged=None, critic_approved=True)
    assert d["trustworthy_result"] is True          # nothing here is WRONG
    assert any("convergence" in c for c in d.get("checks_not_run", []))
    assert "COVERAGE" in d["verification"]          # and the verdict says so
