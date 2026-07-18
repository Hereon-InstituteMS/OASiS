"""Tests for the verification-gate verdict (_stamp_verification).

OASiS enforces verification IN SOFTWARE: OASiS verifies via numerical checks and
*attestation* — binding every reported number to run evidence — but does not
validate. The independent pre-execution critic is MANDATORY: a result is
trustworthy ONLY when it passes attestation + numerical checks AND the critic
approved (critic_approved=True). Enforced by verdict, never by error. These tests
pin exactly that, plus the anti-fabrication labelling and the eval ablation
(OFA_DISABLE_CRITIC lifts only the critic requirement).
"""
import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "src")
sys.path.insert(0, SRC)

from tools.consolidated import _stamp_verification  # noqa: E402


def test_evidence_and_critic_is_verified():
    # A verified result needs BOTH attestation and the mandatory critic.
    r = _stamp_verification({}, evidence_ok=True, critic_approved=True)
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
    with_critic = _stamp_verification({}, evidence_ok=True, critic_approved=True)
    without = _stamp_verification({}, evidence_ok=True, critic_approved=False)
    assert with_critic["trustworthy_result"] is True
    assert without["trustworthy_result"] is False        # critic is MANDATORY
    assert without["verification"].startswith("NOT VERIFIED")
    assert "MANDATORY" in without["verification"]
    assert with_critic["critic_review"] == "approved"
    assert "REQUIRED" in without["critic_review"]


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


def test_ablation_lifts_the_mandatory_critic_requirement():
    """OFA_DISABLE_CRITIC (the held-out eval's ablation) lifts ONLY the mandatory
    critic requirement, so an evidence-backed run verifies without critic
    approval — this is how the eval measures the critic's contribution."""
    review, trust = _critic_review_under_ablation().splitlines()
    assert review == "disabled for evaluation"
    assert trust == "True"   # under ablation, attestation alone verifies


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
    # A best-effort scan must NEVER raise (incl. SystemExit). An unreadable file
    # yields the HONESTY NOTE (finiteness could not be asserted) — never a false
    # NaN finding and never a crash.
    w = check_result_files_finite([bad])
    assert all("non-finite" not in x for x in w)   # no false NaN finding
    assert any("not asserted" in x for x in w)     # honesty note present


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
    # only a NaN/Inf presented as a numeric result (after = or :) counts.
    assert check_summary_finite(tmp_path, "computed 42 nanometers of information") == []


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


def test_run_simulation_no_output_is_completed_unverified(tmp_path):
    d = _run_sim(_Backend([]), critic_approved=True)
    assert d["status"] == "completed_unverified"
    assert d["trustworthy_result"] is False
    assert d["critic_review"] == "approved"   # critic set, but no run evidence


def test_run_simulation_error_is_not_verified(tmp_path):
    d = _run_sim(_Backend([], status="failed", error="boom"), critic_approved=True)
    assert d["trustworthy_result"] is False
    assert d["verification"].startswith("NOT VERIFIED")


def _run_precice(logs, converged=True, **kw):
    caps = _capture_tools()
    import core.precice_config as PC
    stub = lambda *a, **k: {"converged": converged,
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
    d = _run_precice({"B": "coupling ok, residual 1e-9"}, critic_approved=True)
    assert d["trustworthy_result"] is True


def test_couple_precice_flags_nan_in_logs_despite_converged():
    d = _run_precice({"B": "residual nan detected"}, converged=True,
                     critic_approved=True)
    assert d["trustworthy_result"] is False   # log NaN downgrades, fails safe
