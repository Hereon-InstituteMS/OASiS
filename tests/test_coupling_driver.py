"""Tests for the general physics-agnostic coupling driver + output-side validators.

These verify the OVERHAUL: that coupling works with no hardcoded physics/geometry,
converges a known fixed point, and that the silent-wrong guards fire correctly.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.coupling_driver import Participant, run_coupling
from core.quality_checks import (
    check_finite, check_convergence, check_interface_balance,
    check_monolithic_consistency,
)


def _write_participant(d: Path, name: str, body: str):
    p = d / name
    p.mkdir(parents=True, exist_ok=True)
    (p / "run.py").write_text(body)
    return p


def test_general_fixedpoint_converges(tmp_path):
    """Two opaque participants (no physics): x=0.5y+1, y=0.5x+2 -> x=8/3, y=10/3.
    The driver must converge to the analytic fixed point via file-handshake only."""
    a = _write_participant(tmp_path, "A", (
        'import json\nfrom pathlib import Path\n'
        'imp=json.loads(Path("imports.json").read_text())\n'
        'y=imp["B"]["values"][0] if "B" in imp else 0.0\n'
        'json.dump({"field_name":"x","n_points":1,"coordinates":[[0.0]],"values":[0.5*y+1.0]},open("exports.json","w"))\n'))
    b = _write_participant(tmp_path, "B", (
        'import json\nfrom pathlib import Path\n'
        'imp=json.loads(Path("imports.json").read_text())\n'
        'x=imp["A"]["values"][0] if "A" in imp else 0.0\n'
        'json.dump({"field_name":"y","n_points":1,"coordinates":[[0.0]],"values":[0.5*x+2.0]},open("exports.json","w"))\n'))
    pa = Participant("A", [sys.executable, "run.py"], a, imports_from=["B"])
    pb = Participant("B", [sys.executable, "run.py"], b, imports_from=["A"])
    r = run_coupling([pa, pb], max_iter=80, tol=1e-9)
    assert r.converged
    assert abs(r.exports["A"]["values"][0] - 8 / 3) < 1e-5
    assert abs(r.exports["B"]["values"][0] - 10 / 3) < 1e-5


def test_nonconvergence_reported_as_failure(tmp_path):
    """A divergent map must be reported converged=False with a loud error — never a result."""
    a = _write_participant(tmp_path, "A", (
        'import json\nfrom pathlib import Path\n'
        'imp=json.loads(Path("imports.json").read_text())\n'
        'y=imp["B"]["values"][0] if "B" in imp else 1.0\n'
        'json.dump({"field_name":"x","n_points":1,"coordinates":[[0.0]],"values":[3.0*y+1.0]},open("exports.json","w"))\n'))
    b = _write_participant(tmp_path, "B", (
        'import json\nfrom pathlib import Path\n'
        'imp=json.loads(Path("imports.json").read_text())\n'
        'x=imp["A"]["values"][0] if "A" in imp else 1.0\n'
        'json.dump({"field_name":"y","n_points":1,"coordinates":[[0.0]],"values":[3.0*x+1.0]},open("exports.json","w"))\n'))
    pa = Participant("A", [sys.executable, "run.py"], a, imports_from=["B"])
    pb = Participant("B", [sys.executable, "run.py"], b, imports_from=["A"])
    r = run_coupling([pa, pb], max_iter=8, tol=1e-9, accelerator="constant", theta0=1.0)
    assert not r.converged
    assert r.error and "not converge" in r.error.lower()


def test_missing_exports_is_failure(tmp_path):
    """A participant that writes no exports.json must produce a clear failure, not a hang."""
    a = _write_participant(tmp_path, "A", 'print("I do nothing")\n')
    b = _write_participant(tmp_path, "B", 'print("me neither")\n')
    pa = Participant("A", [sys.executable, "run.py"], a, imports_from=["B"])
    pb = Participant("B", [sys.executable, "run.py"], b, imports_from=["A"])
    r = run_coupling([pa, pb], max_iter=5, tol=1e-6)
    assert not r.converged
    assert "exports.json" in (r.error or "")


def test_validators():
    assert check_finite([1.0, np.nan])
    assert not check_finite([1.0, 2.0])
    assert check_convergence(False, 1e-2, 1e-6)
    assert not check_convergence(True, 1e-8, 1e-6)
    # balanced: A=+3, B=-3 -> ok; imbalanced otherwise
    assert not check_interface_balance({"normal_fluxes": [1, 1, 1]}, {"normal_fluxes": [-1, -1, -1]})
    assert check_interface_balance({"normal_fluxes": [1, 1, 1]}, {"normal_fluxes": [-1, -1, 2]})
    assert check_monolithic_consistency(96.9, 50.0)
    assert not check_monolithic_consistency(50.0, 50.5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ── the stochastic branch: a residual floor that is MEASURED, not assumed ────
#
# A Monte-Carlo participant answers the same question slightly differently every
# time, so the residual — which is the CHANGE in the export — cannot fall below
# that scatter however well the coupling has converged. A `tol` under the floor
# is unreachable by construction and every run of a correct coupling was
# reported as a failure. These use a noisy participant with a seed drawn per
# invocation, so they exercise the real mechanism with no solver involved.

_NOISY = (
    'import json, os, random\nfrom pathlib import Path\n'
    'random.seed(int.from_bytes(os.urandom(8), "little"))\n'
    'imp=json.loads(Path("imports.json").read_text() or "{}")\n'
    'y=imp["B"]["values"][0] if "B" in imp else 0.0\n'
    'v=0.5*y+1.0\n'
    'v*= 1.0 + AMP*(random.random()-0.5)\n'
    'json.dump({"field_name":"x","n_points":1,"coordinates":[[0.0]],'
    '"values":[v]},open("exports.json","w"))\n')

_QUIET_B = (
    'import json\nfrom pathlib import Path\n'
    'imp=json.loads(Path("imports.json").read_text() or "{}")\n'
    'x=imp["A"]["values"][0] if "A" in imp else 0.0\n'
    'json.dump({"field_name":"y","n_points":1,"coordinates":[[0.0]],'
    '"values":[0.5*x+2.0]},open("exports.json","w"))\n')


def _noisy_pair(tmp_path, amp=0.05):
    a = _write_participant(tmp_path, "A", _NOISY.replace("AMP", repr(amp)))
    b = _write_participant(tmp_path, "B", _QUIET_B)
    return (Participant("A", [sys.executable, "run.py"], a, imports_from=["B"]),
            Participant("B", [sys.executable, "run.py"], b, imports_from=["A"]))


def test_stochastic_participant_fails_without_the_noise_branch(tmp_path):
    """The symptom the branch exists to remove: a correct coupling, a tol under
    the sampling noise, and an unconditional FAILURE."""
    pa, pb = _noisy_pair(tmp_path)
    r = run_coupling([pa, pb], max_iter=25, tol=1e-9, accelerator="constant")
    assert not r.converged
    assert r.noise_floor is None, "no floor was asked for, so none may be claimed"
    assert "did not converge" in (r.error or "")
    # And the failure message must point at the measurement rather than leaving
    # the reader to halve theta forever.
    assert "noise_replicates" in (r.error or ""), (
        "a residual that STOPPED FALLING should name the noise-floor route")


def test_measured_floor_lets_a_stochastic_coupling_converge(tmp_path):
    """Same run, floor measured: converged, judged against max(tol, floor), and
    the floor reported so nobody grades tighter than the sampler allows."""
    pa, pb = _noisy_pair(tmp_path)
    # FIVE, not three. The floor is itself an estimate and three replicates
    # give only three non-independent pairs: the same coupling measured
    # 1.2e-03 from three and 9.6e-03 from five, a factor of eight of pure
    # estimator scatter on the number the verdict is compared against.
    # A generous iteration budget, on purpose. Stopping needs a BLOCK of
    # consecutive residuals to average below the floor, and once the run is IN
    # the noise each block is roughly a coin toss — which is the point of block
    # averaging. Twenty-odd chances make a spurious failure vanishingly
    # unlikely without weakening anything the test asserts.
    r = run_coupling([pa, pb], max_iter=60, tol=1e-9, accelerator="constant",
                     noise_replicates=6)
    assert r.converged
    assert r.stopped_at_noise_floor
    assert r.noise_floor and r.noise_floor > 1e-9
    assert r.tol_effective == max(1e-9, r.noise_floor)
    # SAID, and said in `criterion_notes` rather than in `warnings`. The
    # sentence and its job are unchanged — a verdict reached at the floor that
    # does not SAY so is a softened failure, which is worse than the honest one
    # it replaced — but the field moved when this branch met
    # feature/coupling-robustness. There, `couple` takes ANY entry in the
    # findings list as making a coupling untrustworthy, on purpose, so that no
    # check can be lost by rewording; and it builds that list from `warnings`.
    # A correct stochastic coupling therefore came back NOT VERIFIED, which is
    # the verdict this whole branch exists to stop being unavoidable. The notice
    # is now reported in the coverage channel, which is always printed in the
    # verdict and never flips it.
    assert any("NOISE FLOOR" in w.upper() for w in r.criterion_notes), (
        "a verdict reached at the floor that does not SAY so is a softened "
        "failure, which is worse than the honest one it replaced")
    assert not any("NOISE FLOOR" in w.upper() for w in r.warnings), (
        "the criterion notice must NOT be a warning: `couple` copies warnings "
        "into `validation` and treats a non-empty `validation` as untrustworthy")
    # The physics is still right: the fixed point of x=0.5y+1, y=0.5x+2.
    assert abs(r.exports["A"]["values"][0] - 8 / 3) < 0.3
    assert abs(r.exports["B"]["values"][0] - 10 / 3) < 0.3


def test_noise_branch_is_inert_for_deterministic_participants(tmp_path):
    """max(tol, 0) is tol. A branch that changed a deterministic verdict would
    be a way of passing failures."""
    pa, pb = _noisy_pair(tmp_path, amp=0.0)
    r = run_coupling([pa, pb], max_iter=80, tol=1e-9, accelerator="constant",
                     noise_replicates=6)
    assert r.converged
    assert r.noise_floor == 0.0
    assert not r.stopped_at_noise_floor
    assert r.tol_effective == 1e-9
    assert abs(r.exports["A"]["values"][0] - 8 / 3) < 1e-5
    # The measurement's provenance is a NOTE, not a warning: `warnings` becomes
    # the tool's `validation` block, and an agent is told an empty validation
    # block is what a correct coupling looks like.
    assert not r.warnings, r.warnings
    assert any("SEED IS FIXED" in n for n in r.notes), (
        "a floor of exactly zero must be reported \u2014 on a Monte-Carlo "
        "participant it means the seed is fixed, and a residual that falls "
        "under a fixed seed proves only that one draw was repeated")


def test_declared_floor_is_honoured_without_replicates(tmp_path):
    """A caller who measured the floor elsewhere can declare it."""
    pa, pb = _noisy_pair(tmp_path)
    r = run_coupling([pa, pb], max_iter=25, tol=1e-9, accelerator="constant",
                     noise_floor=0.2)
    assert r.converged and r.stopped_at_noise_floor
    assert r.tol_effective == 0.2
