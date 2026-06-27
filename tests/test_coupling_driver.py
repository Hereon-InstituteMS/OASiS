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
