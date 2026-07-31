"""deal.II <-> NGSolve conjugate-heat coupling pair (B3a).

Structure/contract tests always run; the NGSolve participant is exercised
live standalone (ngsolve ships in the repo .venv); the deal.II participant
and the full coupled run are guarded by availability of a deal.II build
environment and skip cleanly without one.

The physics: two stacked slabs, Dirichlet-Neumann partitioned coupling via
the generic driver (core.coupling_driver). Analytic interface temperature
T_if = (k1/H1*T_bot + k2/H2*T_top) / (k1/H1 + k2/H2) is the ground truth.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PAIR_DIR = REPO / "benchmarks" / "coupling_pairs" / "dealii_ngsolve_cht"
PAIR_FILES = ["heat_slab_dealii.cc", "CMakeLists.txt", "participant_dealii.py",
              "participant_ngsolve.py", "run_pair.py"]

_spec = importlib.util.spec_from_file_location("dealii_ngsolve_run_pair",
                                               PAIR_DIR / "run_pair.py")
run_pair_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_pair_mod)

HAVE_NGSOLVE = importlib.util.find_spec("ngsolve") is not None
HAVE_DEALII = run_pair_mod.dealii_available()


# ── (a) structure / contract, always run ────────────────────────────────

def test_pair_files_exist():
    for name in PAIR_FILES:
        assert (PAIR_DIR / name).is_file(), f"missing pair file: {name}"


def test_analytic_interface_temperature():
    f = run_pair_mod.analytic_interface_temperature
    # symmetric slabs -> midpoint temperature
    assert f(2.0, 1.0, 2.0, 1.0, 100.0, 0.0) == pytest.approx(50.0)
    # k1 >> k2 -> interface pinned to the k1-side boundary value
    assert f(1e9, 1.0, 1.0, 1.0, 100.0, 0.0) == pytest.approx(100.0, abs=1e-5)
    # unequal heights: thicker slab = larger resistance
    t = f(1.0, 3.0, 1.0, 1.0, 40.0, 10.0)
    assert t == pytest.approx((40.0 / 3.0 + 10.0) / (1.0 / 3.0 + 1.0))
    # flux continuity at the analytic fixed point
    k1, H1, k2, H2, tb, tt = 1.7, 0.8, 0.6, 1.4, 25.0, 5.0
    t_if = f(k1, H1, k2, H2, tb, tt)
    assert k1 / H1 * (tb - t_if) == pytest.approx(k2 / H2 * (t_if - tt))


def test_participants_follow_driver_contract():
    """Both participant scripts must speak the imports.json/exports.json
    file handshake of the generic driver (static check)."""
    for name in ("participant_dealii.py", "participant_ngsolve.py"):
        src = (PAIR_DIR / name).read_text()
        assert "imports.json" in src, f"{name} ignores imports.json"
        assert "exports.json" in src, f"{name} writes no exports.json"
        assert "params.json" in src, f"{name} has no parameter input"


# ── (b) live standalone participants ────────────────────────────────────

@pytest.mark.skipif(not HAVE_NGSOLVE, reason="ngsolve not importable")
def test_ngsolve_participant_standalone(tmp_path):
    """Feed the NGSolve (Neumann) participant the ANALYTIC interface flux via
    a synthetic imports.json; its exported interface temperature must hit the
    analytic T_if."""
    k1, H1, k2, H2 = 0.9, 1.0, 1.8, 0.7
    tb, tt = 21.0, 3.0
    y_if, y_top = 1.0, 1.0 + H2
    t_if = run_pair_mod.analytic_interface_temperature(k1, H1, k2, H2, tb, tt)
    q_if = run_pair_mod.analytic_interface_flux(k1, H1, k2, H2, tb, tt)

    (tmp_path / "params.json").write_text(json.dumps({
        "k": k2, "x_min": 0.0, "x_max": 1.0, "y_min": y_if, "y_max": y_top,
        "T_dirichlet": tt, "flux_init": 0.0, "maxh": 0.12, "order": 2,
        "n_samples": 9, "poly_degree": 3, "partner": "dealii"}))
    (tmp_path / "imports.json").write_text(json.dumps({"dealii": {
        "field_name": "temperature", "n_points": 5,
        "coordinates": [[x / 4.0, y_if] for x in range(5)],
        "values": [t_if] * 5, "normal_fluxes": [q_if] * 5}}))

    r = subprocess.run([sys.executable, str(PAIR_DIR / "participant_ngsolve.py")],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    exp = json.loads((tmp_path / "exports.json").read_text())
    assert exp["n_points"] == 9
    for v in exp["values"]:
        assert v == pytest.approx(t_if, abs=1e-4)


@pytest.mark.skipif(not HAVE_DEALII, reason="deal.II build env not found")
def test_dealii_participant_standalone(tmp_path):
    """Feed the deal.II (Dirichlet) participant the ANALYTIC interface
    temperature; its exported normal flux must hit the analytic q_if."""
    k1, H1, k2, H2 = 1.1, 0.9, 0.4, 1.2
    tb, tt = 55.0, 15.0
    t_if = run_pair_mod.analytic_interface_temperature(k1, H1, k2, H2, tb, tt)
    q_if = run_pair_mod.analytic_interface_flux(k1, H1, k2, H2, tb, tt)
    exe = run_pair_mod.ensure_dealii_built()

    (tmp_path / "params.json").write_text(json.dumps({
        "k": k1, "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": H1,
        "T_dirichlet": tb, "T_interface_init": 0.5 * (tb + tt),
        "nx": 10, "ny": 10, "degree": 2, "exe": str(exe),
        "partner": "ngsolve"}))
    (tmp_path / "imports.json").write_text(json.dumps({"ngsolve": {
        "field_name": "temperature", "n_points": 3,
        "coordinates": [[0.0, H1], [0.5, H1], [1.0, H1]],
        "values": [t_if] * 3}}))

    r = subprocess.run([sys.executable, str(PAIR_DIR / "participant_dealii.py")],
                       cwd=tmp_path, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    exp = json.loads((tmp_path / "exports.json").read_text())
    assert exp["n_points"] > 0
    for q in exp["normal_fluxes"]:
        assert q == pytest.approx(q_if, abs=1e-6)


# ── (c) full live coupled run through the generic driver ────────────────

@pytest.mark.skipif(not (HAVE_DEALII and HAVE_NGSOLVE),
                    reason="needs both deal.II build env and ngsolve")
def test_live_pair_converges_to_analytic(tmp_path):
    rep = run_pair_mod.run_pair(
        params={"nx": 10, "ny": 10, "maxh": 0.14, "n_samples": 9},
        workdir=tmp_path / "coupling", max_iter=40, tol=1e-7)
    assert rep["converged"], f"coupling failed: {rep['error']}"
    assert rep["iterations"] >= 2
    assert rep["residual"] < 1e-7
    assert rep["T_if_error"] < 1e-4, (
        f"interface T {rep['T_if_computed']} vs analytic {rep['T_if_analytic']}")
    # both sides must agree on the converged interface temperature
    assert rep["T_if_dealii"] == pytest.approx(rep["T_if_ngsolve"], abs=1e-3)
    # exported flux must match the analytic series flux
    assert rep["q_if_dealii"] == pytest.approx(rep["q_if_analytic"], abs=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
