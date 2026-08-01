"""End-to-end smoke of the repo's preCICE bridge (core.precice_config).

Runs a REAL two-participant coupling through preCICE (sockets m2n, generated
precice-config.xml, serial-implicit Dirichlet-Neumann + Aitken) — not the
file-handshake driver. Both participants are scikit-fem 1D heat slabs; the
exchanged interface temperature is checked against the analytic two-slab
answer. Exercises run_precice_coupling, the exact function behind the
couple_precice MCP tool.

Skips cleanly when /opt/precice (libprecice) is absent, pyprecice does not
import, or scikit-fem is missing.
"""
import json
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

PAIR_DIR = REPO / "benchmarks" / "coupling_pairs" / "precice_smoke"
PRECICE_LIB_DIR = os.environ.get("PRECICE_LIB_DIR", "/opt/precice/lib")


def _precice_ok() -> bool:
    if not Path(PRECICE_LIB_DIR).exists():
        return False
    from core.precice_config import check_precice_available
    ok, _ = check_precice_available()
    return ok


def _skfem_ok() -> bool:
    try:
        import skfem  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = [
    pytest.mark.skipif(not Path(PRECICE_LIB_DIR).exists(),
                       reason=f"libprecice not found at {PRECICE_LIB_DIR}"),
    pytest.mark.skipif(Path(PRECICE_LIB_DIR).exists() and not _precice_ok(),
                       reason="pyprecice not importable against "
                              f"{PRECICE_LIB_DIR}"),
    pytest.mark.skipif(not _skfem_ok(), reason="scikit-fem not installed"),
]


def test_check_precice_available_reports_version():
    from core.precice_config import check_precice_available
    ok, msg = check_precice_available()
    assert ok
    assert "preCICE 3" in msg


def test_precice_two_slab_dn_converges_to_analytic(tmp_path):
    from benchmarks.coupling_pairs.precice_smoke.run_smoke import analytic, run

    params = json.loads((PAIR_DIR / "params.json").read_text())
    T_ref, q_ref = analytic(params)

    result = run(work_dir=tmp_path, timeout=300)
    assert result["converged"], f"preCICE coupling failed: {result}"
    assert result["returncodes"] == {"SlabA": 0, "SlabB": 0}

    rb = json.loads((tmp_path / "result_SlabB.json").read_text())
    ra = json.loads((tmp_path / "result_SlabA.json").read_text())
    T_if = rb["final_interface_temperature"]
    q = ra["final_flux"]
    assert abs(T_if - T_ref) < 1e-3, f"T_if={T_if} vs analytic {T_ref}"
    assert abs(q - q_ref) < 1e-3, f"q={q} vs analytic {q_ref}"

    # implicit scheme actually iterated and preCICE reports convergence
    assert ra["iterations"] >= 2
    it_log = tmp_path / "precice-SlabA-iterations.log"
    assert it_log.exists(), "preCICE iterations log missing"
    last = [ln for ln in it_log.read_text().splitlines() if ln.strip()][-1]
    assert last.split()[-1] == "1", f"final window not converged: {last!r}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
