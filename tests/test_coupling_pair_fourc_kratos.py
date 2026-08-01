"""Live coupled smoke: 4C <-> Kratos two-slab conjugate heat transfer through
the generic file-handshake coupling driver, verified against the analytic
series-resistance interface temperature.

Slab A is solved by the REAL 4C binary (Thermo, interface Dirichlet from
imports.json, exports one-sided interface flux); slab B by REAL Kratos
ConvectionDiffusion under /usr/bin/python3 (interface Neumann via
FACE_HEAT_FLUX, exports interface temperature). See
benchmarks/coupling_pairs/fourc_kratos_cht/.

Skips cleanly when the 4C binary or the system-python Kratos install is
absent (both are machine-local, not CI dependencies).
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

PAIR_DIR = REPO / "benchmarks" / "coupling_pairs" / "fourc_kratos_cht"
PARAMS = json.loads((PAIR_DIR / "params.json").read_text())


def _kratos_usable() -> bool:
    py = PARAMS["kratos_python"]
    if not shutil.which(py) and not Path(py).exists():
        return False
    try:
        r = subprocess.run(
            [py, "-c", "import KratosMultiphysics,"
             " KratosMultiphysics.ConvectionDiffusionApplication"],
            capture_output=True, timeout=120)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.skipif(not Path(PARAMS["fourc_binary"]).exists(),
                       reason=f"4C binary not found at {PARAMS['fourc_binary']}"),
    pytest.mark.skipif(not _kratos_usable(),
                       reason="Kratos (ConvectionDiffusionApplication) not importable "
                              f"under {PARAMS['kratos_python']}"),
]


def test_fourc_kratos_two_slab_cht_converges_to_analytic(tmp_path):
    from benchmarks.coupling_pairs.fourc_kratos_cht.run_pair import (
        analytic, build_participants)
    from core.coupling_driver import run_coupling

    pa, pb = build_participants(tmp_path, PAIR_DIR / "params.json", PARAMS)
    result = run_coupling([pa, pb], max_iter=100, tol=1e-6, accelerator="aitken")

    assert result.converged, f"coupling failed: {result.error}"
    assert result.residual < 1e-6

    T_ref, q_ref = analytic(PARAMS)
    T_if = float(np.mean(result.exports["KratosSlabB"]["values"]))
    assert abs(T_if - T_ref) < 0.05, (
        f"interface temperature {T_if} != analytic {T_ref}")

    # interface flux balance: A's outward (+x) flux vs B's outward (-x) flux
    q_a = float(np.mean(result.exports["FourCSlabA"]["normal_fluxes"]))
    q_b = float(np.mean(result.exports["KratosSlabB"]["normal_fluxes"]))
    assert abs(q_a - q_ref) < 0.05
    assert abs(q_a + q_b) < 0.05, f"interface fluxes not balanced: {q_a} vs {q_b}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
