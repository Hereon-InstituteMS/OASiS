"""Run the 4C <-> Kratos two-slab conjugate-heat pair through the generic
file-handshake coupling driver (src/core/coupling_driver.run_coupling).

Slab A: real 4C binary (thermo, interface Dirichlet, exports interface flux).
Slab B: real Kratos ConvectionDiffusion (interface Neumann, exports interface T).
Verified against the analytic series-resistance interface temperature.

Usage:  .venv/bin/python benchmarks/coupling_pairs/fourc_kratos_cht/run_pair.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PAIR_DIR = Path(__file__).resolve().parent
REPO = PAIR_DIR.parents[2]
sys.path.insert(0, str(REPO / "src"))

from core.coupling_driver import Participant, run_coupling  # noqa: E402


def analytic(params: dict) -> tuple[float, float]:
    """Series-resistance answer: interface temperature and flux."""
    TL, TR = params["TL"], params["TR"]
    a, b = params["k1"] / params["L1"], params["k2"] / params["L2"]
    T_if = (TL * a + TR * b) / (a + b)
    q = (TL - TR) / (params["L1"] / params["k1"] + params["L2"] / params["k2"])
    return T_if, q


def build_participants(base: Path, params_file: Path, params: dict):
    pa = Participant(
        name="FourCSlabA",
        command=[sys.executable, str(PAIR_DIR / "participant_fourc.py")],
        work_dir=base / "slabA_4c",
        imports_from=["KratosSlabB"],
        data_files=[str(params_file)],
        timeout=600,
    )
    pb = Participant(
        name="KratosSlabB",
        command=[params["kratos_python"], str(PAIR_DIR / "participant_kratos.py")],
        work_dir=base / "slabB_kratos",
        imports_from=["FourCSlabA"],
        data_files=[str(params_file)],
        timeout=600,
    )
    return pa, pb


def main(base: Path = None, max_iter: int = 40, tol: float = 1e-8):
    params_file = PAIR_DIR / "params.json"
    params = json.loads(params_file.read_text())
    base = Path(base or tempfile.mkdtemp(prefix="fourc_kratos_cht_"))
    pa, pb = build_participants(base, params_file, params)
    result = run_coupling([pa, pb], max_iter=max_iter, tol=tol, accelerator="aitken")

    T_ref, q_ref = analytic(params)
    print(f"converged={result.converged} iterations={result.iterations} "
          f"residual={result.residual:.3e}")
    if result.error:
        print("ERROR:", result.error)
    if result.converged:
        T_if = float(np.mean(result.exports["KratosSlabB"]["values"]))
        q_a = float(np.mean(result.exports["FourCSlabA"]["normal_fluxes"]))
        q_b = float(np.mean(result.exports["KratosSlabB"]["normal_fluxes"]))
        print(f"T_if   coupled={T_if:.6f}  analytic={T_ref:.6f}  "
              f"|err|={abs(T_if - T_ref):.2e}")
        print(f"flux   A(out)={q_a:.6f}  B(out)={q_b:.6f}  analytic q={q_ref:.6f}")
    return result, params


if __name__ == "__main__":
    r, _ = main()
    sys.exit(0 if r.converged else 1)
