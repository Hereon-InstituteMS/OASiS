"""End-to-end preCICE bridge smoke: two scikit-fem heat-slab participants
coupled serial-implicitly THROUGH the repo's preCICE orchestration
(core.precice_config.run_precice_coupling — the exact function the
couple_precice MCP tool calls), verified against the analytic two-slab
interface temperature.

Usage:  .venv/bin/python benchmarks/coupling_pairs/precice_smoke/run_smoke.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

PAIR_DIR = Path(__file__).resolve().parent
REPO = PAIR_DIR.parents[2]
sys.path.insert(0, str(REPO / "src"))

from core.precice_config import run_precice_coupling  # noqa: E402


def analytic(params: dict) -> tuple[float, float]:
    TL, TR = params["TL"], params["TR"]
    a, b = params["k1"] / params["L1"], params["k2"] / params["L2"]
    return (TL * a + TR * b) / (a + b), \
        (TL - TR) / (params["L1"] / params["k1"] + params["L2"] / params["k2"])


def run(work_dir: Path = None, timeout: int = 300) -> dict:
    wd = Path(work_dir or tempfile.mkdtemp(prefix="precice_smoke_"))
    wd.mkdir(parents=True, exist_ok=True)
    shutil.copy(PAIR_DIR / "params.json", wd / "params.json")
    script = str(PAIR_DIR / "participant_slab.py")
    py = sys.executable
    result = run_precice_coupling(
        participants=[
            {"name": "SlabA", "mesh": "SlabA-Mesh",
             "writes": ["Heat-Flux"], "reads": ["Interface-Temperature"],
             "command": [py, script, "SlabA"]},
            {"name": "SlabB", "mesh": "SlabB-Mesh",
             "writes": ["Interface-Temperature"], "reads": ["Heat-Flux"],
             "command": [py, script, "SlabB"]},
        ],
        data=[{"name": "Interface-Temperature", "type": "scalar"},
              {"name": "Heat-Flux", "type": "scalar"}],
        # exchanges[0] doubles as the convergence-measure/acceleration datum in
        # the generated config; it must be written by the SECOND participant
        # (SlabB) for a serial-implicit scheme.
        exchanges=[{"data": "Interface-Temperature", "from": "SlabB", "to": "SlabA"},
                   {"data": "Heat-Flux", "from": "SlabA", "to": "SlabB"}],
        work_dir=wd,
        scheme="serial-implicit",
        dimensions=2,
        max_time=1.0,
        time_window=1.0,
        max_iterations=50,
        convergence_tol=1e-8,
        timeout=timeout,
    )
    result["work_dir"] = str(wd)
    return result


def main() -> int:
    params = json.loads((PAIR_DIR / "params.json").read_text())
    T_ref, q_ref = analytic(params)
    r = run()
    wd = Path(r["work_dir"])
    print("converged:", r["converged"], "returncodes:", r["returncodes"])
    if not r["converged"]:
        print(json.dumps(r, indent=2))
        return 1
    rb = json.loads((wd / "result_SlabB.json").read_text())
    ra = json.loads((wd / "result_SlabA.json").read_text())
    T_if, q = rb["final_interface_temperature"], ra["final_flux"]
    print(f"T_if  precice={T_if:.8f}  analytic={T_ref:.8f}  |err|={abs(T_if - T_ref):.2e}")
    print(f"q     precice={q:.8f}  analytic={q_ref:.8f}  |err|={abs(q - q_ref):.2e}")
    print(f"coupling iterations: SlabA={ra['iterations']}, SlabB={rb['iterations']}")
    it_log = wd / "precice-SlabA-iterations.log"
    if it_log.exists():
        print("preCICE iterations log:\n" + it_log.read_text())
    return 0 if abs(T_if - T_ref) < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
