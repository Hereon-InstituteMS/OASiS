"""deal.II <-> NGSolve conjugate-heat coupling pair (T12-type, steady).

Two stacked conducting slabs sharing the interface y = y_if:
  * slab A (deal.II):  [x_min,x_max] x [y_bot,y_if], conductivity k1,
    Dirichlet T = T_bot on y = y_bot, imports the interface TEMPERATURE
    (Dirichlet), exports the interface normal heat flux q = -k1 dT/dy.
  * slab B (NGSolve):  [x_min,x_max] x [y_if,y_top], conductivity k2,
    Dirichlet T = T_top on y = y_top, imports the interface FLUX (Neumann),
    exports the interface temperature.
Lateral sides insulated -> the exact solution is 1D series conduction with
analytic interface temperature
    T_if = (k1/H1 * T_bot + k2/H2 * T_top) / (k1/H1 + k2/H2),
H1 = y_if - y_bot, H2 = y_top - y_if. The Dirichlet-Neumann fixed point is
iterated by the generic physics-agnostic driver (core.coupling_driver) with
Aitken relaxation.

Usage:
    .venv/bin/python benchmarks/coupling_pairs/dealii_ngsolve_cht/run_pair.py
Requires a deal.II installation (env DEAL_II_DIR or a discoverable build);
the deal.II solver is compiled on demand with cmake/make.

All problem numbers are parameters (DEFAULT_PARAMS holds arbitrary dev-draw
values); override any of them via run_pair(params={...}).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PAIR_DIR = Path(__file__).resolve().parent
REPO_SRC = PAIR_DIR.parents[2] / "src"

# Arbitrary development values — every one of them is overridable.
DEFAULT_PARAMS = {
    "k1": 1.3,            # conductivity slab A (deal.II)
    "k2": 2.6,            # conductivity slab B (NGSolve)
    "T_bot": 310.0,       # Dirichlet on y = y_bot (slab A outer boundary)
    "T_top": 290.0,       # Dirichlet on y = y_top (slab B outer boundary)
    "x_min": 0.0,
    "x_max": 1.0,
    "y_bot": 0.0,
    "y_if": 1.0,          # coupling interface
    "y_top": 2.0,
    # discretization
    "nx": 16, "ny": 16, "degree": 2,      # deal.II slab
    "maxh": 0.09, "order": 2,             # NGSolve slab
    "n_samples": 13,                      # NGSolve interface export points
    "poly_degree": 3,                     # flux-fit degree on Neumann side
}


def analytic_interface_temperature(k1: float, H1: float, k2: float, H2: float,
                                   T_bot: float, T_top: float) -> float:
    """1D series-conduction interface temperature for two stacked slabs."""
    a, b = k1 / H1, k2 / H2
    return (a * T_bot + b * T_top) / (a + b)


def analytic_interface_flux(k1: float, H1: float, k2: float, H2: float,
                            T_bot: float, T_top: float) -> float:
    """Heat flux through the interface in +y direction."""
    a = k1 / H1
    T_if = analytic_interface_temperature(k1, H1, k2, H2, T_bot, T_top)
    return a * (T_bot - T_if)


def find_dealii_dir() -> Path | None:
    """Locate a deal.II installation (env DEAL_II_DIR first)."""
    candidates = []
    if os.environ.get("DEAL_II_DIR"):
        candidates.append(Path(os.environ["DEAL_II_DIR"]))
    candidates += [Path.home() / "dealii" / "build",
                   Path("/usr/local"), Path("/usr")]
    for c in candidates:
        if (c / "lib" / "cmake" / "deal.II" / "deal.IIConfig.cmake").is_file():
            return c
    return None


def dealii_available() -> bool:
    return find_dealii_dir() is not None and shutil.which("cmake") is not None


def ensure_dealii_built(build_dir: Path | None = None) -> Path:
    """Configure + build the deal.II solver (cached); return the executable."""
    build_dir = Path(build_dir) if build_dir else PAIR_DIR / "build_dealii"
    exe = build_dir / "heat_slab_dealii"
    src_newer = (not exe.exists() or
                 (PAIR_DIR / "heat_slab_dealii.cc").stat().st_mtime
                 > exe.stat().st_mtime)
    if not src_newer:
        return exe
    dealii_dir = find_dealii_dir()
    if dealii_dir is None:
        raise RuntimeError("no deal.II installation found (set DEAL_II_DIR)")
    build_dir.mkdir(parents=True, exist_ok=True)
    for cmd in (["cmake", "-S", str(PAIR_DIR), "-B", str(build_dir),
                 f"-DDEAL_II_DIR={dealii_dir}", "-DCMAKE_BUILD_TYPE=Release"],
                ["make", "-C", str(build_dir), "-j4"]):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"deal.II build step failed: {' '.join(cmd)}\n"
                f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    if not exe.exists():
        raise RuntimeError(f"build succeeded but {exe} is missing")
    return exe


def run_pair(params: dict | None = None, workdir: Path | None = None,
             python_exe: str | None = None, dealii_exe: Path | None = None,
             max_iter: int = 40, tol: float = 1e-8) -> dict:
    """Assemble the two Participants and run the generic coupling driver.

    Returns a report dict: converged, iterations, residual, T_if_computed
    (mean over both participants' interface exports), T_if_analytic,
    T_if_error, and the raw per-participant interface values.
    """
    sys.path.insert(0, str(REPO_SRC))
    from core.coupling_driver import Participant, run_coupling

    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    python_exe = python_exe or sys.executable
    dealii_exe = Path(dealii_exe) if dealii_exe else ensure_dealii_built()

    workdir = Path(workdir) if workdir else Path(
        tempfile.mkdtemp(prefix="dealii_ngsolve_cht_"))
    wd_a = workdir / "dealii"
    wd_b = workdir / "ngsolve"
    wd_a.mkdir(parents=True, exist_ok=True)
    wd_b.mkdir(parents=True, exist_ok=True)

    H1 = p["y_if"] - p["y_bot"]
    H2 = p["y_top"] - p["y_if"]
    T_if_exact = analytic_interface_temperature(
        p["k1"], H1, p["k2"], H2, p["T_bot"], p["T_top"])

    params_a = {
        "k": p["k1"], "x_min": p["x_min"], "x_max": p["x_max"],
        "y_min": p["y_bot"], "y_max": p["y_if"],
        "T_dirichlet": p["T_bot"],
        # neutral initial interface guess: mean of the two boundary values
        "T_interface_init": 0.5 * (p["T_bot"] + p["T_top"]),
        "nx": p["nx"], "ny": p["ny"], "degree": p["degree"],
        "exe": str(dealii_exe), "partner": "ngsolve",
        "ld_library_path": os.environ.get("OASIS_DEALII_LD", ""),
    }
    params_b = {
        "k": p["k2"], "x_min": p["x_min"], "x_max": p["x_max"],
        "y_min": p["y_if"], "y_max": p["y_top"],
        "T_dirichlet": p["T_top"], "flux_init": 0.0,
        "maxh": p["maxh"], "order": p["order"],
        "n_samples": p["n_samples"], "poly_degree": p["poly_degree"],
        "partner": "dealii",
    }
    (wd_a / "params.json").write_text(json.dumps(params_a, indent=2))
    (wd_b / "params.json").write_text(json.dumps(params_b, indent=2))

    participants = [
        Participant("dealii", [python_exe, str(PAIR_DIR / "participant_dealii.py")],
                    wd_a, imports_from=["ngsolve"], timeout=600),
        Participant("ngsolve", [python_exe, str(PAIR_DIR / "participant_ngsolve.py")],
                    wd_b, imports_from=["dealii"], timeout=600),
    ]
    result = run_coupling(participants, max_iter=max_iter, tol=tol)

    report = {
        "converged": result.converged,
        "iterations": result.iterations,
        "residual": result.residual,
        "error": result.error,
        "warnings": result.warnings,
        "params": p,
        "T_if_analytic": T_if_exact,
        "q_if_analytic": analytic_interface_flux(
            p["k1"], H1, p["k2"], H2, p["T_bot"], p["T_top"]),
        "workdir": str(workdir),
    }
    if result.converged:
        t_a = result.exports["dealii"]["values"]
        t_b = result.exports["ngsolve"]["values"]
        q_a = result.exports["dealii"].get("normal_fluxes", [])
        T_if_computed = 0.5 * (sum(t_a) / len(t_a) + sum(t_b) / len(t_b))
        report.update({
            "T_if_dealii": sum(t_a) / len(t_a),
            "T_if_ngsolve": sum(t_b) / len(t_b),
            "T_if_computed": T_if_computed,
            "T_if_error": abs(T_if_computed - T_if_exact),
            "q_if_dealii": sum(q_a) / len(q_a) if q_a else None,
        })
    return report


def main() -> int:
    report = run_pair()
    print(json.dumps(report, indent=2))
    if not report["converged"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
