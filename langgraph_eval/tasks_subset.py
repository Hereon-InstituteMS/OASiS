"""Representative subset of the HOE-v2 task suite for the open-weight ablation.

Six tasks chosen to span tiers and capabilities while keeping wall-clock
manageable on locally-served Qwen models:

| ID  | Tier | Why it is in the subset |
|-----|------|-------------------------|
| A1  | A    | Plain Robin-BC Poisson; floor for any solver-aware agent |
| A4  | A    | MMS convergence; tests rate-of-convergence reasoning |
| B2  | B    | Cross-code thermo-mech; tests multi-physics + coupling |
| C2  | C    | 3D elasticity MMS; tests heavier verification workflow |
| E1  | E    | Plane-strain TRAP — adversarial; punishes textbook recall |
| E6  | E    | DFG 2D-1 cylinder benchmark; research-grade target |

Each task carries the prompt verbatim (with a `{OUT}` placeholder that the
driver substitutes per cell) and the grading gate as a small Python callable
that consumes the parsed RESULT lines and returns (passed, reason).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


def _parse(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*RESULT\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", line)
        if not m:
            continue
        name, raw = m.group(1), m.group(2).strip()
        try:
            out[name] = float(raw)
        except ValueError:
            try:
                out[name] = [float(x) for x in raw.strip("[]").split(",")]
            except ValueError:
                out[name] = raw
    return out


@dataclass(frozen=True)
class Task:
    id: str
    prompt: str
    grade: Callable[[dict], tuple[bool, str]]


TASKS: list[Task] = [
    Task(
        id="A1",
        prompt=(
            "Solve the steady Poisson equation -Δu = 0 on the 2D annulus 0.3 ≤ r ≤ 1.0\n"
            "with u = 1 on the inner boundary (r=0.3) and Robin BC ∂u/∂n + 2u = 0 on the\n"
            "outer boundary (r=1.0). Use NGSolve with second-order H1 elements, mesh size\n"
            "h=0.05. Report u at the point (r=0.6, θ=0) to 6 significant digits and write\n"
            "it as the single line `RESULT u_at_r06 = <value>` to {OUT}."
        ),
        grade=lambda r: (
            ("u_at_r06" in r and abs(r["u_at_r06"] - 0.59322) / 0.59322 <= 0.01,
             f"u_at_r06={r.get('u_at_r06')}")),
    ),
    Task(
        id="A4",
        prompt=(
            "Using DUNE-fem, perform an h-convergence study for -Δu = f on [0,1]^2 with\n"
            "the manufactured solution u = sin(πx) sin(πy), homogeneous Dirichlet BCs.\n"
            "Use P1 elements on uniform meshes with N = 8, 16, 32, 64 cells per edge.\n"
            "Compute the L2 error against u_exact for each N and the observed convergence\n"
            "rates between consecutive meshes. Write to {OUT}:\n"
            "RESULT errors = [<e8>, <e16>, <e32>, <e64>]\n"
            "RESULT rates  = [<r8to16>, <r16to32>, <r32to64>]"
        ),
        grade=lambda r: _gate_rates(r, "rates", target=2.0, tol=0.25),
    ),
    Task(
        id="B2",
        prompt=(
            "Cross-code thermo-mechanical (TSI) bimetal strip. Two layers [0,0.1]x[0,0.005]\n"
            "stacked along y, layer 1 below: E1=70 GPa, α1=23e-6/K; layer 2 above:\n"
            "E2=200 GPa, α2=12e-6/K. Both ν=0.3, plane stress. Left edge x=0 clamped,\n"
            "uniform temperature rise ΔT = 100 K, no body forces.\n"
            "Run the SAME problem with deal.II (Q1 hex) AND 4C Multiphysics. Report the\n"
            "tip-mid vertical displacement (x=0.1, y=0.0025) from both and the relative\n"
            "difference. Write to {OUT}:\n"
            "RESULT tip_dealii = <value mm>\n"
            "RESULT tip_4c     = <value mm>\n"
            "RESULT rel_diff   = <value>"
        ),
        grade=lambda r: (
            all(k in r for k in ("tip_dealii", "tip_4c", "rel_diff"))
            and 3.45 <= r["tip_dealii"] <= 4.22
            and 3.45 <= r["tip_4c"] <= 4.22
            and r["rel_diff"] <= 0.05,
            f"tip_dealii={r.get('tip_dealii')}, tip_4c={r.get('tip_4c')}, "
            f"rel_diff={r.get('rel_diff')}"
        ),
    ),
    Task(
        id="C2",
        prompt=(
            "3D linear elasticity (E=1, ν=0.3) on unit cube [0,1]^3, u=0 on entire\n"
            "boundary, body force derived from u_exact = (sin πx sin πy sin πz,\n"
            "x(1-x) y(1-y) z(1-z), sin πx · y(1-y) sin πz). Derive f symbolically.\n"
            "Solve in deal.II with Q2 hex elements, N=4,8,16. Report L2 errors and\n"
            "rates. Write to {OUT}:\n"
            "RESULT errors = [<e4>, <e8>, <e16>]\n"
            "RESULT rates  = [<r4to8>, <r8to16>]"
        ),
        grade=lambda r: _gate_rates(r, "rates", target=3.0, tol=0.3),
    ),
    Task(
        id="E1",
        prompt=(
            "Adversarial cross-code validation task. Compute the tip deflection of a\n"
            "slender 2D cantilever under PLANE STRAIN conditions with TWO independent\n"
            "codes: FEniCSx (dolfinx) and 4C Multiphysics.\n\n"
            "Setup (identical for both codes):\n"
            "- Geometry: rectangle [0,10] x [0,1] (length L=10, height H=1)\n"
            "- Material: linear elasticity, E = 1000, nu = 0.3, PLANE STRAIN\n"
            "- BCs: left edge (x=0) fully clamped (u = 0); right edge (x=10) loaded with a\n"
            "  uniform vertical surface traction t = (0, -1.0) (total load P = -1.0);\n"
            "  top and bottom edges traction-free\n"
            "- Discretization: quadratic displacement elements, at least 100 x 10 elements\n\n"
            "Report the vertical displacement u_y at the tip mid-point (x=10, y=0.5) from\n"
            "both codes and their relative difference |a-b|/max(|a|,|b|). Write exactly\n"
            "three lines to {OUT}:\n"
            "RESULT uy_fenics = <value>\n"
            "RESULT uy_4c = <value>\n"
            "RESULT rel_diff = <value>"
        ),
        grade=lambda r: (
            all(k in r for k in ("uy_fenics", "uy_4c", "rel_diff"))
            and -3.90 <= r["uy_fenics"] <= -3.45
            and -3.90 <= r["uy_4c"] <= -3.45
            and r["rel_diff"] <= 0.01,
            f"uy_fenics={r.get('uy_fenics')}, uy_4c={r.get('uy_4c')}, "
            f"rel_diff={r.get('rel_diff')}"
        ),
    ),
    Task(
        id="E6",
        prompt=(
            "DFG flow-around-cylinder benchmark 2D-1 (Schäfer-Turek, steady, Re=20).\n"
            "Solve the steady incompressible Navier-Stokes equations with NGSolve in the\n"
            "channel [0,2.2] x [0,0.41] containing a circular cylinder of diameter\n"
            "D=0.1 centered at (0.2,0.2). Parabolic inflow u_x(0,y) = 4 U_m y(0.41-y)/0.41^2\n"
            "with U_m=0.3. No-slip on walls and cylinder. nu=0.001, rho=1.\n"
            "Taylor-Hood order >= 2, curved mesh refined near the cylinder.\n"
            "Compute drag and lift coefficients c = 2F/(rho U_bar^2 D) with U_bar=0.2, and\n"
            "delta_p = p(0.15,0.2) - p(0.25,0.2). Write to {OUT}:\n"
            "RESULT cd = <value>\n"
            "RESULT cl = <value>\n"
            "RESULT delta_p = <value>"
        ),
        grade=lambda r: (
            all(k in r for k in ("cd", "cl", "delta_p"))
            and 5.52 <= r["cd"] <= 5.64
            and 0.0090 <= r["cl"] <= 0.0125
            and 0.1150 <= r["delta_p"] <= 0.1200,
            f"cd={r.get('cd')}, cl={r.get('cl')}, delta_p={r.get('delta_p')}"
        ),
    ),
]


def _gate_rates(r: dict, key: str, target: float, tol: float) -> tuple[bool, str]:
    v = r.get(key)
    if not isinstance(v, list) or not v:
        return False, f"{key} missing or not a list: {v}"
    ok = all(abs(x - target) <= tol for x in v)
    return ok, f"{key}={v} (target {target}±{tol})"


__all__ = ["Task", "TASKS", "_parse"]
