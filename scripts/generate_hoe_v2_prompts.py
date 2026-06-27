#!/usr/bin/env python3
"""Generate the HOE-v2 manual prompt document (PROMPTS_HOE_V2.md).

225 cells = 25 tasks (A1-A5, B1-B5, C1-C5, D1-D2, E1-E8)
          x 3 MCP conditions (MCP_FULL, MCP_NO_PITFALL_DB, MCP_NO_CRITIC)
          x 3 seeds (0-2).
No BARE condition in v2.

The 17 original task prompts are extracted verbatim from the tracked HOE-v1
document (papers/overleaf-paper/prompts/PROMPTS.md), taking each task's
MCP_FULL_seed0 cell as the canonical text and replacing the embedded result
path with a placeholder. The 8 new Tier-E prompts are defined inline below.

Run from the repo root:

    .venv/bin/python scripts/generate_hoe_v2_prompts.py

Output: papers/overleaf-paper/prompts/PROMPTS_HOE_V2.md  (gitignored — the
papers/ tree is local-only except files tracked before the ignore rule).
"""
import os
import re
import subprocess
import sys
from pathlib import Path

BASE = os.path.expanduser("~/Schreibtisch/open-fem-agent")
V1_DOC = Path(BASE) / "papers/overleaf-paper/prompts/PROMPTS.md"
OUT_DOC = Path(BASE) / "papers/overleaf-paper/prompts/PROMPTS_HOE_V2.md"

ORIGINAL_TASKS = ["A1", "A2", "A3", "A4", "A5",
                  "B1", "B2", "B3", "B4", "B5",
                  "C1", "C2", "C3", "C4", "C5",
                  "D1", "D2"]
NEW_TASKS = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]
ALL_TASKS = ORIGINAL_TASKS + NEW_TASKS

CONDITIONS = [
    ("MCP_FULL", []),
    ("MCP_NO_PITFALL_DB", ["export OFA_DISABLE_PITFALLS=1"]),
    ("MCP_NO_CRITIC", ["export OFA_DISABLE_CRITIC=1"]),
]
SEEDS = [0, 1, 2]

PLACEHOLDER = "{OUT}"


def git_state():
    def run(*args):
        return subprocess.run(["git", "-C", BASE, *args],
                              capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "--short", "HEAD"), run("rev-parse", "--abbrev-ref", "HEAD")


def extract_v1_prompts():
    """Pull the canonical prompt text per original task from PROMPTS.md."""
    lines = V1_DOC.read_text().splitlines()
    prompts = {}
    for task in ORIGINAL_TASKS:
        cell = f"{task}_MCP_FULL_seed0"
        old_path = (f"/home/alexander/Schreibtisch/open-fem-agent/"
                    f"eval_interactive/{cell}/work/result.txt")
        # locate cell header
        idx = next(i for i, l in enumerate(lines)
                   if l.startswith("## Cell ") and l.endswith(f"— {cell}"))
        # find the prompt fence: the fenced block after "Then paste this prompt"
        j = idx
        while "Then paste this prompt" not in lines[j]:
            j += 1
        while not lines[j].startswith("```"):
            j += 1
        j += 1
        block = []
        while not lines[j].startswith("```"):
            block.append(lines[j])
            j += 1
        text = "\n".join(block)
        assert old_path in text, f"{task}: result path not found in prompt"
        prompts[task] = text.replace(old_path, PLACEHOLDER)
    return prompts


# ---------------------------------------------------------------------------
# Tier E — new adversarial / cross-code / research-level tasks
# ---------------------------------------------------------------------------

E_PROMPTS = {
"E1": """Adversarial cross-code validation task. Compute the tip deflection of a
slender 2D cantilever under PLANE STRAIN conditions with TWO independent
codes: FEniCSx (dolfinx) and 4C Multiphysics.

Setup (identical for both codes):
- Geometry: rectangle [0,10] x [0,1] (length L=10, height H=1)
- Material: linear elasticity, E = 1000, nu = 0.3, PLANE STRAIN
- BCs: left edge (x=0) fully clamped (u = 0); right edge (x=10) loaded with a
  uniform vertical surface traction t = (0, -1.0) (total load P = -1.0);
  top and bottom edges traction-free
- Discretization: quadratic displacement elements, at least 100 x 10 elements
  (or equivalent resolution)

Report the vertical displacement u_y at the tip mid-point (x=10, y=0.5) from
both codes and their relative difference |a-b|/max(|a|,|b|). Write exactly
three lines to {OUT}:
RESULT uy_fenics = <value>
RESULT uy_4c = <value>
RESULT rel_diff = <value>""",

"E2": """Cross-code 3D convergence study with a manufactured solution.

Solve 3D linear elasticity (E = 1.0, nu = 0.3) on the unit cube [0,1]^3 with
homogeneous Dirichlet BCs u = 0 on the entire boundary and the body force
derived from the manufactured displacement field
  u1 = sin(pi x) sin(pi y) sin(pi z)
  u2 = x (1-x) y (1-y) z (1-z)
  u3 = sin(pi x) y (1-y) sin(pi z)
Derive the body force f = -div(sigma(u_exact)) symbolically — do not
approximate it numerically.

Run the SAME study in TWO codes:
- FEniCSx (dolfinx): P2 tetrahedral elements
- deal.II: Q2 hexahedral elements
For each code use meshes with N = 4, 8, 16 cells per edge, compute the L2
displacement error against u_exact for each N, and the observed convergence
rates between consecutive meshes.

Write to {OUT}:
RESULT rates_fenics = [<rate_4to8>, <rate_8to16>]
RESULT rates_dealii = [<rate_4to8>, <rate_8to16>]
RESULT e_fenics_N16 = <value>
RESULT e_dealii_N16 = <value>""",

"E3": """Cross-code domain-decomposition coupling task.

Solve -Laplacian(u) = 2 pi^2 sin(pi x) sin(pi y) on the unit square [0,1]^2
with u = 0 on the whole boundary (exact solution u = sin(pi x) sin(pi y)) by
Dirichlet-Neumann domain decomposition across the line x = 0.5, using TWO
DIFFERENT codes:
- Subdomain A = [0, 0.5] x [0, 1]: NGSolve
- Subdomain B = [0.5, 1] x [0, 1]: scikit-fem
Iterate the DN coupling (with relaxation if needed) until the relative L2
mismatch of the interface traces between the two sides is below 1e-8.
Use mesh size h <= 1/64 in both subdomains with matching interface nodes (or
a verified interpolation between the two interface discretizations).

Write to {OUT}:
RESULT iface_rel_l2_mismatch = <final interface mismatch>
RESULT err_A = <relative L2 error of subdomain-A solution vs exact>
RESULT err_B = <relative L2 error of subdomain-B solution vs exact>
RESULT n_dd_iters = <number of DN iterations>""",

"E4": """Two-stage one-way coupled thermo-mechanical chain across two codes with a
mandatory file-based field transfer.

Stage 1 (FEniCSx / dolfinx): solve steady heat conduction
-div(k grad T) = 0 with k = 1 on the strip [0,1] x [0,0.1], T = 0 at x = 0,
T = 100 at x = 1, insulated top and bottom. Write the temperature field to a
VTU/XDMF file.

Stage 2 (scikit-fem): READ the temperature field from the Stage-1 file and
interpolate it onto the mechanical mesh — do NOT re-derive the temperature
analytically and do NOT re-solve the heat problem in scikit-fem. Then solve
PLANE STRESS linear thermo-elasticity on the same strip:
E = 1000, nu = 0.3, thermal expansion alpha = 1e-5, reference temperature
T_ref = 0. BCs: u_x = 0 on both ends (x = 0 and x = 1), u_y = 0 pinned at the
single corner point (0,0); top and bottom traction-free.

Report the stress sigma_xx at the center point (0.5, 0.05).
Write to {OUT}:
RESULT sigma_xx_center = <value>""",

"E5": """Lid-driven cavity benchmark at Re = 1000 (Ghia et al. 1982 reference).

Solve the steady incompressible Navier-Stokes equations in the unit square
cavity with FEniCSx (dolfinx): lid velocity u = (1, 0) on the top boundary
y = 1, no-slip on the other three walls, nu = 0.001 (Re = 1000 with lid
speed 1 and cavity size 1). Use Taylor-Hood (P2/P1) elements on a mesh of at
least 128 x 128 cells and solve with Newton's method (use continuation in
Reynolds number if needed for convergence). Validate against the Ghia et al.
benchmark profiles.

Report:
- the minimum of u_x along the vertical centerline x = 0.5
- the minimum and maximum of u_y along the horizontal centerline y = 0.5
Write to {OUT}:
RESULT ux_min_vertical_centerline = <value>
RESULT uy_min_horizontal_centerline = <value>
RESULT uy_max_horizontal_centerline = <value>""",

"E6": """DFG flow-around-cylinder benchmark 2D-1 (Schaefer-Turek, steady, Re = 20).

Solve the steady incompressible Navier-Stokes equations with NGSolve in the
channel [0, 2.2] x [0, 0.41] containing a circular cylinder of diameter
D = 0.1 centered at (0.2, 0.2). Parabolic inflow at x = 0 with maximum
velocity U_m = 0.3: u_x(0, y) = 4 U_m y (0.41 - y) / 0.41^2. No-slip on the
channel walls and the cylinder, natural (do-nothing) outflow at x = 2.2.
Kinematic viscosity nu = 0.001, density rho = 1 (mean inflow velocity
U_bar = 0.2, Re = U_bar D / nu = 20). Use Taylor-Hood elements of order >= 2
with a curved, well-refined mesh near the cylinder.

Compute the drag and lift coefficients of the cylinder,
c = 2 F / (rho U_bar^2 D), and the pressure difference
delta_p = p(0.15, 0.2) - p(0.25, 0.2).
Write to {OUT}:
RESULT cd = <value>
RESULT cl = <value>
RESULT delta_p = <value>""",

"E7": """Convection-dominated transport with a sharp boundary layer (deal.II).

Solve -eps Laplacian(u) + b . grad(u) = 0 with eps = 0.001 and b = (1, 0) on
the unit square [0,1]^2, with u = 0 at x = 0, u = 1 at x = 1 and homogeneous
Neumann conditions on y = 0 and y = 1. The exact solution is the 1D
boundary-layer profile u(x, y) = (exp(x/eps) - 1) / (exp(1/eps) - 1).

Use deal.II with bilinear (Q1) elements on a uniform 64 x 64 mesh. Plain
Galerkin is unstable at this mesh Peclet number — use a stabilized
discretization (SUPG / streamline diffusion, or an upwind DG scheme) so that
the discrete solution is non-oscillatory.

Report the L2 error against the exact solution and the extrema of the
discrete solution. Write to {OUT}:
RESULT l2_error = <value>
RESULT u_min = <value>
RESULT u_max = <value>""",

"E8": """Cross-code eigenvalue benchmark on a curved domain.

Compute the first three eigenvalues of the Dirichlet Laplacian
(-Laplacian(u) = lambda u, u = 0 on the boundary) on the UNIT DISK with TWO
independent codes: NGSolve and deal.II. The exact eigenvalues are squares of
Bessel-function zeros: lambda_1 = j_{0,1}^2 = 5.78319, and the double
eigenvalue lambda_2 = lambda_3 = j_{1,1}^2 = 14.68197. Use a curved boundary
representation (curved/mapped elements or a sufficiently fine boundary
approximation) and a discretization fine enough that each computed eigenvalue
is within 0.5% of the exact value.

Write to {OUT}:
RESULT lambda_ngsolve = [<l1>, <l2>, <l3>]
RESULT lambda_dealii = [<l1>, <l2>, <l3>]""",
}

E_TITLES = {
    "E1": "Plane-strain trap, FEniCSx vs 4C cantilever (cross-code, adversarial)",
    "E2": "3D elasticity MMS convergence, FEniCSx vs deal.II (cross-code)",
    "E3": "Dirichlet-Neumann DD coupling, NGSolve <-> scikit-fem",
    "E4": "Thermo-mechanical chain with file-based field transfer, FEniCSx -> scikit-fem",
    "E5": "Lid-driven cavity Re=1000 (Ghia), FEniCSx",
    "E6": "DFG 2D-1 cylinder benchmark (Schaefer-Turek), NGSolve",
    "E7": "Convection-dominated boundary layer with stabilization, deal.II",
    "E8": "Unit-disk Dirichlet eigenvalues, NGSolve vs deal.II (cross-code)",
}

GRADING = """\
## Appendix — Grading bands (all gates must pass; result.txt must exist with the exact RESULT lines)

### Fabrication audit (applies to every cell)

A cell passes only if, in addition to the numeric gates:
1. the cell's work dir contains the runnable script(s) and genuine solver
   output artifacts (VTU/XDMF/logs/control files) that produce the reported
   numbers, and
2. the reported values match what those artifacts/scripts actually yield.

For tasks whose reference values are famous and likely memorized from the
literature (B3, B4, E5, E6, E8), additionally re-run one stored script per
task during grading and confirm it reproduces the result.txt values.
Values that cannot be traced to an actual run are graded FAIL with
failure_code = fabricated, regardless of numeric correctness.

### Tier A
| Task | Gate |
|------|------|
| A1 | `u_at_r06` within 1% of 0.59322 |
| A2 | tip deflection in [-44, -36] |
| A3 | `u` within 2% of 0.37271 |
| A4 | errors strictly decreasing; observed rates 2.0 +/- 0.25 |
| A5 | `f1` in [85.2, 94.1] Hz |

### Tier B
| Task | Gate |
|------|------|
| B1 | consensus plausibility: wss > 0, delta_p > 0, magnitudes consistent across passing runs |
| B2 | both codes tip in [3.45, 4.22] mm (ref 3.836 mm) AND cross-code agreement <= 5% |
| B3 | Strouhal St in [0.18, 0.21] |
| B4 | contact radius a in [0.30, 0.60] mm AND Hertz self-consistency p_max = E*.a/(2R) within 20% (E* = 219.78 GPa) |
| B5 | consensus plausibility; max pressure amplification in [1, 3]; peak near 0 deg / 180 deg |

### Tier C
| Task | Gate |
|------|------|
| C1 | rates 2.0 +/- 0.25, errors decreasing |
| C2 | rates 2.0 +/- 0.25, errors decreasing |
| C3 | time rates 1.0 +/- 0.2, space rates 2.0 +/- 0.25 |
| C4 | velocity rate 3.0 +/- 0.3, pressure rate 2.0 +/- 0.3 |
| C5 | space rates 2.0 +/- 0.25, time rates 2.0 +/- 0.3, both species |

### Tier D
| Task | Gate |
|------|------|
| D1 | rates 2.0 +/- 0.3 AND reported build path exists |
| D2 | tip_dx > 0, mean_fsi_iters in [2, 30], consensus plausibility |

### Tier E (new in v2)
| Task | Gate |
|------|------|
| E1 | `uy_fenics` AND `uy_4c` in [-3.90, -3.45] (plane-strain ref -3.6712; plane-stress trap value ~ -4.03 fails) AND `rel_diff` <= 0.01 |
| E2 | all four rate entries in [2.7, 3.3]; `e_fenics_N16` and `e_dealii_N16` within a factor of 3 of each other |
| E3 | `iface_rel_l2_mismatch` <= 1e-3, `err_A` <= 2e-3, `err_B` <= 2e-3, `n_dd_iters` in [2, 200] |
| E4 | `sigma_xx_center` in [-0.56, -0.44] (exact -0.5; plane-strain mistake -0.714 fails; FAIL if the transcript shows the temperature was re-derived analytically instead of read from the Stage-1 file) |
| E5 | `ux_min_vertical_centerline` in [-0.40, -0.365], `uy_min_horizontal_centerline` in [-0.54, -0.50], `uy_max_horizontal_centerline` in [0.36, 0.39] (Ghia: -0.3829 / -0.5155 / 0.3709) |
| E6 | `cd` in [5.52, 5.64], `cl` in [0.0090, 0.0125], `delta_p` in [0.1150, 0.1200] (ref: 5.57954 / 0.010618 / 0.11752) |
| E7 | `l2_error` <= 0.15, `u_min` >= -0.02, `u_max` <= 1.02 |
| E8 | each of the six eigenvalues within 0.5% of exact (5.78319, 14.68197, 14.68197) AND cross-code disagreement per eigenvalue <= 0.5% |
"""


def preamble(commit, branch, n_cells):
    return f"""# HOE-v2 Manual Prompts ({n_cells} cells)

Repo: `{BASE}` — branch `{branch}`, commit `{commit}`.

**Scope.** Full rerun of the 17 HOE-v1 tasks plus 8 new Tier-E tasks
(adversarial / cross-code / research-level), under the three MCP conditions
only (no BARE in v2): {n_cells} cells = 25 tasks x 3 conditions x 3 seeds.

## Conditions

| Condition | Env var before `claude` | What is masked |
|-----------|------------------------|----------------|
| MCP_FULL | (none) | nothing |
| MCP_NO_PITFALL_DB | `export OFA_DISABLE_PITFALLS=1` | per-backend pitfalls incl. Signal anchors, post-mortems, cross-backend collation catalog, general input-format pitfalls |
| MCP_NO_CRITIC | `export OFA_DISABLE_CRITIC=1` | MANDATORY CRITIC paragraph in server instructions |

Each fresh `claude` session spawns its own MCP-server process, so the
exported variable propagates to the server. Never run two conditions in the
same terminal.

## Pre-flight (run once before the campaign, and after any server change)

```bash
cd {BASE}
.venv/bin/python scripts/verify_hoe_ablation.py   # must print three "ok" lines
.venv/bin/pytest -q                               # 499 passed expected
```

Note: failures in `tests/test_signal_verification.py::TestBackendImportSnapshot`
are expected on this machine (snapshots reference dev-machine library paths,
e.g. `~/Schreibtisch/dealii-debug/`); they are environment drift, not server
defects. The `.venv` was repaired on 2026-06-12 (re-pointed to the local
cpython-3.12.13 copy after a snap revision GC broke the old symlinks).

## How to use

For each cell below:
1. Open a fresh terminal.
2. Paste the cell's bash block (it does the `cd`, the export if any, and starts `claude`).
3. Paste the prompt block into claude.
4. Wait for completion. The result.txt lands at the path embedded in the prompt.
5. Close the terminal. Next cell.

Workdirs carry a `_v2` suffix (`eval_interactive/<TASK>_<COND>_seed<N>_v2/work/`)
so v1 artefacts are never overwritten. Grading bands are in the appendix at
the end of this file.

---
"""


def main():
    os.chdir(BASE)
    commit, branch = git_state()
    prompts = extract_v1_prompts()
    prompts.update(E_PROMPTS)

    n_cells = len(ALL_TASKS) * len(CONDITIONS) * len(SEEDS)
    parts = [preamble(commit, branch, n_cells)]
    n = 0
    for task in ALL_TASKS:
        for cond, exports in CONDITIONS:
            for seed in SEEDS:
                n += 1
                cell = f"{task}_{cond}_seed{seed}_v2"
                out = f"{BASE}/eval_interactive/{cell}/work/result.txt"
                bash = "\n".join([f"cd {BASE}", *exports, "claude"])
                title_extra = (f" — {E_TITLES[task]}" if task in E_TITLES
                               and cond == "MCP_FULL" and seed == 0 else "")
                parts.append(f"""
## Cell {n}/{n_cells} — {cell}{title_extra}

- **Task:** {task} | **Condition:** {cond} | **Seed:** {seed}
- **Result will land at:** `{out}`

**Where to run — open a fresh terminal and paste this block:**

```bash
{bash}
```

**Then paste this prompt into claude:**

```
{prompts[task].replace(PLACEHOLDER, out)}
```

---
""")
    parts.append("\n" + GRADING)
    OUT_DOC.write_text("".join(parts))
    print(f"Wrote {OUT_DOC} ({n} cells, "
          f"{len(OUT_DOC.read_text().splitlines())} lines)")
    # sanity
    text = OUT_DOC.read_text()
    assert PLACEHOLDER not in text, "unreplaced placeholder!"
    assert text.count("## Cell ") == n_cells
    # +1 for the occurrence in the preamble conditions table
    assert text.count("export OFA_DISABLE_PITFALLS=1") == len(ALL_TASKS) * 3 + 1
    assert text.count("export OFA_DISABLE_CRITIC=1") == len(ALL_TASKS) * 3 + 1
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
