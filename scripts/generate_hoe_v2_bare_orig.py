#!/usr/bin/env python3
"""Generate the BARE document for the 17 ORIGINAL HOE-v1 tasks under v2.

We have BARE for Tier-E (PROMPTS_HOE_V2_BARE_EXT.md, 24 cells). This
fills the remaining gap so the v2 BARE column is complete: 17 tasks
(A1-A5, B1-B5, C1-C5, D1, D2) × 3 seeds = 51 cells, BARE only, with
the same `claude --mcp-config <empty> --strict-mcp-config` invocation
verified to suppress all MCP servers.

Together with PROMPTS_HOE_V2.md (225 main MCP cells) and
PROMPTS_HOE_V2_BARE_EXT.md (24 Tier-E BARE cells), the campaign becomes
225 + 24 + 51 = 300 cells: a full 4-condition (BARE / NO_PITDB /
NO_CRITIC / FULL) × 25 tasks × 3 seeds matrix.

Workdirs use the same `_v2` suffix as the rest of the campaign so the
BARE-orig results sit alongside the existing MCP cells of the same task.

Run from the repo root:

    .venv/bin/python scripts/generate_hoe_v2_bare_orig.py

Output: papers/overleaf-paper/prompts/PROMPTS_HOE_V2_BARE_ORIG.md (51 cells).
"""
import os
import subprocess
import sys
from pathlib import Path

BASE = os.path.expanduser("~/Schreibtisch/open-fem-agent")
sys.path.insert(0, os.path.join(BASE, "scripts"))
from generate_hoe_v2_prompts import (  # noqa: E402
    ORIGINAL_TASKS, extract_v1_prompts, PLACEHOLDER)

OUT_DOC = Path(BASE) / "papers/overleaf-paper/prompts/PROMPTS_HOE_V2_BARE_ORIG.md"
BARE_CFG = f"{BASE}/scripts/bare_mcp_config.json"
SEEDS = [0, 1, 2]

TIER_LABELS = {
    "A1": "annulus Poisson + Robin BC (NGSolve)",
    "A2": "plane-stress cantilever (scikit-fem)",
    "A3": "1D transient heat (FEniCSx)",
    "A4": "MMS Poisson convergence (DUNE-fem)",
    "A5": "clamped plate first eigenfrequency (Kratos)",
    "B1": "Stokes converging-diverging channel (FEniCSx)",
    "B2": "bimetallic strip TSI cross-code (deal.II + 4C)",
    "B3": "cylinder wake Strouhal Re=200 (NGSolve)",
    "B4": "Hertzian contact (Kratos)",
    "B5": "Helmholtz acoustic scattering (FEniCSx)",
    "C1": "P1 Poisson MMS (FEniCSx)",
    "C2": "Q1 3D elasticity MMS (deal.II)",
    "C3": "heat-equation MMS time + space (NGSolve)",
    "C4": "Taylor-Hood Stokes MMS (scikit-fem)",
    "C5": "reaction-diffusion MMS u/v (DUNE-fem)",
    "D1": "deal.II step-7 modification + convergence",
    "D2": "partitioned FSI 4C+scikit-fem",
}


def git_state():
    def run(*args):
        return subprocess.run(["git", "-C", BASE, *args],
                              capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "--short", "HEAD"), run("rev-parse", "--abbrev-ref", "HEAD")


def preamble(commit, branch, n_cells):
    return f"""# HOE-v2 BARE Originals — Tier A/B/C/D ({n_cells} cells)

Repo: `{BASE}` — branch `{branch}`, commit `{commit}`.

**Why this exists.** The v2 main document (`PROMPTS_HOE_V2.md`, 225 cells)
runs only the three MCP conditions. `PROMPTS_HOE_V2_BARE_EXT.md` (24 cells)
adds BARE for the eight new Tier-E tasks. This file fills the last gap:
the 17 ORIGINAL HOE-v1 tasks (A1-A5, B1-B5, C1-C5, D1, D2) under BARE,
so the v2 BARE column is complete and an honest v2 BARE-vs-MCP_FULL uplift
can be measured on apples-to-apples cells (same model, same prompts,
same machine).

**{n_cells} cells = 17 tasks × 3 seeds, BARE only.**

## BARE invocation

```bash
cd {BASE}
claude --mcp-config {BARE_CFG} --strict-mcp-config
```

Verified (claude-code-guide): `--strict-mcp-config` plus an empty
`{{"mcpServers": {{}}}}` config makes `claude` ignore both user-scope
(`~/.claude.json`) and project-scope (`.mcp.json`) MCP registrations,
so the `open-fem-agent` server is **not** loaded. The agent has only
Claude Code's native tools (Bash, Read, Write, WebSearch, Agent).

If `scripts/bare_mcp_config.json` does not exist, create it with the
single line `{{"mcpServers": {{}}}}` before running these cells.

## Workflow

Same as the other v2 documents: fresh terminal per cell, paste the
bash block, paste the prompt block, wait, close. Workdirs keep the
`_v2` suffix so the BARE results sit alongside the three MCP conditions
of the same task.

## Grading

Use the SAME gates and amendments applied to the MCP cells
(`scripts/grade_hoe_v2.py`); the grader walks all `*_BARE_seed*_v2/`
directories automatically and writes pass/fail to
`papers/overleaf-paper/data/v2_grades.csv`.

---
"""


def main():
    os.chdir(BASE)
    commit, branch = git_state()
    prompts = extract_v1_prompts()
    n_cells = len(ORIGINAL_TASKS) * len(SEEDS)
    parts = [preamble(commit, branch, n_cells)]
    n = 0
    for task in ORIGINAL_TASKS:
        for seed in SEEDS:
            n += 1
            cell = f"{task}_BARE_seed{seed}_v2"
            out = f"{BASE}/eval_interactive/{cell}/work/result.txt"
            bash = (f"cd {BASE}\n"
                    f"claude --mcp-config {BARE_CFG} --strict-mcp-config")
            title_extra = f" — {TIER_LABELS[task]}" if seed == 0 else ""
            parts.append(f"""
## Cell {n}/{n_cells} — {cell}{title_extra}

- **Task:** {task} | **Condition:** BARE (no MCP) | **Seed:** {seed}
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
    OUT_DOC.write_text("".join(parts))
    text = OUT_DOC.read_text()
    assert PLACEHOLDER not in text
    assert text.count("## Cell ") == n_cells
    assert text.count("--strict-mcp-config") >= n_cells
    print(f"Wrote {OUT_DOC} ({n} cells, {len(text.splitlines())} lines)")
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
