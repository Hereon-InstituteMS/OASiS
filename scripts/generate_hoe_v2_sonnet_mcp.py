#!/usr/bin/env python3
"""Generate Sonnet-4.6 MCP_FULL manual-paste document (HOE-v2 family).

75 cells = 25 tasks (A1-A5, B1-B5, C1-C5, D1, D2, E1-E8) × 3 seeds. All
under MCP_FULL — the open-fem-agent MCP server is registered globally in
``~/.claude.json`` and stays attached (no ``--mcp-config`` override).

Distinguishing feature: ``claude --model claude-sonnet-4-6`` instead of
the default Opus. Workdirs use the ``SONNET`` prefix to keep them
separated from the Opus 4.7 runs:

    eval_interactive/<TASK>_SONNET_MCP_FULL_seed<N>_v2/work/result.txt

Why manual paste: Sonnet (like Opus) can spawn sub-agents via the Agent
tool only when running interactively in Claude Code, not in headless
``-p`` mode. Manual paste preserves the MANDATORY CRITIC capability that
the OASiS server prompts for.

Grading: same gates and amendments as the Opus runs
(``scripts/grade_hoe_v2.py``). The grader regex was extended to accept
the ``SONNET`` model prefix so all four conditions × two models get
graded uniformly with no per-model drift.

Run from the repo root:

    .venv/bin/python scripts/generate_hoe_v2_sonnet_mcp.py

Output: papers/overleaf-paper/prompts/PROMPTS_HOE_V2_SONNET_MCP.md.
"""
import os
import subprocess
import sys
from pathlib import Path

BASE = os.path.expanduser("~/Schreibtisch/open-fem-agent")
sys.path.insert(0, os.path.join(BASE, "scripts"))
from generate_hoe_v2_prompts import (  # noqa: E402
    ORIGINAL_TASKS, NEW_TASKS, E_PROMPTS, E_TITLES,
    extract_v1_prompts, PLACEHOLDER)

OUT_DOC = Path(BASE) / "papers/overleaf-paper/prompts/PROMPTS_HOE_V2_SONNET_MCP.md"
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
    **E_TITLES,
}

ALL_TASKS = ORIGINAL_TASKS + NEW_TASKS


def git_state():
    def run(*args):
        return subprocess.run(["git", "-C", BASE, *args],
                              capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "--short", "HEAD"), run("rev-parse", "--abbrev-ref", "HEAD")


def preamble(commit, branch, n_cells):
    return f"""# HOE-v2 Sonnet-4.6 MCP_FULL ({n_cells} cells)

Repo: `{BASE}` — branch `{branch}`, commit `{commit}`.

**Why this exists.** The v2 campaign so far ran on Opus 4.7. On that
strong model the 25-task suite is too easy: BARE = 73/75 = 97.3 %,
MCP_FULL = 73/75 = 97.3 % — statistically tied. To test the OASiS uplift
claim cleanly we need a comparison where the base model is weaker, so
the OASiS scaffolding has room to make a measurable difference.

This file runs all 25 tasks × 3 seeds under **Sonnet 4.6** with the full
OASiS MCP server attached. Together with the existing Opus-4.7 BARE
column (73/75), it tests the headline hypothesis:

  **Sonnet 4.6 + OASiS ≈ Opus 4.7 BARE**

If yes, the contribution becomes "OASiS lets a cheaper model perform
like a more expensive one" — a sharper, less-erodable claim than the
original "78 % → 94 % uplift on the same model."

## Invocation

Each cell launches `claude` with the Sonnet model flag. The OASiS MCP
server stays attached via the standard `open-fem-agent` registration in
`~/.claude.json` — **no** `--mcp-config` override, **no**
`--strict-mcp-config`. The OFA_DISABLE_* ablation env vars are NOT set,
so the agent gets the full pitfall DB, critic instruction, cross-backend
collation catalog — everything.

Manual paste (interactive Claude Code) so the agent can spawn sub-agents
via its Agent tool, as the MANDATORY CRITIC paragraph in the OASiS
server instructions asks for. Headless `-p` mode blocks Agent.

## Workflow

Fresh terminal per cell. Paste the bash block, paste the prompt, wait,
close the terminal. Workdirs use the `SONNET` prefix so the runs sit
alongside (not overwrite) the existing Opus cells:

    eval_interactive/<TASK>_SONNET_MCP_FULL_seed<N>_v2/work/result.txt

## Grading

Same gates and amendments as the Opus runs — handled by
`scripts/grade_hoe_v2.py`. The grader regex was extended to accept the
`SONNET` model prefix; no rule drift between models or conditions.

---
"""


def main():
    os.chdir(BASE)
    commit, branch = git_state()
    prompts = extract_v1_prompts()
    prompts.update(E_PROMPTS)

    n_cells = len(ALL_TASKS) * len(SEEDS)
    parts = [preamble(commit, branch, n_cells)]
    n = 0
    for task in ALL_TASKS:
        for seed in SEEDS:
            n += 1
            cell = f"{task}_SONNET_MCP_FULL_seed{seed}_v2"
            out = f"{BASE}/eval_interactive/{cell}/work/result.txt"
            bash = (f"cd {BASE}\n"
                    f"claude --model claude-sonnet-4-6")
            title_extra = f" — {TIER_LABELS.get(task, task)}" if seed == 0 else ""
            parts.append(f"""
## Cell {n}/{n_cells} — {cell}{title_extra}

- **Task:** {task} | **Condition:** MCP_FULL | **Model:** Sonnet 4.6 | **Seed:** {seed}
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
    assert text.count("--model claude-sonnet-4-6") == n_cells
    assert "_SONNET_MCP_FULL_seed" in text
    print(f"Wrote {OUT_DOC} ({n} cells, {len(text.splitlines())} lines)")
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
