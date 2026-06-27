#!/usr/bin/env python3
"""Generate Sonnet-4.6 BARE manual-paste document (HOE-v2 family).

75 cells = 25 tasks × 3 seeds. Same prompts as the Sonnet MCP document,
but `claude` is launched with an empty MCP config and
``--strict-mcp-config`` so no MCP server is attached. Together with
``PROMPTS_HOE_V2_SONNET_MCP.md`` this gives the within-Sonnet uplift
that the paper's main claim needs:

  Sonnet BARE  → low
  Sonnet MCP   → ≈ Opus BARE (≈ 97 %)
  Opus BARE    → ≈ 97 %  (already have)
  Opus MCP     → ≈ 97 %  (already have)

Workdirs use the ``SONNET`` model prefix to keep them separated from
the Opus 4.7 runs:

    eval_interactive/<TASK>_SONNET_BARE_seed<N>_v2/work/result.txt

Run from the repo root:

    .venv/bin/python scripts/generate_hoe_v2_sonnet_bare.py

Output: papers/overleaf-paper/prompts/PROMPTS_HOE_V2_SONNET_BARE.md.
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
from generate_hoe_v2_sonnet_mcp import TIER_LABELS  # noqa: E402

OUT_DOC = Path(BASE) / "papers/overleaf-paper/prompts/PROMPTS_HOE_V2_SONNET_BARE.md"
BARE_CFG = f"{BASE}/scripts/bare_mcp_config.json"
SEEDS = [0, 1, 2]
ALL_TASKS = ORIGINAL_TASKS + NEW_TASKS


def git_state():
    def run(*args):
        return subprocess.run(["git", "-C", BASE, *args],
                              capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "--short", "HEAD"), run("rev-parse", "--abbrev-ref", "HEAD")


def preamble(commit, branch, n_cells):
    return f"""# HOE-v2 Sonnet-4.6 BARE ({n_cells} cells)

Repo: `{BASE}` — branch `{branch}`, commit `{commit}`.

**Why this exists.** Companion to `PROMPTS_HOE_V2_SONNET_MCP.md`. The
paper's main claim is "OASiS lets a cheaper model perform like a more
expensive one." That claim needs three anchors:

| | BARE | MCP_FULL |
|-|------|----------|
| Opus 4.7 | 73/75 = 97.3 % (have) | 73/75 = 97.3 % (have) |
| Sonnet 4.6 | **this file** | `PROMPTS_HOE_V2_SONNET_MCP.md` |

The within-Sonnet contrast (BARE vs MCP) shows the OASiS uplift; the
Sonnet-MCP vs Opus-BARE contrast shows MCP closes the model-size gap.

## BARE invocation

```bash
cd {BASE}
claude --model claude-sonnet-4-6 --mcp-config {BARE_CFG} --strict-mcp-config
```

`--strict-mcp-config` plus the empty `bare_mcp_config.json` makes
`claude` ignore the user-scope `open-fem-agent` registration in
`~/.claude.json`, so no MCP server is attached. The agent has only
Claude Code's native tools (Bash, Read, Write, WebSearch, Agent).

If `scripts/bare_mcp_config.json` is missing, create it with the single
line `{{"mcpServers": {{}}}}` before running these cells.

## Workflow

Fresh terminal per cell. Manual paste so Sonnet can still spawn
sub-agents via its Agent tool (the MANDATORY CRITIC the original prompt
asks for — actually only injected when the MCP server is attached, so
this BARE column tests Sonnet without that nudge).

## Grading

Same gates as the Opus runs and the Sonnet-MCP runs —
`scripts/grade_hoe_v2.py` walks every `*_v2/` directory and applies one
consistent rule set across all 4 model×condition combinations.

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
            cell = f"{task}_SONNET_BARE_seed{seed}_v2"
            out = f"{BASE}/eval_interactive/{cell}/work/result.txt"
            bash = (f"cd {BASE}\n"
                    f"claude --model claude-sonnet-4-6 "
                    f"--mcp-config {BARE_CFG} --strict-mcp-config")
            title_extra = f" — {TIER_LABELS.get(task, task)}" if seed == 0 else ""
            parts.append(f"""
## Cell {n}/{n_cells} — {cell}{title_extra}

- **Task:** {task} | **Condition:** BARE (no MCP) | **Model:** Sonnet 4.6 | **Seed:** {seed}
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
    # ≥ n_cells (preamble has one extra occurrence in its example)
    assert text.count("--model claude-sonnet-4-6") >= n_cells
    print(f"Wrote {OUT_DOC} ({n} cells, {len(text.splitlines())} lines)")
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
