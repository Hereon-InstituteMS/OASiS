#!/usr/bin/env python3
"""Generate the BARE-extension document for the Tier-E tasks (HOE-v2).

The 17 original HOE-v1 tasks already have published BARE numbers, so the
v2 main document deliberately omits BARE. The 8 new Tier-E tasks (E1-E8)
have no BARE coverage at all — this script emits a separate manual
prompt document for just those 24 cells (8 tasks × 3 seeds).

BARE invocation: ``claude --mcp-config scripts/bare_mcp_config.json
--strict-mcp-config`` — verified to ignore the user-scope
``~/.claude.json`` entries so the OASiS MCP server is NOT attached.

Workdirs use a ``_v2`` suffix and live alongside the MCP_FULL/NO_PITDB/
NO_CRITIC cells:

    eval_interactive/{TASK}_BARE_seed{N}_v2/work/result.txt

Run from the repo root:

    .venv/bin/python scripts/generate_hoe_v2_bare_ext.py

Output: papers/overleaf-paper/prompts/PROMPTS_HOE_V2_BARE_EXT.md (24 cells).
"""
import os
import subprocess
import sys
from pathlib import Path

BASE = os.path.expanduser("~/Schreibtisch/open-fem-agent")
sys.path.insert(0, os.path.join(BASE, "scripts"))
from generate_hoe_v2_prompts import E_PROMPTS, E_TITLES, NEW_TASKS, PLACEHOLDER  # noqa: E402

OUT_DOC = Path(BASE) / "papers/overleaf-paper/prompts/PROMPTS_HOE_V2_BARE_EXT.md"
BARE_CFG = f"{BASE}/scripts/bare_mcp_config.json"
SEEDS = [0, 1, 2]


def git_state():
    def run(*args):
        return subprocess.run(["git", "-C", BASE, *args],
                              capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "--short", "HEAD"), run("rev-parse", "--abbrev-ref", "HEAD")


def preamble(commit, branch, n_cells):
    return f"""# HOE-v2 BARE Extension — Tier-E ({n_cells} cells)

Repo: `{BASE}` — branch `{branch}`, commit `{commit}`.

**Why this exists.** The HOE-v2 main document (`PROMPTS_HOE_V2.md`)
runs the three MCP conditions only, because the 17 original tasks
(A1-D2) already have published BARE numbers from HOE-v1. The 8 new
Tier-E tasks (E1-E8) have no BARE coverage at all. This file fills
that gap: **{n_cells} cells = 8 tasks × 3 seeds, BARE only.**

## BARE invocation

The bash block in each cell launches `claude` with an empty MCP config
and `--strict-mcp-config`, which guarantees that **no MCP server is
attached** — not even the globally-registered `open-fem-agent` entry
in `~/.claude.json`. The agent therefore has only Claude Code's native
tools (Bash, Read, Write, WebSearch, Agent) and must solve each Tier-E
task from first principles, with no OASiS catalog, no pitfall DB, no
example finder, no coupling helper, no critic prompt.

If `scripts/bare_mcp_config.json` does not exist, create it with the
single line `{{"mcpServers": {{}}}}` before running these cells.

## Workflow

Same as the main document: fresh terminal per cell, paste the bash
block, paste the prompt block, wait, close. Workdirs use the same
`_v2` suffix so v1 artefacts are never overwritten and the BARE
results sit alongside the three MCP conditions of the same task.

Grading uses the same Tier-E bands in the appendix of
`PROMPTS_HOE_V2.md` — gates and fabrication audit apply identically.

---
"""


def main():
    os.chdir(BASE)
    commit, branch = git_state()
    n_cells = len(NEW_TASKS) * len(SEEDS)
    parts = [preamble(commit, branch, n_cells)]
    n = 0
    for task in NEW_TASKS:
        for seed in SEEDS:
            n += 1
            cell = f"{task}_BARE_seed{seed}_v2"
            out = f"{BASE}/eval_interactive/{cell}/work/result.txt"
            bash = (f"cd {BASE}\n"
                    f"claude --mcp-config {BARE_CFG} --strict-mcp-config")
            title_extra = (f" — {E_TITLES[task]}" if seed == 0 else "")
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
{E_PROMPTS[task].replace(PLACEHOLDER, out)}
```

---
""")
    OUT_DOC.write_text("".join(parts))
    text = OUT_DOC.read_text()
    assert PLACEHOLDER not in text
    assert text.count("## Cell ") == n_cells
    # ≥ n_cells (preamble also mentions the flag)
    assert text.count("--strict-mcp-config") >= n_cells
    print(f"Wrote {OUT_DOC} ({n} cells, "
          f"{len(text.splitlines())} lines)")
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()
