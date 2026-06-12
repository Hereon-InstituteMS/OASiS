# LAUNCH_PROMPTS.md — kickoff prompts for the three OASiS runs

Each run below needs **its own fresh terminal** running Claude Code interactively (not `claude --print`). Sub-agents (`Agent` tool) only work in interactive sessions.

For every block:
1. Open a new terminal.
2. `cd ~/Schreibtisch/Open-FEM-agent`
3. Start Claude Code: `claude`
4. Paste the prompt block verbatim. Walk away. Come back to results.

The three blocks are **independent** — they can run in parallel in separate terminals, or sequentially. (B) is the highest priority because it tells us whether the recent catalog work moved the needle.

---

## Run (A) — Fix broken templates (tasks #27 / #28 / #29 / #30)

**What this does:** Identifies every `prepare_simulation()` template that fails when run, fixes the catalog/template, and verifies via the live execution gate. Doesn't strictly need sub-agents (single Claude can do it serially), but interactive is cleaner.

**Estimated wall-time:** 30–90 min depending on how many breakages exist.

**Paste this:**

```
We are working on the OASiS MCP at ~/Schreibtisch/Open-FEM-agent (this directory). Goal: close pending tasks #27, #28, #29 (NGSolve / FEniCSx / 4C failing templates). Task #30 (deal.II rebuild) requires a heavy rebuild; SKIP it unless I tell you otherwise.

For each backend in (ngsolve, fenics, fourc):
  1. Read scripts/scan_results/tier2_results.json and find all rows where backend matches and "captured_head" looks like a real Python traceback or 4C error — these are the broken templates the catalog returns when a user calls prepare_simulation().
  2. For each broken row: locate the corresponding generator in src/backends/<backend>/generators/<physics>.py, read its template literal, identify the bug (wrong API, retired symbol, wrong tag name, etc.) using the live solver in the venv (.venv/bin/python for kratos/skfem; for fenics use the conda env ~/miniconda3/envs/ofa-fenicsx/bin/python; for fourc use the binary in $FOURC_BINARY).
  3. Fix the template. Re-run via the corresponding fixture under scripts/tier2_fixtures/<backend>/<physics>/source.py and confirm rc=0.
  4. Add a "Signal:" line documenting the failure mode you just fixed, so the critic gate can match it next time.
  5. Update the per-backend Signal-coverage floor in tests/test_pitfall_signal_coverage.py only if it climbs naturally.

Honesty constraint: do NOT mark a template "fixed" until you actually re-ran it and it returned rc=0. If a template is fundamentally broken (e.g. requires a backend feature that doesn't exist upstream), mark it stub-only via the existing "_stub": true mechanism and document why — don't silently leave it broken.

Run pytest tests/ --tb=short -q after every cluster of fixes to make sure nothing regressed. Commit per-physics ("ngsolve: close hyperelasticity_3d template", etc.). Push when you're done.

Wrap up by writing a short report to RUN_A_REPORT.md: how many templates were broken, how many are now passing, how many you marked stub-only, with one-line diagnosis each.
```

---

## Run (B) — Rerun HOE-v1 (does recent catalog work move the needle?)

**What this does:** Re-executes the held-out evaluation across the 204 cells (ablation grid: BARE × MCP_NO_PITFALL_DB × MCP_NO_CRITIC × MCP_FULL). Spawns a sub-agent per cell. Compares against the published baseline (BARE 78% / MCP_FULL 94%).

**Why this MUST run in interactive mode:** Each cell needs a fresh sub-Claude to generate the FEM script, the sub-Claude must use the MCP itself, and the parent loop must stay alive to collect results.

**Estimated wall-time:** 2–6 hours depending on parallelism cap and sub-agent runtimes.

**Paste this:**

```
We are working on the OASiS MCP at ~/Schreibtisch/Open-FEM-agent (this directory). Goal: rerun HOE-v1 (Held-Out Evaluation v1) to measure whether the catalog work over the past two days (290-entry physics alias map, 26 cross-backend collation topics + 45 pitfalls, 100% Signal: coverage across 8 backends) moved the MCP_FULL number above the published 94%.

Step 1. Find the HOE-v1 harness. Look in benchmarks/hoe_v1/. There should be a runner script + a 204-cell task list + a baseline results file (BARE 78% / MCP_FULL 94%). If you can't find it, read SESSION_STATUS.md in this directory which documents where it lives.

Step 2. Confirm you understand the 4 conditions:
  - BARE: LLM with no MCP access (control)
  - MCP_NO_PITFALL_DB: MCP enabled but pitfall knowledge stripped
  - MCP_NO_CRITIC: MCP enabled but critic-gate disabled
  - MCP_FULL: everything on

Step 3. Spawn one sub-agent per cell using the Agent tool (subagent_type=general-purpose, model=opus or whatever HOE-v1 used originally — check the runner). Each sub-agent gets the cell's task description and the condition flags. Each must return a JSON verdict { passed: bool, l2_error: float, exit_code: int, evidence: str }.

Step 4. Concurrency cap: run at most 8 sub-agents in parallel (so we don't blow through the Max subscription quota — each sub-agent counts as one separate Claude invocation). Use parallel() / a worker-pool pattern.

Step 5. After all 204 cells complete, compute pass-rate per condition. Diff vs the published baseline. Write the result to HOE_V1_RESULTS_$(date +%Y%m%d).md with:
  - Per-condition pass rate (before vs after)
  - Cells that changed verdict (passed-now / failed-now)
  - Which cells used a cross-backend collation pitfall (knowledge(topic='cross_backend')) and how that correlated with passing
  - Honest assessment: did the recent catalog work demonstrably help?

Step 6. If MCP_FULL crossed 95%, commit + push the results file. If it didn't move or regressed, commit anyway — the negative result is valuable data.

Honesty constraint: do NOT fudge by re-running a failing cell until it passes. Each cell gets one shot per condition. If you hit rate-limits on the Max sub, pause and tell me, don't half-finish silently.

Wrap up: short 5-bullet summary at the top of the results file.
```

---

## Run (C) — Per-pitfall ablation (task #224)

**What this does:** For each of the 45 cross-backend pitfalls (and a sample of the per-backend ones), runs a tiny version of HOE-v1 with that single pitfall masked off. Tells us **which catalog entries actually change LLM-generated code quality** — so we know where future investment pays off.

**Estimated wall-time:** 4–12 hours (one mini-HOE per pitfall). Heavy. Probably run overnight.

**Paste this:**

```
We are working on the OASiS MCP at ~/Schreibtisch/Open-FEM-agent (this directory). Goal: build the per-pitfall ablation suite (task #224). Until now we know MCP_FULL beats BARE by 8 percentage points on HOE-v1, but we don't know WHICH pitfalls deserve credit. This run identifies the high-impact ones.

Step 1. Read benchmarks/hoe_v1/ to understand how the existing harness picks cells and conditions. We're going to add a 5th condition: MCP_FULL_MINUS_<pitfall_id>.

Step 2. Build the pitfall-masking infrastructure. The MCP server has a knowledge() tool with topic='cross_backend' + per-backend topics. Add a hidden env-var hook (or a runtime patch in the harness) that lets you suppress a specific pitfall by its ID/slug for the duration of one sub-agent's session. Verify the mask works: call knowledge() with the mask on, confirm the targeted pitfall is absent.

Step 3. Select the candidate pitfalls. Start with the 45 cross-backend pitfalls (src/backends/_cross.py). For each pitfall: pick the 5 HOE-v1 cells most likely to need that pitfall (you can guess from the description, e.g. the "units" pitfall is most likely to matter on a cell that mixes mm and m). Run those 5 cells under MCP_FULL_MINUS_<this_pitfall>. Compare pass-rate vs the unmasked MCP_FULL baseline.

Step 4. Concurrency cap: 8 parallel sub-agents max. Same as Run (B).

Step 5. For each pitfall, record:
  - delta = MCP_FULL_pass_rate(this_pitfall) - MCP_FULL_minus_pitfall_pass_rate
  - cells where masking the pitfall caused a regression
  - any "anti-helpful" pitfalls (delta < 0, meaning the pitfall HURTS performance)

Step 6. Write PER_PITFALL_ABLATION_$(date +%Y%m%d).md sorted by delta descending. Top 10 = "deserves to stay + maybe extend". Bottom 10 = "candidate for rewrite or removal."

Step 7. Commit + push the results file.

Honesty constraint: 5 cells per pitfall × 45 pitfalls = 225 sub-agent runs. If you hit rate limits, pause and tell me. Do NOT skip pitfalls silently; either run them all or report which ones you skipped and why.

This is the run that turns "the catalog has 45 cross-backend pitfalls" into "the catalog has 13 high-impact pitfalls and 32 marginal-or-redundant ones, here's the priority order for future work."
```

---

## After each run

Whichever Claude Code session finishes:
- Commits + pushes its results file to the `layer-a/kratos-source-scanner` branch (or a sub-branch — its choice).
- Tells you (in chat) the bottom-line number.

You can then come back to **this** session (the one you're reading this from) and we'll synthesize the three results into the next strategic move (PR #224 follow-ups, paper figure updates, etc.).

## Common gotchas

- **MCP server key after rename**: your `~/.claude/settings.json` probably still has `"open-fem-agent": {...}` as the MCP server entry. Rename it to `"oasis"` once, OR keep the old key (the server code doesn't care about the key in your local settings — only the `command` + `cwd`).
- **Conda env activation in fresh terminals**: Run (A) needs the right conda env for FEniCSx (`source ~/miniconda3/etc/profile.d/conda.sh && conda activate ofa-fenicsx`). The other runs are venv-based (`.venv/bin/python`).
- **Sub-agent quota**: each spawned `Agent({...})` consumes Max-subscription quota. If you start hitting the per-minute rate limit, drop the concurrency cap in the prompt from 8 → 4.
