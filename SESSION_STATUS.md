# SESSION_STATUS.md — OASiS context dump for parallel Claude Code sessions

**Generated:** 2026-06-04
**Project:** OASiS — an open-source multi-physics and multi-code framework for verified computer simulations
**Repo:** https://github.com/Hereon-InstituteMS/OASiS (renamed today from `Open-FEM-agent`)
**Local working dir:** `~/Schreibtisch/Open-FEM-agent/` (NOT renamed — see "Why the local dir is still Open-FEM-agent" below)
**Active branch:** `layer-a/kratos-source-scanner` (554 commits ahead of `main`; all pushed)

---

## 1. What OASiS is, in one paragraph

A Model Context Protocol (MCP) server that connects AI coding agents to **eight finite element solvers** (FEniCSx, deal.II, 4C Multiphysics, NGSolve, scikit-fem, Kratos Multiphysics, DUNE-fem, FEBio). Any MCP-compatible AI tool (Claude Code, Cursor, Windsurf, GitHub Copilot) can ask OASiS to *operate*, *couple*, and *develop* simulations through these solvers via a small consolidated tool surface (13 tools). The actual differentiator vs. searching individual solver docs is the **cross-backend collation knowledge** — 26 topics, 45 pitfalls that describe gotchas at the *delta* between two or more solvers.

---

## 2. Who's who in the goal

- **Real user:** simulation engineer who knows FEM but doesn't know every solver's defaults. Asks an AI "set this up in FEniCS, run it, compare against my Kratos baseline." Gets correct, runnable code instead of plausible-looking nonsense.
- **Failure mode we're fighting:** the LLM confidently writes code using a retired API, the wrong element name, or a backend-specific convention that silently produces wrong answers. The catalog (knowledge + pitfalls + cross-backend collation + Signal: retrieval anchors + critic gate) is designed to surface those gotchas before the simulation runs.
- **Empirical proof point:** HOE-v1 ablation (already published in the paper) shows MCP_FULL beats BARE by **94% vs 78%** on 204 held-out tasks. That 16-pp gap is the load-bearing number.

---

## 3. Architecture at a glance

```
~/Schreibtisch/Open-FEM-agent/
├── src/
│   ├── server.py                       # FastMCP entry point
│   ├── tools/
│   │   ├── consolidated.py             # 13 MCP tools registered here
│   │   │                               # knowledge() / prepare_simulation() / examples() / discover() / etc.
│   │   ├── coupling.py
│   │   ├── mesh_generation.py
│   │   ├── developer.py
│   │   └── workflows.py
│   ├── core/
│   │   ├── registry.py                 # BACKEND_REGISTRY — per-backend metadata
│   │   ├── autodiscovery.py
│   │   ├── source_config.py            # ~/.config/oasis-agent/sources.json
│   │   ├── source_fetch.py             # auto-clone canonical upstream
│   │   ├── post_processing.py
│   │   ├── precice_config.py
│   │   ├── quality_checks.py
│   │   ├── field_transfer.py
│   │   └── (etc.)
│   ├── backends/
│   │   ├── _cross.py                   # 26 cross-backend topics, 45 pitfalls
│   │   ├── fenics/      generators/ + backend.py
│   │   ├── dealii/      generators/ + backend.py + element_catalog.py
│   │   ├── fourc/       generators/ + backend.py + data/fourc_knowledge.py
│   │   ├── ngsolve/     generators/ + backend.py
│   │   ├── skfem/       generators/ + backend.py
│   │   ├── kratos/      generators/ + backend.py
│   │   ├── dune/        generators/ + backend.py
│   │   └── febio/       generators/ + backend.py
│   └── data/
│       └── fourc_knowledge.py
├── tests/
│   ├── test_cross_backend_pitfalls.py        # invariants on _cross.py
│   ├── test_knowledge_cross_backend_dispatch.py
│   ├── test_physics_alias_map.py             # 290-entry alias map invariants
│   ├── test_pitfall_falsification_live.py    # 17 live probes against installed solvers
│   ├── test_pitfall_signal_coverage.py       # per-backend Signal: floor
│   ├── test_signal_verification.py
│   ├── test_mcp_tools_adversarial.py
│   ├── test_postmortem_retrieval.py
│   └── (etc, 408 tests pass)
├── benchmarks/
│   ├── hoe_v1/                # HELD-OUT EVAL v1 (the paper's ablation: BARE/NO_PITFALL/NO_CRITIC/FULL × 51 cells)
│   ├── run_all_benchmarks.py
│   └── generate_paper_figures.py
├── scripts/
│   ├── cron/
│   │   ├── probe_regression.py        # nightly: re-runs 17 falsification probes vs baseline
│   │   └── hoe_rerun.py               # weekly (currently inert; needs task specs)
│   ├── scan_results/                  # auto-generated; safe to delete + regen
│   └── tier2_fixtures/                # per-bug repro scripts → JSON traceback fixtures
├── logo/
│   ├── logo.png
│   └── logo_w_text.png                # used as the README header
├── README.md
├── CITATION.cff
├── pyproject.toml                     # package name now "oasis-mcp"
├── mcp_config.json                    # template; MCP key "oasis"
├── LAUNCH_PROMPTS.md                  # ← kickoff prompts for fresh terminals
└── SESSION_STATUS.md                  # ← this file
```

---

## 4. What was accomplished in the past ~24 hours

Single most important thing the other session needs to know: **the test sweep is green (408/0)** and the catalog is **structurally complete** (100% Signal: anchor coverage, 100% Tier-0 / Tier-1 floors across all 8 backends). Recent work focused on the cross-backend collation layer (the actual differentiator).

### Concrete shipped items (commit summary, newest first)

| Commit | What | Why it matters |
|---|---|---|
| `96ce900` | Rebrand: Open FEM Agent → OASiS | Repo + 45 files + logo; ready for paper submission |
| `c58feea` | Add 2 cross-backend topics: linear_solver_defaults + nonlinear_convergence_criteria | Two extremely common cross-backend pain points: GMRES restart (PETSc=30 vs NGSolve=no-restart), tolerance scaling (PETSc 1e-5 vs NGSolve 1e-12), Newton convergence-failure signal-path (exception/tuple/exit-code) |
| `e3a7589` | Close 3 test-sweep failures: skfem quadrature ceiling drift + febio _general + kratos cable_net Signal: gaps | Sweep went from 400/3 to 408/0 |
| `3e97311` | Lock the `knowledge(topic='cross_backend')` dispatch path | 6 regression tests pinning the wiring in `src/tools/consolidated.py` |
| `bea9754` | skfem: enumerate Element class names instead of `*` wildcards | Catalog-consistency test now passes |
| `9e945ef`–`75025bf` | Cross-backend pitfalls batches 1–10 | The MCP's actual differentiator — 24 → 26 topics; topics include units, element-node-ordering, dirichlet-BC, restart-checkpoint, MPI launch, element-type-naming, time-integration, solver-tolerance, contact-formulation, output-format, integration-order, boundary-tag, plasticity-return-map, turbulence, material-orientation, frequency, mesh-quality, stress-measure, periodic-BC, damping, timestamp, IC-interpolation, frame-of-ref-BC, **linear_solver_defaults, nonlinear_convergence_criteria** |
| `cecc0b2` | Add probe-regression + HOE-rerun cron scripts | Nightly heartbeat: catches catalog/upstream drift before users do |
| `83d3d95` | Comprehensive **290-entry physics alias map** | User types "heat_transfer", "cfd", "plane_strain", "amr" → reaches the right backend content |
| `15d012e` | Remove `eval/layer_g/` duplicate of HOE-v1 | Caught a duplicate-build mistake mid-session; HOE-v1 already existed |
| `3cadbc5` | Layer G v0 (later deleted ↑) | Aborted parallel build of an eval system that already existed |

### Per-backend Signal: coverage (locked by `tests/test_pitfall_signal_coverage.py`)

| backend | with_signal / total | % |
|---|---|---|
| dealii | 138 / 138 | 100.0% |
| dune | 68 / 68 | 100.0% |
| febio | 61 / 61 | 100.0% |
| fenics | 136 / 136 | 100.0% |
| fourc | 335 / 335 | 100.0% |
| kratos | 159 / 159 | 100.0% (closed today; was 95.6% — 7 cable_net gaps) |
| ngsolve | 135 / 135 | 100.0% |
| skfem | 148 / 148 | 100.0% |

**All 8 backends at 100% Signal: anchor coverage.** Floor in the test is 99% to allow a small reordering noise.

### 17 live falsification probes
Run via `pytest tests/test_pitfall_falsification_live.py`. As of today: **65 passed / 0 failed / 8 skipped** (8 skipped are conditional on the relevant backend's conda env being available). The nightly probe-regression cron diffs vs. baseline and pages if anything drifts (e.g. caught the skfem 11→12.0.1 quadrature ceiling upgrade today).

---

## 5. What still hurts real users (the open pile)

| Task # | Backend | Issue | Why it hurts |
|---|---|---|---|
| #27 | NGSolve | Broken templates (some templates emit code that doesn't run) | User's first `prepare_simulation(solver='ngsolve', ...)` fails |
| #28 | FEniCSx | Same: broken templates | Same |
| #29 | 4C | Broken templates + WALL element naming gap | 4C structural 2D users hit the wrong element name |
| #30 | deal.II | Rebuild needed to unlock 36/36 templates | Heavy: requires fresh deal.II compile. **NOT** to be touched without explicit OK |
| #25 | Kratos | 6 remaining stub-only templates | Honest stubs; should expand if upstream gains the capability |
| #31 | DUNE | F-012 pybind11 JIT FieldVector incompat | Single template breaks; medium |
| #224 | (eval) | Per-pitfall ablation — which catalog entries actually move LLM code | Strategic — tells us where investment pays off |

`LAUNCH_PROMPTS.md` has the kickoff prompts for Runs (A) #27/#28/#29, (B) HOE-v1 rerun, (C) #224 per-pitfall ablation.

---

## 6. Why the local dir is still `Open-FEM-agent`

I deliberately did NOT rename `~/Schreibtisch/Open-FEM-agent/`. Renaming it would break:
- The user's `~/.claude/settings.json` MCP entry (`cwd: ".../Open-FEM-agent/src"`)
- All `scripts/cron/*.py` schedules (they hardcode the path)
- All `scripts/scan_results/*.json` files (they record absolute paths)
- Memory-system path references in this Claude Code's auto-memory
- The user's git-tracked working tree until they `git clone` fresh under a new name

Only the **GitHub repo name** + **package metadata** + **README branding** were renamed. The local dir keeps its old name. Git auto-redirects the old GitHub URL to the new one.

---

## 7. How to interact with the MCP from this session

```bash
# in the working dir:
cd ~/Schreibtisch/Open-FEM-agent

# run all tests:
.venv/bin/python -m pytest tests/ --tb=short -q --ignore=tests/test_pitfall_falsification_live.py

# run live probes (slower; needs all 4 conda envs available):
.venv/bin/python -m pytest tests/test_pitfall_falsification_live.py --tb=short -q

# inspect cross-backend pitfalls programmatically:
.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from backends._cross import CROSS_BACKEND_PITFALLS; print(len(CROSS_BACKEND_PITFALLS), 'topics'); [print('-', k) for k in sorted(CROSS_BACKEND_PITFALLS)]"

# query MCP knowledge tool directly (bypasses the MCP wire):
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from mcp.server.fastmcp import FastMCP
from tools.consolidated import register_consolidated_tools
from core.registry import load_all_backends
load_all_backends()
mcp = FastMCP('test'); register_consolidated_tools(mcp)
print(mcp._tool_manager._tools['knowledge'].fn(topic='cross_backend', physics='units')[:800])
"
```

---

## 8. Hard-won lessons embedded in CLAUDE.md (read these)

- `feedback_open_fem_mcp_critic_gate.md`: invoke a senior-AI-scientist critic sub-agent before major design decisions
- `feedback_open_fem_mcp_falsification_pattern.md`: pitfalls must be live-reproducible, not just Tier-0/1 verified — falsification gate lives in `tests/test_pitfall_falsification_live.py`
- `feedback_open_fem_walker_completion_preference.md`: SUPERSEDED — see Layer G pivot below
- `project_open_fem_layer_g_pivot.md` (2026-06-03): walker paused after 2nd critic; build eval that measures whether catalog entries change LLM-generated FEM code BEFORE more catalog work
- `project_open_fem_hoe_v1_exists.md` (2026-06-03): the critic-claimed gap was wrong — `benchmarks/hoe_v1/` has full ablation already, MCP_FULL beats BARE by +8pp

If you (the other Claude) find yourself about to spend tokens on a "file-walker rotation," **stop and re-read those lessons.** That was the dominant completion-theatre failure mode.

---

## 9. Tests / sweep status as of this commit

```
408 passed, 9 skipped, 2 warnings, 199 subtests passed in ~500s
```

(when `tests/test_pitfall_falsification_live.py` is included, add another 65 passed / 8 skipped)

---

## 10. Standing instructions that don't change

- **Honesty constraints (verbatim from the user):**
  > "Do not invent symbols, do not skip C++ files because they look long, do not claim coverage gains the audit doesn't actually report. If a file is genuinely unreadable in this env (binary, locked), mark it status: 'skip' with the reason. If a backend's source tree isn't locally cloned (e.g. dealii's full .cc tutorials), STOP and ask the user where to clone it from rather than silently narrowing scope."

- **No `--no-verify` on commits.** Hooks must pass.
- **No force-push to `main`.** Branches only.
- **No drift into completion theatre.** If your tick is "X commits, Y files walked" without a connection to "user W has a better experience," you're drifting. Stop, re-read this doc, pick a concrete user-facing pain point instead.

---

## 11. Next deliverables (in priority order)

1. **Run (B) — HOE-v1 rerun.** Does the recent catalog work move MCP_FULL past 94%? One number. Highest information-per-token.
2. **Run (A) — broken templates #27/#28/#29.** Each user's first interaction with the MCP works.
3. **Run (C) — per-pitfall ablation.** Long-term: tells us which catalog entries earn their keep.

`LAUNCH_PROMPTS.md` has the verbatim prompts. Run each in its own fresh terminal so sub-agents are available.

---

*This file is a snapshot. The latest commit hash is in `git log -1`. The current branch is `layer-a/kratos-source-scanner`.*
