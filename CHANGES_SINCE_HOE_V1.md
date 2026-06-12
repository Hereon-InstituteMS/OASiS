# CHANGES_SINCE_HOE_V1.md — what changed in the MCP between the published HOE-v1 run and now

> **TO THE AGENT RUNNING THE HOE-v1 RERUN ON THE OTHER MACHINE:**
> This file is a hand-off briefing, not permanent repo documentation.
> Read it, use it to set up and interpret the rerun, then **delete it
> from the repo (git rm + commit) once you have extracted what you
> need.** Its contents that deserve permanence (the expected-impact
> table, the divergence notes) belong in your rerun results document,
> not here.

**Baseline:** the published HOE-v1 ablation (BARE 40/51 = 78.4%, MCP_NO_PITFALL_DB 44/51, MCP_NO_CRITIC 45/51, **MCP_FULL 48/51 = 94.1%**) was measured against the catalog state of **v1.0.0, commit `e2f32ea`, 2026-05-26**.

**Now:** branch `layer-a/kratos-source-scanner`, HEAD ≥ `c4dae71` (2026-06-12). **662 commits** since the baseline.

**Purpose of the rerun:** same 17 tasks × 3 seeds × 4 conditions, against the current catalog. The question for the manuscript: does the post-submission catalog work move MCP_FULL above 94.1%, and does the gap to the ablated conditions widen (it should — see §2, most of what was added feeds exactly the surfaces the ablations strip)?

---

## 1. Quantitative delta (v1.0.0 → HEAD)

| Surface | v1.0.0 | now | note |
|---|---|---|---|
| Commits | — | +662 | |
| MCP tools (`@mcp.tool()`) | 13 | 16 | +`reload_catalog`, +`setup_backend`, +1 knowledge-surface split |
| `src/backends/_cross.py` (cross-backend collation catalog) | **did not exist** | 1,583 lines | 26 topics, 45+ pitfalls |
| `src/tools/consolidated.py` | 1,371 lines | 2,628 lines | alias map, dispatch surfaces, truncation removal |
| `Signal:` retrieval anchors in catalog text | **0** | ~1,327 lines | the entire critic-retrieval anchor system postdates HOE-v1 |
| Physics-synonym map entries | ~50 | ~290 (332 map lines) | |
| Test suite | 209 | 515+ / 0 failed | |
| Template probe (catalog templates that RUN) | not yet measured | skfem 25/25, ngsolve 29/29, fenics 35/35, dealii 38/39, fourc 25/46 (+21 honest stubs), kratos 7 stubs→real solves | |

**The three headline facts for interpreting the rerun:**
1. The **whole `Signal:` anchor system** (what the critic gate retrieves on failure) was built AFTER the published numbers. At HOE-v1 time the pitfall DB existed but had no failure-signature anchors.
2. The **whole cross-backend collation catalog** (`knowledge(topic='cross_backend')`) postdates HOE-v1.
3. The published MCP_FULL = 94.1% was therefore measured against a catalog roughly **half the size and structurally poorer** than the current one.

---

## 2. Changes by MCP surface, with expected HOE impact

### 2.1 `knowledge` tool — content (HIGH expected impact on MCP_FULL and MCP_NO_CRITIC)

- **`Signal:` anchors on 100% of pitfalls across all 8 backends** (~1,100 catalog pitfalls; dealii 138, dune 68, febio 61, fenics 136, fourc 335, kratos 159, ngsolve 135, skfem 148). Every pitfall now ends in an observable failure signature (exact exception text, rc, output pattern) that an agent can match against a traceback. This directly feeds the failure-recovery loop in any HOE cell where the first attempt errors.
- **Gameable-Tier-0 cleanup to zero** across all backends: every pitfall references at least one REAL code symbol verified against the installed solver (no invented APIs survive the floor tests).
- **17 live-falsification probes** (`tests/test_pitfall_falsification_live.py`): the highest-risk catalog claims are re-executed against the installed solvers; a nightly cron diffs against baseline. Caught real drift already (skfem 11.x→12.0.1 quadrature ceiling).
- **26 cross-backend collation topics / 45+ pitfalls** (`src/backends/_cross.py`, reachable via `knowledge(topic='cross_backend')`): units, element node ordering, Dirichlet enforcement, restart compat, MPI idioms, element-type naming, time-integration defaults, solver tolerances, contact formulations, output formats, integration orders, boundary tags, plasticity return mapping, turbulence, material orientation, frequency conventions, mesh quality, stress measures, periodic BCs, damping, timestamps, IC interpolation, frame-of-reference BCs, linear-solver defaults, nonlinear convergence criteria. Expected to matter most on Tier-B compositional cells and any cell where the model was trained on a *different* backend's conventions.
- **New first-class physics exposure:** fenics helmholtz/maxwell, fourc umbrella physics, dealii advection_dg/contact/nonlinear_elasticity, kratos auxiliary overview, FEBio full module catalog (refactored into per-physics generators).

### 2.2 `knowledge` / `prepare_simulation` — retrieval mechanics (HIGH)

- **290-entry physics alias map**: "heat_transfer", "cfd", "plane_strain", "amr", etc. now resolve to the right catalog rows instead of empty fall-throughs. At HOE-v1 time an agent typing a synonym got nothing.
- **Truncation removed**: `prepare_simulation` used to cut pitfalls, templates, and knowledge JSON at 3,000 chars — the agent literally could not see most of the catalog in one call. All three limits removed.
- **Fuzzy-match hardening**: empty-query silent matches removed; the canonical resolver (synonym map before substring) is used by `examples` and `discover('recommend')` too.
- `generate_input` failures are surfaced instead of swallowed; stub-only templates are explicitly marked so the agent doesn't execute a comment.

### 2.3 Templates (HIGH for any cell that runs catalog code directly)

- **fourc**: 2/46 → 25/46 running. Root cause: templates shipped literal `<placeholder>` YAML straight into 4C's MatchTree (instant MPI_Abort). 21 rows rerouted to self-contained inline-mesh inputs (embedded NODE COORDS, all params defaulted, each verified rc=0 on the real binary); 21 deep-multiphysics rows became clearly-marked reference stubs that can never be miscounted as passes.
- **dealii**: 29/39 → 38/39. Version-aware env resolution (three stacked bugs had pinned compiles to a stale 9.1.1 serial env), SLEPc-free eigenvalue template (deflated inverse power iteration — conda-forge deal.II never ships SLEPc/PETSc), stokes rebuilt as a real lid-driven cavity, dg_transport 9.3/9.4 dual-compatible, minimal-surface Newton fixed (zero-constraints for updates), hp_adaptive's crash root-caused to a stale-DoF DataOut after the final refinement.
- **skfem / ngsolve / fenics**: all template rows green (25/25, 29/29, 35/35) after the Layer-F batches.
- **kratos**: 7 specialized-app import-check stubs replaced with real solving templates.

### 2.4 New MCP tools (MEDIUM — only matters if HOE cells exercise them)

- **`setup_backend`** (detect / plan / install / verify / persist, per-OS routes, honest verification — it cannot report "verified" without a passing smoke test).
- **`reload_catalog`** (hot-reload without server restart).
- **Source-management layer** (`ensure_source`: discover → fetch → build for all 8 backends; single config at `~/.config/oasis/sources.json`, with legacy-path fallback).

### 2.5 Cross-backend numerical verification (NEW — manuscript-relevant beyond HOE)

`tests/test_layer_d_phase2.py`: the catalog's own templates, run through real `generate_input() → run()` subprocess paths, agree across solvers:
- Canonical Poisson peak across **5 backends** (analytic 0.0736713; skfem/dealii/4C to 7 digits).
- Resolution-aligned plane-strain cantilever across **4 backends**, with skfem↔4C agreeing to **10 significant digits** on an identical Q1 mesh (gate pinned at 1e-6).
- Documented genuine divergence: **fenics's default 2D elasticity template is plane STRESS; the other four backends' are plane STRAIN.** If any HOE cell compares 2D elasticity across fenics and another backend, this ~(1−ν²) systematic difference is *expected*, not a bug.

---

## 3. Things the rerun agent must know (gotchas)

1. **Condition mapping**: `MCP_NO_PITFALL_DB` should now strip BOTH the per-backend pitfalls AND the cross-backend collation catalog AND the Signal: anchors — they are all "pitfall DB" in spirit. If the harness only strips the original per-backend lists, the ablation under-measures and the NO_PITFALL condition will look better than it should. Check the masking implementation before running.
2. **`prepare_simulation` responses are much longer now** (truncation removed + bigger catalog). If the harness pipes responses through any length-limited channel, verify nothing re-truncates.
3. **The 4C/dealii cells may behave differently for the *better* reasons**: at HOE-v1 time several catalog templates were broken, so an agent's correct move was to write code from scratch; now the template is often directly runnable. Watch for cells where the agent's strategy changes (template-reuse vs from-scratch).
4. **fourc reference stubs**: 21 fourc rows return clearly-marked "Not a runnable input" stubs. An agent that blindly executes one will fail validate_input with an explanatory message — that is intended behavior, not a regression.
5. **C5 anomaly** (the one cell where BARE 1/3 beat all MCP conditions 0/3 in the published run): nothing was done specifically for C5 because the harness lives on your machine — check whether the new catalog changes its outcome, and if C5 still inverts, that's the per-pitfall-ablation lead (task #224).
6. **Versions on this machine at close**: skfem 12.0.1, Kratos 10.4.2/py3.12, dolfinx in `ofa-fenicsx`, deal.II 9.3.2 in `ofa-dealii-93` (NEW env — the old `ofa-dealii` 9.1.1 is stale; discovery now picks the highest version), 4C at `~/Schreibtisch/4C-src/4C/build/4C`. dune is NOT runnable here (no conda-forge package exists; local source build ABI-broken); febio binary not installed.
7. The repo's own quality gates to re-run before trusting a rerun: `pytest tests/ --ignore=tests/test_pitfall_falsification_live.py` (expect 515+/0) and `pytest tests/test_pitfall_falsification_live.py` (expect 65/0 with env skips).

---

## 4. Honest expectations for the rerun

- **MCP_FULL**: should improve or hold. The added Signal anchors and alias map remove two known first-attempt failure modes (synonym miss, truncated knowledge). 48/51 → 49–50/51 is plausible; the remaining failures (C5, D-tier) may be harness/task-intrinsic.
- **MCP_NO_PITFALL_DB**: should *worsen relative to MCP_FULL* (the pitfall DB is now much stronger, so stripping it costs more) — i.e., the ablation gap should WIDEN. That widened gap is the cleanest manuscript claim.
- **BARE**: unchanged by construction (no MCP access) — any movement here is harness noise; use it as the seed-stability check.
- If MCP_FULL *drops*, suspect (in order): condition-masking mismatch (§3.1), response-length handling (§3.2), backend version drift on the rerun machine vs the task specs.

---

> **Reminder: delete this file after extracting what you need** —
> `git rm CHANGES_SINCE_HOE_V1.md && git commit -m "Remove HOE rerun briefing (consumed)"`.
> The permanent record is your rerun results document plus the git
> history itself.
