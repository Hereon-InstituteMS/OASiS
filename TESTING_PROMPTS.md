# TESTING_PROMPTS.md — fresh-terminal prompts for OASiS capability testing

**Updated:** 2026-06-12 (supersedes the run-list in `LAUNCH_PROMPTS.md` for template/setup work; the HOE-v1 prompts there remain valid once the harness is recovered from the other machine).

Each prompt below runs in **its own fresh terminal with Claude Code interactive** (NOT `claude --print` — sub-agents only exist in interactive sessions). Each one names its working directory. They are independent — run any subset, in parallel if you like.

```bash
# Pattern for every run:
mkdir -p ~/oasis-runs/<run-name> && cd ~/oasis-runs/<run-name>
claude
# paste the prompt block, walk away
```

The dedicated directory matters: the runs write scratch meshes/VTUs/logs, and you don't want that in the repo working tree.

---

## Run 1 — `setup_backend` end-to-end on Ubuntu (NEW tool, this machine)

**Directory:** `~/oasis-runs/setup-ubuntu`
**What it tests:** the brand-new `setup_backend` MCP tool (commit `01039a5`) doing the full journey on a backend that is genuinely missing on this machine: **dune** (conda env was never finished — task #15) and **febio** (binary never downloaded).
**Duration:** ~30 min.

```
We are testing the OASiS MCP's new setup_backend tool. The MCP server code is at ~/Schreibtisch/Open-FEM-agent (run its tests from there), but do all scratch work in the current directory.

1. Call the MCP tool setup_backend(action='status') and report the table. Expected: dune and febio NOT available, everything else available.

2. For dune: setup_backend(action='plan', solver='dune') — verify the plan recommends the conda route with the ofa-dune env. Then setup_backend(action='install', solver='dune'). This runs conda create (~10 min). When it finishes, setup_backend(action='verify', solver='dune') and confirm the smoke test passes and the paths got persisted to ~/.config/oasis/sources.json.

3. For febio: setup_backend(action='plan', solver='febio') — the binary route returns manual instructions. Follow them: download the linux FEBio package from https://febio.org/downloads/ into ./febio/, unpack, then set FEBIO_BINARY and call setup_backend(action='verify', solver='febio').

4. File honest bug reports: for every rough edge (wrong path persisted, misleading plan output, smoke test that passes but the backend doesn't actually solve), write a numbered finding into ./FINDINGS.md with the exact tool call and output. Then fix what's fixable in ~/Schreibtisch/Open-FEM-agent/src/core/backend_setup.py, add a regression test in tests/test_backend_setup.py, run pytest tests/test_backend_setup.py, and commit on the layer-a/kratos-source-scanner branch.

Honesty constraint: do NOT mark a backend 'verified' unless its smoke test genuinely solved something. If conda/download infrastructure is unavailable, say so and stop — don't fake the result.
```

---

## Run 2 — fourc placeholder-template fan-out (the remaining ~27 broken rows)

**Directory:** `~/oasis-runs/fourc-templates`
**What it tests/fixes:** the remaining fourc catalog rows that abort in 4C's MatchTree with un-substituted `<...>` placeholders. 5 rows were already fixed by inline-mesh routing (commit `6a1755e`) — the same pattern extends to several more; the truly exotic multiphysics rows should become honest reference stubs.
**Duration:** 2-5 h with sub-agent fan-out.

```
We are fixing the OASiS MCP's broken 4C templates. Repo: ~/Schreibtisch/Open-FEM-agent (work on branch layer-a/kratos-source-scanner). 4C binary: ~/Schreibtisch/4C-src/4C/build/4C (export FOURC_ROOT=~/Schreibtisch/4C-src/4C and FOURC_BINARY accordingly). Scratch work in the current directory.

Context: benchmarks/probe_all_templates.py --backend fourc --timeout 300 currently shows 9/46 rows completed. The failing rows abort in 4C's MatchTree because generate_input() falls through to generator templates with literal <placeholder> scalars. Commit 6a1755e closed 5 rows by routing them to self-contained inline-mesh inputs (src/backends/fourc/inline_mesh.py — study matched_elasticity_input / matched_heat_transient_input / matched_elasticity_3d_nonlinear_input as the pattern). tests/test_fourc_inline_routing.py shows the gen-only regression-test pattern.

Plan:
1. Run the probe to get the current failing list (read benchmarks/probe_results/templates.json afterwards — it merges per-backend now).
2. Triage every failing row into:
   A. ROUTABLE — physics that a small inline QUAD4/HEX8 mesh can carry (candidates: ale/ale_2d, porous_media/single_phase_3d, level_set/advection_2d, low_mach/heated_channel_2d, electrochemistry/nernst_planck_3d, tsi/monolithic_3d — note inline_mesh.py already has matched_tsi_oneway_input!).
   B. STUB — deep multiphysics that genuinely needs case-specific meshes (xfem, fs3i, ehl, fbi, fpsi, pasi, ssti, sti, cardiac_monodomain, arterial_network, reduced_airways, multiscale/fe2, beam_interaction). Convert these to honest reference stubs via the existing _reference_stub_template mechanism so users get documentation instead of a guaranteed MPI_Abort.
   C. NON-YAML gen-failures (membrane, shell, thermo, mixture, constraint, brownian_dynamics, cardiovascular0d, reduced_lung, fluid_turbulence) — these templates aren't even YAML dictionaries; inspect and either fix the template format or stub them.
3. For category A, spawn one sub-agent per physics with: the failing row, the inline_mesh.py pattern, and the 4C binary path. Each sub-agent iterates generate → run → read stdout.log → fix until rc=0, then writes a gen-only regression test following tests/test_fourc_inline_routing.py.
4. Cap sub-agent parallelism at 4 (4C runs are CPU-hungry).
5. After each cluster: pytest tests/ --tb=short -q (full sweep), commit per-cluster with honest messages stating what was ROUTED vs STUBBED.
6. Finish with a fresh probe run and report the final completed/46 score in ./REPORT.md.

Honesty constraints: a stub is a SUCCESS state only if it is clearly marked as a stub (users must not run a comment); never claim a row fixed without a fresh rc=0 run; if a physics needs upstream 4C features that the build lacks, document that in the stub.
```

---

## Run 3 — Kratos stub replacement (8 availability-probe stubs → real solves)

**Directory:** `~/oasis-runs/kratos-stubs`
**What it fixes:** task #25/#42 — 8 kratos rows whose "template" just import-checks the application and writes a note. Users deserve real physics.
**Duration:** 2-4 h.

```
We are replacing the OASiS MCP's Kratos availability-probe stubs with real solving templates. Repo: ~/Schreibtisch/Open-FEM-agent, branch layer-a/kratos-source-scanner, venv .venv (KratosMultiphysics 10.x installed). Scratch in current directory.

1. Identify the stub rows: grep '_generic_kratos_template' src/backends/kratos/generators/specialized.py — these emit import-check stubs. Cross-check against benchmarks/probe_results/templates.json (kratos rows that gen but fail validate are the stubs).
2. For each stub where the application is actually pip-installed in .venv (probe with python -c "import KratosMultiphysics.<App>"), replace the stub with a minimal REAL solve: smallest meaningful mdpa mesh (or KM.Model built programmatically — see the pattern in commit bb7533c "real KM.Model solves replacing scipy stubs"), real ProjectParameters, real solver loop, results_summary.json with a physical quantity (not just "available").
3. Where the application is NOT pip-installed (e.g. apps that need a source build: Chimera, FemToDem...), keep the stub but make its note say exactly that, with the install hint.
4. One sub-agent per application, cap 4 parallel. Each iterates template → run in scratch dir → fix → rc=0 with a .vtk/.vtu output.
5. Add/extend tests: each replaced template gets a row in the catalog-consistency / signal tests as appropriate; run pytest tests/ before each commit.
6. Report final stub count in ./REPORT.md (target: only genuinely-uninstallable apps remain stubs).

Honesty constraint: a 'real solve' must produce a non-trivial physical result (displacement, temperature, level — something checkable), not a 1-element no-op that technically exits 0.
```

---

## Run 4 — macOS extension session (on your Mac, other Claude instance)

**Directory (on the Mac):** `~/oasis-runs/setup-mac` with a fresh clone of the repo
**What it does:** fills the `darwin` extension points in `src/core/backend_setup.py` with VERIFIED Mac routes — starting with your 4C-on-Mac compile settings from the 4C discussion thread.
**Duration:** depends on how many backends you want verified; 4C alone is ~2 h of compile time.

```
We are extending the OASiS MCP's setup_backend tool with verified macOS routes. Clone https://github.com/Hereon-InstituteMS/OASiS.git, branch layer-a/kratos-source-scanner, work in this directory.

Context: src/core/backend_setup.py has a SETUP_ROUTES catalog where every route carries os_support metadata per OS. The 'darwin' entries are deliberate EXTENSION POINTS — structured but flagged "verified": False. The schema per OS entry: {"verified": bool, "system_deps": [brew package names], "notes": [step-by-step guidance]}. tests/test_backend_setup.py pins the catalog invariants (don't break: every route needs a linux entry; pip/conda routes carry argv command lists; binary/source routes must NOT).

I (the user) documented working 4C-on-Mac compile settings in a 4C GitHub discussion thread — I will paste them when you ask. Your job:

1. Run setup_backend(action='status') on this Mac and record the baseline.
2. For each backend, attempt the fastest route on macOS: skfem (pip), ngsolve (pip wheel — check arm64), kratos (pip — check which application wheels resolve on arm64), fenics + dune + dealii (conda-forge osx-arm64), febio (official mac installer), fourc (source — use my pasted thread settings).
3. After each successful route: setup_backend(action='verify', solver=...) must pass its smoke test. Then update that route's os_support['darwin'] entry in src/core/backend_setup.py: verified: True, the exact brew deps, and notes containing the verified step list (for 4C: the CMake cache entries / compiler pins / any source patches from my thread, as actually re-verified on this machine — not just copied).
4. Where a route FAILS on macOS, set verified: False and write the failure mode into notes (that's valuable too).
5. Keep tests green: pytest tests/test_backend_setup.py after each catalog edit. The darwin-extension-point test must keep passing.
6. Commit per-backend ("setup_backend: verified darwin route for ngsolve (arm64 wheels)" etc.) and push the branch.

Honesty constraint: 'verified' means the smoke test solved something on THIS Mac today — not 'the wheel installed' and not 'the thread says it works'.
```

---

## Run 5 — Coupling pair hardening (task #49, 12 untested DD pairs)

**Directory:** `~/oasis-runs/coupling-pairs`
**What it tests:** `coupled_solve` advertises 20 domain-decomposition solver pairs; only 8 are validated end-to-end. The multi-code claim is the paper's core — every untested pair is exposure.
**Duration:** 2-4 h.

```
We are validating the OASiS MCP's cross-solver coupling claims. Repo ~/Schreibtisch/Open-FEM-agent, branch layer-a/kratos-source-scanner, scratch in current directory.

1. Enumerate the supported pairs: read src/tools/coupling.py and the coupled_solve tool registration in src/tools/consolidated.py. List all advertised (solver_a, solver_b) combos and identify which have an existing passing record (tests/, benchmarks/coupling/, E2E_POSTMORTEMS.md).
2. For each UNTESTED pair that is runnable on this machine (both solvers available — check setup_backend(action='status')): run the canonical Poisson domain-decomposition through the actual MCP tool coupled_solve(problem='poisson_dd', solver_a=..., solver_b=...). Iterate on failures: the fix may live in the per-pair script generator in coupling.py.
3. PASS criterion: the DD iteration converges AND the interface solution mismatch between the two solvers is < 1e-6 relative (read both sides' results).
4. One sub-agent per pair, cap 4 parallel.
5. Every fixed pair: regression-pin it (extend the existing coupling tests), commit per-pair.
6. ./REPORT.md: final validated-pairs matrix (the paper's Table needs this number to be honest).

Honesty constraint: if a pair cannot work (e.g. an output format the field-transfer layer cannot read), do NOT silently drop it from the advertised list — flag it in the report so we decide whether to fix transfer_field or stop advertising the pair.
```

---

## After the runs

Each run commits + pushes on `layer-a/kratos-source-scanner` and leaves a `REPORT.md`/`FINDINGS.md` in its scratch directory. Come back to the main session and we synthesize: probe scores before/after, which catalog claims changed, and what flows into the next release (v1.2.0 with the full catalog).

## Baseline history

### 2026-06-12 morning (when these runs were written)

| metric | value |
|---|---|
| Test sweep | 414 passed / 0 failed (+ falsification probes 65/0) |
| Template probe: skfem | 25/25 run |
| Template probe: ngsolve | 29/29 run |
| Template probe: fenics | 35/35 run |
| Template probe: kratos | 32/40 run (8 stubs → Run 3) |
| Template probe: dealii | 29/39 run |
| Template probe: fourc | 9/46 run (→ Run 2) |
| setup_backend status | 6/8 backends available (dune, febio missing → Run 1) |
| Coupling pairs validated | 8/20 (→ Run 5) |

### 2026-06-12 evening (after Runs 1-3 + main-session batches)

| metric | value | delta |
|---|---|---|
| Test sweep | 515+ passed / 0 failed | +101 tests |
| Template probe: skfem | 25/25 run | — |
| Template probe: ngsolve | 29/29 run | — |
| Template probe: fenics | 35/35 run | — |
| Template probe: kratos | 7 real solves replaced import-stubs (Run 3) | ✓ |
| Template probe: dealii | **38/39 run** (39th = parallel_poisson, impossible on conda builds by design, clean #error) | +9 |
| Template probe: fourc | **25/46 run + 21 honest stubs** (Run 2 + main session) | +16 routed |
| setup_backend | shipped + Run-1 honesty fixes (no fabricated verifies) | NEW |
| Layer D phase 2 | 5-backend Poisson + 4-backend cantilever consistency gates; skfem↔4C twin pair pinned at 1e-6 (observed 1e-10) | NEW |
| Coupling pairs validated | 8/20 (Run 5 still pending) | — |
| Runs still open | Run 4 (macOS), Run 5 (coupling) | |
