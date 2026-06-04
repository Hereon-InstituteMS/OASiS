# Token-Savings Baseline Study

Compares OASiS MCP vs bare LLM (Claude Code without MCP) across 6 tasks of increasing difficulty.

## How to run

### Step 1: Run each task twice (with and without MCP)

**With MCP** (normal setup):
```bash
cd /path/to/oasis
claude  # MCP auto-connects
# Give the prompt from prompts.json, let it work, save session
```

**Without MCP** (remove MCP from Claude Code settings first):
```bash
cd /path/to/oasis
claude  # No MCP connected
# Give the exact same prompt, let it work, save session
```

### Step 2: Record results

After each session, fill in `results.json` with:
- Total tokens (from Claude Code session stats)
- Wall-clock time
- Number of simulation attempts
- Final accuracy (error vs reference)
- Whether first attempt succeeded

### Step 3: Generate figures

```bash
python scripts/baseline_study/analyze_results.py
```

Produces `scripts/baseline_study/figures/token_comparison.pdf`

## Task selection rationale

| Difficulty | Task | Why |
|-----------|------|-----|
| Easy | Poisson (FEniCSx) | Well-documented, zero pitfalls |
| Easy | Elasticity (scikit-fem) | Standard problem, minimal pitfalls |
| Medium | Vortex shedding (FEniCSx) | Transient NS, mesh/dt choices, validation |
| Medium | h-convergence (NGSolve) | Systematic methodology, not just "run" |
| Hard | FSI (4C) | 3+ known pitfalls causing silent wrong results |
| Hard | Cross-solver (3 solvers) | 3 different input formats, equivalent setup |
