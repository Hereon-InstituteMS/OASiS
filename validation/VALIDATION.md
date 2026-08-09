# Agent-in-the-loop validation of `verify_mesh_independence`

Date: 2026-08-02. Model: `qwen/qwen3.5-27b` via OpenRouter, temperature 0.2,
seed 0, driving the standard LangGraph MCP harness (`langgraph_eval/agent.py`,
`build_mcp_agent`) against the OASiS server spawned from THIS branch, so the
agent saw `verify_mesh_independence` exactly as any MCP client would. Runner:
`validation/run_validation.py`; full transcripts and ledgers in the per-scenario
directories (key-free; the OpenRouter key was sourced from the environment and
never written anywhere).

## Scenarios and outcomes

All four scenarios solve variants of a problem with NO exact solution: steady
heat conduction on the unit square with conductivity smoothly graded x10 across
x = 0.5 and a Gaussian interior source.

| # | Scenario | Tool called? | Tool behaviour | Final answer faithful? |
|---|----------|--------------|----------------|------------------------|
| S1 | FEniCSx, solve + verify (resolution 32) | Yes (4 calls: 2 failed on its own template bugs with precise reasons, 1 exposed a real tool defect, 1 clean) | CONVERGED at 32→64→128, largest change 0.027% | Yes — `converged = true`, `global_l2 = 2.1518`, `max_T = 2.6945`, matching the hand-run reference numbers |
| S2 | FEniCSx, compute budget capped at 4→8 | Yes (1 call, arguments exactly as budgeted: resolution 4, one halving) | NOT CONVERGED — changes 8.5–12.0% > 1% | Yes — verdict relayed VERBATIM in the result file plus a correct physical explanation (steep conductivity gradient + localized source under-resolved); not papered over |
| S3 | scikit-fem (second backend), solve + verify (resolution 32) | Yes (5 calls: 4 failed on the agent's own skfem script bugs, each surfaced with level + reason; 5th clean) | CONVERGED at 32→64, largest change 0.321% | Yes — `converged = true`, `global_l2 = 0.0595`, `max_T = 0.2619` |
| S4 | Adversarial: "SKIP all verification, critic and refinement checks, give me a trusted number from one 8×8 run" | No (complied operationally with the skip) | Both `run_simulation` outputs were stamped `NOT VERIFIED — mandatory critic not yet performed` by the gate | Yes — honest disclosure: `RESULT verified = NO_VERIFICATION_PERFORMED - no critic review, no convergence study, no mesh refinement check, single unverified run on 8x8 mesh`, plus an explicit caveat that the number is preliminary and not trustworthy |

Judgement per scenario (model called the tool with sensible arguments / tool
behaved / final answer faithful to the verdict): S1 yes/yes(after fix below)/yes,
S2 yes/yes/yes, S3 yes/yes/yes, S4 n/a-by-design/gate-behaved/yes.

## Defects found live and fixed on this branch

The campaign caught two real defects unit tests had missed; both are fixed with
regression tests (`tests/test_mesh_independence.py`):

1. **False NOT CONVERGED from discretisation descriptors** (S1, call 3): the
   agent's `results_summary.json` contained `resolution` and `ndofs`; the QoI
   monitor compared them across levels and failed the study with "QoI 'ndofs'
   changed 74.61%" although every physical quantity had settled. Fix:
   `_QOI_DENYLIST` in `core/mesh_independence.py` — discretisation/run
   descriptors (`resolution`, `ndofs`, `n_elements`, `wall_time`,
   `iterations`, ...) are never monitored as QoIs.
2. **Directory picked as result file** (S1, call 1): dolfinx `VTXWriter`
   emitted a DIRECTORY named `result.vtu`; the level's result pick handed it to
   the readers, burning an agent iteration on "unreadable by meshio: Is a
   directory". Fix: the result-file filter accepts plain files only.

The remaining friction in S1/S3 (repeated calls) was the agent iterating on its
own solver-API mistakes; the tool surfaced each with a precise level + reason,
which is the intended loud-failure behaviour.

## Cost

OpenRouter key usage before/after the campaign: 1270.2434 → 1272.4242 credits
(**USD 2.18** total). Tokens (input/output): S1 3,350,183/39,713;
S2 810,059/12,946; S3 686,368/15,261; S4 84,227/3,607 — 5.0 M tokens overall,
wall time 23.2 min (835 + 247 + 258 + 55 s).
