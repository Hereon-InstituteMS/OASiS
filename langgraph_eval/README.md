# Open-weight model-size ablation (LangGraph + OASiS-MCP)

Companion experiment to the HOE-v2 campaign. Runs Qwen2.5 at **7 B / 14 B /
32 B** parameters under two conditions:

| Condition | What the agent gets |
|-----------|--------------------|
| `BARE`   | host-side toolset only (run_bash, read_file, write_file, web_search, spawn_subagent). **No OASiS MCP.** |
| `MCP`    | the same host-side tools **plus** every OASiS MCP tool attached via `langchain-mcp-adapters` (MCP_FULL semantics — no ablation flags). |

The host-side toolset mirrors what Claude Code provides natively (Bash,
Read, Write, WebSearch, Agent). Critically, `spawn_subagent` lets Qwen
fulfil the MANDATORY CRITIC protocol the OASiS server prompts it to
follow — a sibling LangGraph agent on the same vLLM endpoint, depth-
capped at 2 to prevent runaway recursion. Without this, the MCP
condition would be unfair (the server tells the agent to spawn a
critic; with no Agent-equivalent the model has to ignore that).

OASiS tools picked up automatically: `prepare_simulation`, `knowledge`,
`discover`, `examples`, `developer`, `generate_mesh`, `run_simulation`,
`run_with_generator`, `coupled_solve`, `transfer_field`, `visualize`,
`session_insights`, `rediscover_backends`. Any tool added to the server
later is included without changes here.

The hypothesis under test: **a small model with MCP matches a much larger
model without it** (e.g. Qwen-7B+MCP ≈ Qwen-32B-BARE), i.e. the tool
substrate carries more value than parameter count.

The subset is small on purpose — 6 representative tasks × 2 conditions × 3
models × 3 seeds = **108 cells** (vs. 225 in HOE-v2).

## Models

Weights on the PortableSSD:

```
/media/alexander/PortableSSD/AstroNet/models/qwen2.5-7b
/media/alexander/PortableSSD/AstroNet/models/qwen2.5-14b
/media/alexander/PortableSSD/AstroNet/models/qwen2.5-32b
```

(Qwen2.5-Instruct variants; native tool-call format.)

## Recommended runtime: vLLM (OpenAI-compatible server)

vLLM exposes an OpenAI-format endpoint, so LangGraph talks to all three
models through identical `ChatOpenAI` clients with proper function calling.

```bash
# in a fresh terminal — one per model size
cd ~/Schreibtisch/open-fem-agent/langgraph_eval
./start_vllm.sh 7b     # → http://localhost:8000/v1
./start_vllm.sh 14b    # → http://localhost:8001/v1
./start_vllm.sh 32b    # → http://localhost:8002/v1
```

Use `--quantization awq` or `--quantization gptq` on the 32 B model if VRAM
is tight; details in `start_vllm.sh`.

## Driver

```bash
cd ~/Schreibtisch/open-fem-agent
.venv-lg/bin/python langgraph_eval/run_eval.py \
    --models 7b 14b 32b \
    --conditions BARE MCP \
    --seeds 0 1 2 \
    --out runs/openweight/
```

Each cell lands at
`runs/openweight/<model>_<condition>_<task>_seed<n>/result.txt` and the
driver writes `summary.csv` next to it.

## Grading

Same gates as the HOE-v2 appendix — `grade.py` re-applies them and emits a
two-axis plot (`scaling.pdf`) of pass-rate vs. model size, BARE vs. MCP.

## Conflict with the live HOE-v2 campaign

**Don't start this while the 225-cell run is in progress** — the MCP server
process is heavyweight, and vLLM will compete for GPU. After the HOE-v2 ping
arrives, the machine is yours.

## venv layout

| venv | for |
|------|-----|
| `.venv` (repo root) | OASiS server (used by the HOE-v2 campaign) — do not modify |
| `.venv-lg`          | LangGraph driver (`requirements-langgraph.txt`) |
| separate conda env  | vLLM (`requirements-vllm.txt`); needs CUDA-matched torch |
