#!/usr/bin/env python3
"""Smoke tests for the LangGraph + OASiS-MCP scaffold.

Run from the repo root:

    .venv-lg/bin/python langgraph_eval/test_scaffold.py

The end-to-end sub-agent spawn test (T8) uses an in-process mock OpenAI
server so it works without vLLM/Qwen. T1-T7 are real:

* T1 imports — agent.py composes
* T2 run_bash — actually runs a shell command in a sandbox
* T3 read_file / write_file — actually round-trip a file
* T4 web_search — actually hits DuckDuckGo and returns hits
* T5 OASiS MCP enumeration — spawns a real OASiS server, lists tools
* T6 OFA_DISABLE_PITFALLS — calls `knowledge` over MCP and asserts the
  pitfall keys are stripped end-to-end (server-side masking propagates
  through the MCP bridge)
* T7 spawn_subagent depth limiter — depth==2 rejects (synthetic check)
* T8 end-to-end spawn — agent built against the mock OpenAI server,
  parent emits a spawn_subagent tool_call, the critic sub-agent fires
  with its own request, returns "APPROVED…", parent terminates
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "langgraph_eval"))


EXPECTED_OASIS = {
    "prepare_simulation", "knowledge", "discover", "examples",
    "developer", "generate_mesh", "run_simulation", "run_with_generator",
    "coupled_solve", "transfer_field", "visualize", "session_insights",
    "rediscover_backends",
}


_RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, fn):
    print(f"  …  {name}", flush=True)
    try:
        fn()
        _RESULTS.append((name, True, ""))
        print(f"  PASS  {name}", flush=True)
    except AssertionError as e:
        _RESULTS.append((name, False, str(e)))
        print(f"  FAIL  {name}\n        {e}", flush=True)
    except Exception as e:
        traceback.print_exc()
        _RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        print(f"  ERR   {name}: {type(e).__name__}: {e}", flush=True)


# ───────────────────────────────────────────────────────────────
# T1
# ───────────────────────────────────────────────────────────────
def t_imports():
    import agent
    assert agent.PORTS == {"7b": 8000, "14b": 8001, "32b": 8002}
    assert callable(agent.build_bare_agent)
    assert callable(agent.build_mcp_agent)


# ───────────────────────────────────────────────────────────────
# T2
# ───────────────────────────────────────────────────────────────
def t_bash():
    import agent
    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        bash = agent._bash_tool_for(wd)
        out = bash.invoke({"command": "echo hello && pwd"})
        assert "hello" in out, out
        assert str(wd) in out, out


# ───────────────────────────────────────────────────────────────
# T3
# ───────────────────────────────────────────────────────────────
def t_rw():
    import agent
    with tempfile.TemporaryDirectory() as d:
        wd = Path(d)
        rt, wt = agent._read_write_tools_for(wd)
        wt.invoke({"path": "foo.txt", "content": "abc123"})
        assert (wd / "foo.txt").read_text() == "abc123"
        assert rt.invoke({"path": "foo.txt"}) == "abc123"


# ───────────────────────────────────────────────────────────────
# T4 — live web search
# ───────────────────────────────────────────────────────────────
def t_web_search():
    """Live web search. Tries several queries so a single DDG rate-limit
    doesn't flake the test; passes if ANY query returns real hits."""
    import agent
    queries = [
        "deal.II tutorial step-7",
        "FEniCSx documentation",
        "NGSolve Schaefer Turek DFG cylinder benchmark",
    ]
    last = ""
    for q in queries:
        out = agent.web_search.invoke({"query": q, "max_results": 3})
        last = out
        assert "[web_search unavailable" not in out, out
        if "http" in out.lower() and len(out) > 60:
            return  # got real hits
    raise AssertionError(
        "web_search returned no usable results from any of the test "
        f"queries (DDG may be rate-limiting). Last output: {last!r}")


# ───────────────────────────────────────────────────────────────
# T5 — OASiS MCP enumeration
# ───────────────────────────────────────────────────────────────
def _mcp_client(env_override: dict | None = None):
    from langchain_mcp_adapters.client import MultiServerMCPClient
    env = os.environ.copy()
    env.pop("OFA_DISABLE_CRITIC", None)
    env.pop("OFA_DISABLE_PITFALLS", None)
    if env_override:
        env.update(env_override)
    env["PYTHONPATH"] = str(REPO / "src")
    env["FOURC_ROOT"] = env.get("FOURC_ROOT", str(Path.home() / "4C"))
    env["FOURC_BINARY"] = env.get(
        "FOURC_BINARY", str(Path.home() / "4C/build/4C"))
    env["LD_LIBRARY_PATH"] = env.get(
        "LD_LIBRARY_PATH", "/opt/4C-dependencies/lib")
    return MultiServerMCPClient({"oasis": {
        "command": str(REPO / ".venv/bin/python"),
        "args": ["-m", "server"],
        "cwd": str(REPO / "src"),
        "env": env,
        "transport": "stdio",
    }})


def t_oasis_enumeration():
    client = _mcp_client()
    tools = asyncio.run(client.get_tools())
    names = {t.name for t in tools}
    missing = EXPECTED_OASIS - names
    assert not missing, \
        f"missing OASiS tools: {missing}; got: {sorted(names)}"


# ───────────────────────────────────────────────────────────────
# T6 — masking through MCP bridge
# ───────────────────────────────────────────────────────────────
async def _call_knowledge(env_override):
    client = _mcp_client(env_override)
    tools = await client.get_tools()
    knowledge = next(t for t in tools if t.name == "knowledge")
    out = await knowledge.ainvoke(
        {"topic": "physics", "solver": "skfem", "physics": "poisson"})
    return str(out)


def t_masking_pitfalls():
    full = asyncio.run(_call_knowledge({}))
    masked = asyncio.run(_call_knowledge({"OFA_DISABLE_PITFALLS": "1"}))
    # Sanity: under MCP_FULL, the unmasked text must contain a pitfall marker
    has_pf_full = ('"pitfalls"' in full or "Signal:" in full
                   or "pitfall" in full.lower())
    assert has_pf_full, (
        "MCP_FULL knowledge response unexpectedly contains no pitfall content; "
        "the masking test below cannot be conclusive. Excerpt: "
        f"{full[:400]!r}")
    # Masked: pitfall keys and Signal anchors must be absent
    assert '"pitfalls"' not in masked, \
        f"`pitfalls` key leaked under OFA_DISABLE_PITFALLS=1: {masked[:400]!r}"
    assert "Signal:" not in masked, \
        f"`Signal:` anchor leaked under OFA_DISABLE_PITFALLS=1: {masked[:400]!r}"


# ───────────────────────────────────────────────────────────────
# T7 — spawn_subagent depth limiter
# ───────────────────────────────────────────────────────────────
def t_spawn_depth_limit():
    import agent
    spawn = agent._make_spawn_subagent_tool(
        size="7b", seed=0, workdir=Path("/tmp"),
        parent_tools=[], depth=2)
    out = spawn.invoke({"role": "critic", "task": "x", "context": ""})
    assert "denied" in out and "max depth" in out, \
        f"depth limiter did not refuse at depth=2: {out!r}"


# ───────────────────────────────────────────────────────────────
# T8 — end-to-end spawn_subagent via mock OpenAI server
# ───────────────────────────────────────────────────────────────
def t_spawn_endtoend():
    import _mock_openai_server as mock
    import agent
    from langchain_openai import ChatOpenAI

    port = 18234
    srv, _ = mock.start(port=port)
    mock.reset_log()
    try:
        orig_llm = agent._llm
        agent._llm = lambda size, *, temperature, seed: ChatOpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="not-used",
            model="mock-qwen",
            temperature=temperature,
            seed=seed,
            timeout=15,
        )
        try:
            with tempfile.TemporaryDirectory() as d:
                bare = agent.build_bare_agent(
                    size="7b", seed=0, workdir=Path(d))
                result = bare.invoke(
                    {"messages": [
                        ("user",
                         "Plan a Poisson MMS run on a 32x32 grid. "
                         "Use the critic.")]},
                    config={"recursion_limit": 25},
                )
            log = mock.get_log()
            n_parent = sum(1 for e in log if not e["is_critic"])
            n_critic = sum(1 for e in log if e["is_critic"])
            assert n_parent >= 2, (
                f"parent should have been called at least twice (initial + "
                f"after tool result); got {n_parent}. log={log}")
            assert n_critic >= 1, (
                f"critic sub-agent was never invoked; spawn_subagent did "
                f"not fire. log={log}")
            # Final assistant message should be the parent's terminal text
            final_text = result["messages"][-1].content or ""
            assert "critic approved" in final_text.lower(), \
                f"unexpected final message: {final_text!r}"
        finally:
            agent._llm = orig_llm
    finally:
        srv.shutdown()


# ───────────────────────────────────────────────────────────────
# Driver
# ───────────────────────────────────────────────────────────────
CHECKS = [
    ("T1 imports", t_imports),
    ("T2 run_bash", t_bash),
    ("T3 read_file/write_file", t_rw),
    ("T4 web_search (live)", t_web_search),
    ("T5 OASiS MCP tool enumeration", t_oasis_enumeration),
    ("T6 masking propagates over MCP", t_masking_pitfalls),
    ("T7 spawn_subagent depth limit", t_spawn_depth_limit),
    ("T8 end-to-end critic spawn via mock LLM", t_spawn_endtoend),
]


def main():
    t0 = time.time()
    for name, fn in CHECKS:
        check(name, fn)
    dt = time.time() - t0
    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    print(f"\n{passed}/{len(_RESULTS)} passed in {dt:.1f}s")
    if passed != len(_RESULTS):
        for name, ok, why in _RESULTS:
            if not ok:
                print(f"  - {name}: {why}")
        sys.exit(1)


if __name__ == "__main__":
    main()
