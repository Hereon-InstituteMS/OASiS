"""LangGraph agents for the open-weight ablation.

Two factory functions:

* :func:`build_bare_agent`  — host-side toolset only (bash + web search +
  spawn_subagent). No OASiS MCP. Mirrors what Claude has in v1 BARE.
* :func:`build_mcp_agent`   — same host-side toolset PLUS every OASiS MCP
  tool attached via langchain-mcp-adapters (MCP_FULL semantics).

Both conditions get parity with what Claude Code offers natively:

| Host-side tool       | Why both conditions need it                        |
|---------------------|----------------------------------------------------|
| ``run_bash``        | Equivalent of Claude's Bash; runs scripts/solvers  |
| ``read_file``       | Equivalent of Claude's Read                        |
| ``write_file``      | Equivalent of Claude's Write                       |
| ``web_search``      | Equivalent of Claude's WebSearch (literature/benchmarks) |
| ``spawn_subagent``  | Equivalent of Claude's Agent tool; needed so the   |
|                     | model can fulfil the MANDATORY CRITIC protocol the |
|                     | OASiS server prompts it to follow                  |

MCP_FULL additionally exposes all OASiS tools (``prepare_simulation``,
``knowledge``, ``discover``, ``examples``, ``developer``, ``generate_mesh``,
``run_simulation``, ``run_with_generator``, ``coupled_solve``,
``transfer_field``, ``visualize``, ``session_insights``,
``rediscover_backends``) — picked up automatically from the running OASiS
server, so any future tool added there is included without code changes
here.

All LLM calls go through ``langchain_openai.ChatOpenAI`` pointed at a local
vLLM server, which surfaces Qwen2.5's native tool-call format through the
OpenAI schema.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Sequence

from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

REPO = Path(__file__).resolve().parents[1]

PORTS = {"7b": 8000, "14b": 8001, "32b": 8002}

_CRITIC_BLOCK = (
    "MANDATORY CRITIC: For every major step (problem setup, parameter "
    "choices, mesh/discretisation, BCs, solver choice, result "
    "interpretation), call `spawn_subagent` with role=\"critic\" and a "
    "ruthlessly skeptical task description. Pass the current state to the "
    "critic. Only proceed once the critic returns an explicit \"approved\" "
    "verdict. Do not approve your own work.\n"
)

BARE_SYSTEM = (
    "You are a finite-element simulation assistant. You will be given a "
    "problem statement and a writable result file path. You have host-side "
    "tools only — no FEM-aware MCP layer. Solve the problem from first "
    "principles using whatever solvers are installed on the system. Write "
    "scripts (Python / 4C YAML / etc.), run them with `run_bash`, and "
    "produce the requested RESULT lines in the result file.\n\n"
    "Tools available: run_bash, read_file, write_file, web_search, "
    "spawn_subagent.\n\n"
    + _CRITIC_BLOCK
)

MCP_SYSTEM = (
    "You are connected to the OASiS MCP server with its full toolset "
    "(prepare_simulation, knowledge, discover, examples, developer, "
    "generate_mesh, run_simulation, run_with_generator, coupled_solve, "
    "transfer_field, visualize, session_insights). Follow the OASiS "
    "workflow: discover → prepare_simulation → examples → "
    "run_(simulation|with_generator) → visualize. Use `knowledge` for "
    "physics + pitfalls, `developer` for source/architecture lookups, and "
    "`coupled_solve` / `transfer_field` for cross-code coupling.\n\n"
    "Host-side tools (also available): run_bash, read_file, write_file, "
    "web_search, spawn_subagent.\n\n"
    + _CRITIC_BLOCK
)


# ────────────────────────────────────────────────────────────────────
# LLM factory
# ────────────────────────────────────────────────────────────────────
def _llm(size: str, *, temperature: float, seed: int) -> ChatOpenAI:
    port = PORTS[size]
    return ChatOpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="not-used-by-vllm",
        model=f"qwen2.5-{size}",
        temperature=temperature,
        seed=seed,
        timeout=600,
    )


# ────────────────────────────────────────────────────────────────────
# Host-side tools (parity with Claude Code's native surface)
# ────────────────────────────────────────────────────────────────────
def _bash_tool_for(workdir: Path):
    @tool
    def run_bash(command: str) -> str:
        """Run a shell command inside the cell's sandbox dir. Returns stdout+stderr (truncated to 12 KB)."""
        try:
            p = subprocess.run(
                ["bash", "-lc", command],
                cwd=workdir, capture_output=True, text=True, timeout=900,
            )
            out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
            return out[-12000:] if len(out) > 12000 else out
        except subprocess.TimeoutExpired:
            return "[timeout after 900s]"
    return run_bash


def _read_write_tools_for(workdir: Path):
    @tool
    def read_file(path: str, max_bytes: int = 200_000) -> str:
        """Read a file (absolute path, or relative to the cell sandbox)."""
        p = Path(path)
        if not p.is_absolute():
            p = workdir / p
        try:
            data = p.read_bytes()[:max_bytes]
            return data.decode("utf-8", errors="replace")
        except FileNotFoundError:
            return f"[file not found: {p}]"

    @tool
    def write_file(path: str, content: str) -> str:
        """Write `content` to `path` (relative paths resolve inside the cell sandbox)."""
        p = Path(path)
        if not p.is_absolute():
            p = workdir / p
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} chars to {p}"

    return [read_file, write_file]


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web (DuckDuckGo). Returns up to max_results result snippets."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return ("[web_search unavailable: install duckduckgo-search "
                "(pip install duckduckgo-search) to enable]")
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))
        return "\n\n".join(
            f"{h.get('title')}\n{h.get('href')}\n{h.get('body')}" for h in hits
        ) or "[no results]"
    except Exception as e:
        return f"[web_search error: {type(e).__name__}: {e}]"


# ────────────────────────────────────────────────────────────────────
# spawn_subagent — sibling LangGraph agent on the same vLLM endpoint
# ────────────────────────────────────────────────────────────────────
def _make_spawn_subagent_tool(
    *, size: str, seed: int, workdir: Path,
    parent_tools: Sequence[BaseTool], depth: int,
):
    """Returns a tool that spawns a depth-limited sibling agent.

    The sub-agent reuses the same vLLM server (cheap on memory) with a
    slightly higher temperature and a derived seed. It gets the same
    workdir-bound bash/read/write/web_search and the parent's OASiS
    tools, but its own ``spawn_subagent`` is *not* re-installed beyond
    depth 1 to prevent runaway recursion.
    """

    @tool
    def spawn_subagent(role: str, task: str, context: str = "") -> str:
        """Spawn a sub-agent. role∈{critic, researcher, verifier}; task = what it should do; context = facts to pass in.

        The critic role should ruthlessly challenge the parent's setup; the
        verifier should re-derive numbers independently; the researcher
        should look things up via web_search and the OASiS knowledge tool.
        Returns the sub-agent's final message text.
        """
        if depth >= 2:
            return "[spawn_subagent denied: max depth 2 to prevent recursion]"
        sub_tools = list(parent_tools)
        # Give the sub-agent the same host tools, but no further nesting:
        sub_tools = [t for t in sub_tools if t.name != "spawn_subagent"]
        if role == "critic":
            sys = (
                "You are a ruthlessly critical reviewer. Challenge every "
                "parameter choice, check units, look for sign errors, "
                "verify BCs, and validate against the literature via "
                "web_search. Respond with one of: APPROVED: <reason> | "
                "REJECTED: <issue and required fix>."
            )
        elif role == "verifier":
            sys = (
                "You are an independent verifier. Re-derive the requested "
                "quantity from first principles or by an alternative "
                "method/solver, then compare with the parent's number."
            )
        else:
            sys = (
                "You are a research assistant. Look up authoritative "
                "sources for the requested information and summarise."
            )
        sub_llm = _llm(size, temperature=0.3, seed=seed + 100 + depth)
        sub_agent = create_react_agent(
            sub_llm, tools=sub_tools,
            state_modifier=SystemMessage(sys),
        )
        msg = f"Task: {task}\n\nContext provided by parent:\n{context}"
        try:
            out = sub_agent.invoke(
                {"messages": [("user", msg)]},
                config={"recursion_limit": 40},
            )
            return out["messages"][-1].content
        except Exception as e:
            return f"[sub-agent error: {type(e).__name__}: {e}]"

    return spawn_subagent


# ────────────────────────────────────────────────────────────────────
# OASiS MCP tool loader (langchain-mcp-adapters)
# ────────────────────────────────────────────────────────────────────
def _load_oasis_mcp_tools() -> list[BaseTool]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    env = os.environ.copy()
    env.pop("OFA_DISABLE_CRITIC", None)
    env.pop("OFA_DISABLE_PITFALLS", None)
    env["FOURC_ROOT"] = env.get("FOURC_ROOT", str(Path.home() / "4C"))
    env["FOURC_BINARY"] = env.get(
        "FOURC_BINARY", str(Path.home() / "4C/build/4C"))
    env["LD_LIBRARY_PATH"] = env.get(
        "LD_LIBRARY_PATH", "/opt/4C-dependencies/lib")
    env["PYTHONPATH"] = str(REPO / "src")

    client = MultiServerMCPClient({
        "oasis": {
            "command": str(REPO / ".venv/bin/python"),
            "args": ["-m", "server"],
            "cwd": str(REPO / "src"),
            "env": env,
            "transport": "stdio",
        }
    })
    return asyncio.run(client.get_tools())


# ────────────────────────────────────────────────────────────────────
# Public factories
# ────────────────────────────────────────────────────────────────────
def _host_tools(workdir: Path, *, size: str, seed: int,
                parent_tools: list[BaseTool], depth: int) -> list[BaseTool]:
    tools: list[BaseTool] = []
    tools.append(_bash_tool_for(workdir))
    tools.extend(_read_write_tools_for(workdir))
    tools.append(web_search)
    spawn = _make_spawn_subagent_tool(
        size=size, seed=seed, workdir=workdir,
        parent_tools=parent_tools + tools, depth=depth,
    )
    tools.append(spawn)
    return tools


def build_bare_agent(*, size: str, seed: int, workdir: Path, depth: int = 0):
    tools = _host_tools(workdir, size=size, seed=seed,
                        parent_tools=[], depth=depth)
    llm = _llm(size, temperature=0.2, seed=seed)
    return create_react_agent(llm, tools=tools,
                              state_modifier=SystemMessage(BARE_SYSTEM))


def build_mcp_agent(*, size: str, seed: int, workdir: Path, depth: int = 0):
    mcp_tools = _load_oasis_mcp_tools()
    host = _host_tools(workdir, size=size, seed=seed,
                       parent_tools=mcp_tools, depth=depth)
    llm = _llm(size, temperature=0.2, seed=seed)
    return create_react_agent(llm, tools=mcp_tools + host,
                              state_modifier=SystemMessage(MCP_SYSTEM))


__all__ = ["build_bare_agent", "build_mcp_agent"]
