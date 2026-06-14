"""Agent runner that streams structured events to a single WebSocket.

Wraps the existing ``langgraph_eval.agent`` factories and surfaces every
intermediate step the user wants to see in the UI:

* user_msg / agent_msg / agent_chunk
* tool_call_pending  → in plan mode, blocks waiting for an approve/reject
                       event from the client before executing
* tool_call_executing / tool_result
* subagent_spawned / subagent_returned (via a callback we wire into
  spawn_subagent so its children are visible)
* token_count after each LLM call
* error / done

The runner is mode-aware: plan mode gates every tool call; accept mode
auto-approves; autonomous is identical to accept but the UI is told not
to interrupt.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from . import config

REPO = config.REPO
sys.path.insert(0, str(REPO / "langgraph_eval"))


# ───────────────────────────────────────────────────────────────────
# Mock LLM (no GPU) — drives a deterministic plan→spawn_critic→done
# chain so the UI is usable without vLLM. Real models go through
# ChatOpenAI as configured in langgraph_eval/agent.py.
#
# Implementation reuses the langgraph_eval/_mock_openai_server.py
# fake OpenAI server (proven by the langgraph_eval smoke tests). We
# start one thread per app process on first mock use; subsequent
# sessions all point at the same loopback endpoint.
# ───────────────────────────────────────────────────────────────────
_MOCK_PORT = 18234
_MOCK_STARTED = False


def _ensure_mock_server():
    global _MOCK_STARTED
    if _MOCK_STARTED:
        return
    import _mock_openai_server as mock_srv  # from langgraph_eval/
    mock_srv.start(port=_MOCK_PORT)
    _MOCK_STARTED = True


def _mock_chat_model():
    _ensure_mock_server()
    from langchain_openai import ChatOpenAI
    # disable_streaming forces astream_events to go through the
    # non-streaming agenerate path, so the mock server's plain JSON
    # response works (no SSE needed).
    return ChatOpenAI(
        base_url=f"http://127.0.0.1:{_MOCK_PORT}/v1",
        api_key="not-used", model="mock-qwen",
        temperature=0.0, timeout=30,
        disable_streaming=True,
    )


# ───────────────────────────────────────────────────────────────────
# Per-session workdir
# ───────────────────────────────────────────────────────────────────
def _session_workdir(session_id: str) -> Path:
    p = config.SANDBOX_ROOT / f"webui_{session_id}" / "work"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ───────────────────────────────────────────────────────────────────
# Approval gate (plan mode)
# ───────────────────────────────────────────────────────────────────
class ApprovalGate:
    """A future-per-pending-call gate. The UI signals approve/reject via
    ``resolve`` from a separate code path."""

    def __init__(self):
        self._pending: dict[str, asyncio.Future] = {}

    def open(self, call_id: str) -> asyncio.Future:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._pending[call_id] = fut
        return fut

    def resolve(self, call_id: str, approved: bool, reason: str = ""):
        fut = self._pending.pop(call_id, None)
        if fut and not fut.done():
            fut.set_result({"approved": approved, "reason": reason})


# ───────────────────────────────────────────────────────────────────
# Tool wrapping for mode gating + event emission
# ───────────────────────────────────────────────────────────────────
def _wrap_tool(tool, *, emitter, get_mode, gate, agent_label="agent"):
    """Return a copy of ``tool`` whose invoke emits events and (when in
    plan mode) waits for explicit approval before running.

    The original tool's func is preserved; we only intercept invocation.
    """
    from langchain_core.tools import BaseTool

    class Gated(BaseTool):
        name: str = tool.name
        description: str = tool.description
        args_schema: Any = getattr(tool, "args_schema", None)

        async def _arun(self, *args, **kwargs):
            call_id = f"tc_{uuid.uuid4().hex[:8]}"
            await emitter({"type": "tool_call_pending",
                           "call_id": call_id, "tool": tool.name,
                           "args": kwargs, "agent": agent_label})
            mode = get_mode()
            if mode == "plan":
                fut = gate.open(call_id)
                decision = await fut
                if not decision["approved"]:
                    await emitter({"type": "tool_call_rejected",
                                   "call_id": call_id,
                                   "reason": decision.get("reason", "")})
                    return f"[rejected by user: {decision.get('reason','')}]"
            await emitter({"type": "tool_call_executing",
                           "call_id": call_id, "tool": tool.name})
            try:
                if hasattr(tool, "ainvoke"):
                    result = await tool.ainvoke(kwargs)
                else:
                    result = tool.invoke(kwargs)
            except Exception as e:
                await emitter({"type": "tool_error",
                               "call_id": call_id,
                               "error": f"{type(e).__name__}: {e}"})
                raise
            await emitter({"type": "tool_result",
                           "call_id": call_id, "tool": tool.name,
                           "result": str(result)[:8000]})
            return result

        def _run(self, *args, **kwargs):
            return asyncio.get_event_loop().run_until_complete(
                self._arun(*args, **kwargs))

    return Gated()


# ───────────────────────────────────────────────────────────────────
# Agent factories with WebUI hooks
# ───────────────────────────────────────────────────────────────────
def build_agent_for_session(*, model: str, mcp_on: bool,
                            workdir: Path,
                            emitter,
                            get_mode,
                            gate: ApprovalGate):
    """Build a LangGraph ReAct agent with all WebUI hooks wired in.

    * ``model`` is a key from ``config.MODELS``. ``mock`` skips vLLM.
    * ``mcp_on`` attaches OASiS via langchain-mcp-adapters when True.
    * ``emitter`` is an async function ``(event_dict) -> None`` used to
      stream events back over the WebSocket.
    * ``get_mode`` is a callable returning the current mode string.
    * ``gate`` is the per-session ApprovalGate.
    """
    import agent as la

    # ── LLM
    if model == "mock":
        llm = _mock_chat_model()
    else:
        if model not in config.MODELS:
            raise ValueError(f"unknown model: {model}")
        port = config.MODELS[model]["port"]
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            base_url=f"http://localhost:{port}/v1",
            api_key="not-used", model=model,
            temperature=0.2, timeout=900,
        )

    # ── OASiS MCP tools (optional)
    mcp_tools = la._load_oasis_mcp_tools() if mcp_on else []

    # ── Host tools (bash/read/write/web_search/spawn_subagent)
    host = []
    host.append(la._bash_tool_for(workdir))
    host.extend(la._read_write_tools_for(workdir))
    host.append(la.web_search)

    # spawn_subagent: wrap so we can emit subagent_spawned events
    base_spawn = la._make_spawn_subagent_tool(
        size="7b", seed=0, workdir=workdir,
        parent_tools=mcp_tools + host, depth=0)

    async def spawn_subagent_emitting(role: str, task: str,
                                      context: str = "") -> str:
        sa_id = f"sa_{uuid.uuid4().hex[:8]}"
        await emitter({"type": "subagent_spawned", "sa_id": sa_id,
                       "role": role, "task": task, "context": context})
        try:
            # base_spawn.invoke calls a sync LangGraph subagent. Running
            # it on the current event loop would deadlock; offload to a
            # worker thread so the outer await keeps draining events.
            res = await asyncio.to_thread(
                base_spawn.invoke,
                {"role": role, "task": task, "context": context})
        except Exception as e:
            res = f"[sub-agent error: {type(e).__name__}: {e}]"
        await emitter({"type": "subagent_returned",
                       "sa_id": sa_id, "result": str(res)[:6000]})
        return res

    from langchain_core.tools import StructuredTool
    spawn_wrapped = StructuredTool.from_function(
        coroutine=spawn_subagent_emitting,
        name="spawn_subagent",
        description=base_spawn.description,
    )
    host.append(spawn_wrapped)

    # ── Gate every tool through the wrapper for mode gating
    gated = [_wrap_tool(t, emitter=emitter, get_mode=get_mode, gate=gate,
                        agent_label="main") for t in mcp_tools + host]

    from langgraph.prebuilt import create_react_agent
    prompt = (la.MCP_SYSTEM if mcp_on else la.BARE_SYSTEM)
    return create_react_agent(llm, tools=gated, prompt=prompt)


# ───────────────────────────────────────────────────────────────────
# Streamed turn
# ───────────────────────────────────────────────────────────────────
async def stream_turn(*, agent, user_text: str, emitter):
    """Run one user turn. Streams chunks/events via ``emitter`` and
    returns the final message text."""
    final_text = ""
    inputs = {"messages": [("user", user_text)]}
    try:
        async for event in agent.astream_events(inputs, version="v2"):
            kind = event.get("event")
            name = event.get("name")
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if chunk and getattr(chunk, "content", None):
                    await emitter({"type": "agent_chunk",
                                   "text": chunk.content})
            elif kind == "on_chat_model_end":
                gen = event["data"].get("output")
                if gen is not None and hasattr(gen, "usage_metadata"):
                    um = gen.usage_metadata or {}
                    await emitter({"type": "token_count",
                                   "input": um.get("input_tokens"),
                                   "output": um.get("output_tokens")})
                if gen is not None and getattr(gen, "content", ""):
                    final_text = gen.content
            elif kind == "on_chain_end" and name == "LangGraph":
                msgs = event["data"].get("output", {}).get("messages") or []
                if msgs:
                    last = msgs[-1]
                    final_text = (getattr(last, "content", "")
                                  or final_text)
    except Exception as e:
        await emitter({"type": "error",
                       "message": f"{type(e).__name__}: {e}",
                       "traceback": traceback.format_exc()[-4000:]})
        raise
    await emitter({"type": "done", "final_text": final_text})
    return final_text
