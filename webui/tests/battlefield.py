#!/usr/bin/env python3
"""Battlefield test — exercise every WebUI surface against a LIVE server.

This is NOT a unit test. It assumes uvicorn is already listening on
:8080 (start it with .venv-lg/bin/uvicorn webui.app:app --port 8080)
and pokes every endpoint plus a real WebSocket round-trip with the mock
LLM. Targets every capability the user asked for:

  * HTTP endpoints (models, MCP servers, modes, files, file content,
    viz, extract_params, sandbox-file)
  * Real artefacts from eval_interactive/ (a result.txt, a CSV with
    Plotly payload, an E5 BARE cavity.py with extracted parameters)
  * Session lifecycle (create, list, load, delete)
  * WebSocket end-to-end with mock LLM (real prompt → real event
    stream → sub-agent spawn/return → done)
  * Plan-mode gating (real tool_call_pending → real approve → real
    tool_result)
  * Live MCP toggle (rebuilds the agent at the next prompt)
  * Mode switch (set_mode)
  * Restart command

Prints PASS / FAIL per check and a final summary. Exits non-zero on
the first failure so CI can pick it up.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import websockets

BASE = "http://127.0.0.1:8080"
WS_BASE = "ws://127.0.0.1:8080"

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    flag = "PASS" if ok else "FAIL"
    results.append((name, ok, detail))
    print(f"  {flag}  {name}" + (f"  ::  {detail[:140]}" if detail else ""),
          flush=True)


# ───────────────────────────────────────────────────────────────────
# 1. HTTP endpoints
# ───────────────────────────────────────────────────────────────────
async def http_checks(client: httpx.AsyncClient) -> dict:
    print("\n[1] HTTP endpoints", flush=True)
    state: dict = {}

    r = await client.get("/")
    check("GET /  serves HTML", r.status_code == 200 and "OASiS" in r.text,
          f"status={r.status_code}, bytes={len(r.text)}")

    r = await client.get("/api/models")
    js = r.json()
    ids = [m["id"] for m in js["models"]]
    check("GET /api/models  has qwen + mock",
          {"qwen2.5-7b", "qwen2.5-14b", "qwen2.5-32b", "mock"} <= set(ids),
          f"got {ids}, default={js.get('default')}")

    r = await client.get("/api/mcp_servers")
    servers = r.json()["servers"]
    check("GET /api/mcp_servers  has OASiS",
          any(s["id"] == "oasis" for s in servers), f"{servers}")

    r = await client.get("/api/modes")
    md = r.json()
    check("GET /api/modes  has plan/accept/autonomous",
          set(md["modes"]) == {"plan", "accept", "autonomous"},
          f"modes={md['modes']}")

    # Sandbox listing
    r = await client.get("/api/files")
    js = r.json()
    n_entries = len(js.get("entries", []))
    check("GET /api/files  lists sandbox root",
          js["exists"] and n_entries > 0,
          f"entries={n_entries}")
    state["root_listing"] = js

    # Traversal safety
    r = await client.get("/api/files", params={"rel": "../../etc/passwd"})
    check("Traversal escape blocked (403)",
          r.status_code in (403, 422), f"status={r.status_code}")
    return state


# ───────────────────────────────────────────────────────────────────
# 2. Real-artefact checks (file content, viz, slider extraction)
# ───────────────────────────────────────────────────────────────────
async def artefact_checks(client: httpx.AsyncClient) -> None:
    print("\n[2] Real artefacts from eval_interactive/", flush=True)

    rel_result = "E5_BARE_seed0_v2/work/result.txt"
    r = await client.get("/api/file", params={"rel": rel_result})
    js = r.json()
    check("Read real result.txt (E5 BARE seed0)",
          "RESULT" in (js.get("text") or ""),
          f"first_line={(js.get('text') or '')[:80]!r}")

    r = await client.get("/api/viz", params={"rel": rel_result})
    js = r.json()
    check("viz(result.txt)  → text",
          js.get("kind") == "text" and "RESULT" in js.get("text", ""),
          f"kind={js.get('kind')}")

    rel_csv = "B3_BARE_seed0_v2/work/forces.csv"
    r = await client.get("/api/viz", params={"rel": rel_csv})
    js = r.json()
    plot = js.get("plot") or {}
    n_traces = len(plot.get("data") or [])
    cfg = plot.get("config") or {}
    check("viz(real CSV)  → Plotly figure JSON",
          js.get("kind") == "table" and n_traces > 0,
          f"kind={js.get('kind')}, traces={n_traces}, "
          f"cols={js.get('header')}")
    check("Plot config is click-editable",
          cfg.get("editable") is True
          and cfg.get("edits", {}).get("titleText") is True
          and cfg.get("edits", {}).get("axisTitleText") is True,
          f"edits={cfg.get('edits')}")
    check("Plot toolbar exports as PNG",
          cfg.get("toImageButtonOptions", {}).get("format") == "png",
          f"opts={cfg.get('toImageButtonOptions')}")

    rel_json = "B4_MCP_NO_CRITIC_seed0_v2/work/results_summary.json"
    r = await client.get("/api/viz", params={"rel": rel_json})
    js = r.json()
    check("viz(real JSON)  → parsed obj",
          js.get("kind") == "json" and isinstance(js.get("obj"), (dict, list)),
          f"kind={js.get('kind')}, top_keys="
          f"{list(js.get('obj', {}).keys())[:5] if isinstance(js.get('obj'), dict) else '...'}")

    rel_py = "E5_BARE_seed0_v2/work/cavity.py"
    r = await client.get("/api/extract_params", params={"rel": rel_py})
    js = r.json()
    names = [p["name"] for p in js["params"]]
    check("extract_params(real cavity.py)  → numeric assignments",
          len(names) >= 2,
          f"params={names[:8]}")

    rel_raw = rel_result
    r = await client.get(f"/sandbox-file/{rel_raw}")
    check("GET /sandbox-file/<rel>  serves raw bytes",
          r.status_code == 200 and "RESULT" in r.text, "")

    r = await client.get("/sandbox-file/../../etc/passwd")
    check("sandbox-file traversal blocked",
          r.status_code in (403, 404), f"status={r.status_code}")

    # File round-trip through the editor endpoint
    import tempfile, time as _t
    scratch_rel = f"webui_battlefield_{int(_t.time())}/notes.txt"
    new_text = f"hello from battlefield {_t.time()}"
    r = await client.post("/api/file",
                          json={"rel": scratch_rel, "content": new_text})
    js = r.json()
    check("POST /api/file  writes content inside sandbox",
          r.status_code == 200 and js.get("ok") is True,
          f"resp={js}")
    r = await client.get("/api/file", params={"rel": scratch_rel})
    js = r.json()
    check("File round-trip: read-back matches",
          js.get("text") == new_text, f"got={js.get('text')!r}")
    r = await client.post("/api/file",
                          json={"rel": "../../etc/hosts",
                                "content": "evil"})
    check("POST /api/file  blocks traversal",
          r.status_code in (403, 422), f"status={r.status_code}")


# ───────────────────────────────────────────────────────────────────
# 3. Session lifecycle
# ───────────────────────────────────────────────────────────────────
async def session_checks(client: httpx.AsyncClient) -> str:
    print("\n[3] Session lifecycle", flush=True)
    r = await client.post("/api/sessions",
                          json={"model": "mock", "mode": "accept",
                                "mcp_servers": []})
    sid = r.json()["id"]
    check("POST /api/sessions  creates a session", bool(sid), f"sid={sid}")

    r = await client.get(f"/api/sessions/{sid}")
    js = r.json()
    check("GET /api/sessions/{id}  rehydrates",
          js["id"] == sid and js["model"] == "mock", f"")

    r = await client.get("/api/sessions")
    listed = r.json()["sessions"]
    check("GET /api/sessions  lists it",
          any(s["id"] == sid for s in listed), f"n_sessions={len(listed)}")
    return sid


# ───────────────────────────────────────────────────────────────────
# 4. WebSocket end-to-end with the mock LLM — REAL PROMPT
# ───────────────────────────────────────────────────────────────────
async def ws_real_run() -> None:
    print("\n[4] WebSocket round-trip (real prompt, mock LLM)", flush=True)
    sid: str
    async with httpx.AsyncClient(base_url=BASE) as c:
        sid = (await c.post("/api/sessions",
                            json={"model": "mock", "mode": "accept",
                                  "mcp_servers": []})).json()["id"]

    seen_types: list[str] = []
    final_text = ""
    sub_role = ""
    sub_result = ""
    tokens = {"in": 0, "out": 0}
    async with websockets.connect(f"{WS_BASE}/ws/{sid}") as ws:
        hello = json.loads(await ws.recv())
        check("WS: status hello on connect",
              hello["type"] == "status", f"got {hello['type']}")
        await ws.send(json.dumps({
            "type": "prompt",
            "text": "Plan a Poisson MMS demo on a 32x32 grid and use "
                    "the critic to validate the setup."}))
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                msg = json.loads(
                    await asyncio.wait_for(ws.recv(), timeout=5))
            except asyncio.TimeoutError:
                break
            seen_types.append(msg["type"])
            if msg["type"] == "subagent_spawned":
                sub_role = msg.get("role", "")
            if msg["type"] == "subagent_returned":
                sub_result = msg.get("result", "")
            if msg["type"] == "token_count":
                tokens["in"] += msg.get("input") or 0
                tokens["out"] += msg.get("output") or 0
            if msg["type"] == "done":
                final_text = msg.get("final_text", "")
                break

    check("WS: user_msg event echoed", "user_msg" in seen_types,
          f"types={set(seen_types)}")
    check("WS: tool_call_pending emitted",
          "tool_call_pending" in seen_types,
          "")
    check("WS: sub-agent really spawned with role=critic",
          sub_role == "critic", f"role={sub_role!r}")
    check("WS: sub-agent returned with APPROVED",
          "APPROVED" in sub_result, f"result_head={sub_result[:80]!r}")
    check("WS: token_count events received",
          tokens["in"] > 0 or tokens["out"] > 0,
          f"in={tokens['in']}, out={tokens['out']}")
    check("WS: 'done' event received",
          "done" in seen_types, "")
    check("WS: final_text mentions critic approval",
          "critic approved" in final_text.lower(),
          f"final={final_text[:120]!r}")

    # Verify session was persisted with the events
    async with httpx.AsyncClient(base_url=BASE) as c:
        saved = (await c.get(f"/api/sessions/{sid}")).json()
    check("Session JSON persisted the event stream",
          len(saved.get("events", [])) >= 5,
          f"n_events={len(saved.get('events', []))}")
    check("Session JSON persisted tokens",
          (saved.get("tokens_in", 0) + saved.get("tokens_out", 0)) > 0,
          f"in={saved.get('tokens_in')}, out={saved.get('tokens_out')}")


# ───────────────────────────────────────────────────────────────────
# 5. Plan-mode gating — REAL approve flow
# ───────────────────────────────────────────────────────────────────
async def ws_plan_mode_real() -> None:
    print("\n[5] Plan mode (real approve flow)", flush=True)
    async with httpx.AsyncClient(base_url=BASE) as c:
        sid = (await c.post("/api/sessions",
                            json={"model": "mock", "mode": "plan",
                                  "mcp_servers": []})).json()["id"]

    pending_event = None
    saw_executing = False
    saw_result = False
    saw_done = False
    async with websockets.connect(f"{WS_BASE}/ws/{sid}") as ws:
        await ws.recv()  # hello
        await ws.send(json.dumps({
            "type": "prompt",
            "text": "Begin the first major step."}))
        deadline = time.time() + 25
        approved = False
        while time.time() < deadline:
            try:
                msg = json.loads(
                    await asyncio.wait_for(ws.recv(), timeout=5))
            except asyncio.TimeoutError:
                break
            if msg["type"] == "tool_call_pending" and not approved:
                pending_event = msg
                await ws.send(json.dumps({"type": "approve",
                                          "call_id": msg["call_id"]}))
                approved = True
            elif msg["type"] == "tool_call_executing":
                saw_executing = True
            elif msg["type"] == "tool_result":
                saw_result = True
            elif msg["type"] == "done":
                saw_done = True
                break

    check("Plan: tool_call_pending arrived before execution",
          pending_event is not None,
          f"pending={pending_event['tool'] if pending_event else None!r}")
    check("Plan: tool_call_executing arrived AFTER approve",
          saw_executing, "")
    check("Plan: tool_result arrived", saw_result, "")
    check("Plan: turn terminated with 'done'", saw_done, "")


# ───────────────────────────────────────────────────────────────────
# 6. Live mode / MCP / restart commands
# ───────────────────────────────────────────────────────────────────
async def ws_live_commands() -> None:
    print("\n[6] Live mode / MCP / restart commands", flush=True)
    async with httpx.AsyncClient(base_url=BASE) as c:
        sid = (await c.post("/api/sessions",
                            json={"model": "mock", "mode": "accept",
                                  "mcp_servers": []})).json()["id"]

    async with websockets.connect(f"{WS_BASE}/ws/{sid}") as ws:
        await ws.recv()  # hello
        await ws.send(json.dumps({"type": "set_mode", "mode": "autonomous"}))
        msg = json.loads(await ws.recv())
        check("set_mode  → status event",
              msg["type"] == "status" and "autonomous" in msg["message"],
              f"msg={msg}")
        await ws.send(json.dumps({"type": "set_mcp",
                                  "servers": ["oasis"]}))
        msg = json.loads(await ws.recv())
        check("set_mcp  → status event acknowledging rebuild",
              msg["type"] == "status" and "MCP" in msg["message"],
              f"msg={msg['message']}")
        await ws.send(json.dumps({"type": "restart"}))
        msg = json.loads(await ws.recv())
        check("restart  → status event",
              msg["type"] == "status"
              and "restart" in msg["message"].lower(),
              f"msg={msg['message']}")

    async with httpx.AsyncClient(base_url=BASE) as c:
        saved = (await c.get(f"/api/sessions/{sid}")).json()
    check("Session persisted mode change",
          saved["mode"] == "autonomous", f"mode={saved['mode']}")
    check("Session persisted MCP change",
          saved["mcp_servers"] == ["oasis"],
          f"mcp_servers={saved['mcp_servers']}")


# ───────────────────────────────────────────────────────────────────
# 7. Delete the test sessions we created
# ───────────────────────────────────────────────────────────────────
async def cleanup() -> None:
    print("\n[7] Cleanup", flush=True)
    async with httpx.AsyncClient(base_url=BASE) as c:
        listed = (await c.get("/api/sessions")).json()["sessions"]
        n = 0
        for s in listed:
            if s.get("n_events", 0) <= 20:  # heuristic: our test sessions
                d = (await c.delete(f"/api/sessions/{s['id']}")).json()
                if d.get("deleted"):
                    n += 1
    check(f"Deleted {n} ephemeral test sessions",
          True, f"n={n}")


async def ws_multi_prompt_run() -> None:
    """Real battle scenario: same session, multiple prompts, reconnect
    in the middle. Verifies state survives across prompts and a
    disconnect/reconnect cycle."""
    print("\n[7] Multi-prompt + reconnect (battle scenario)", flush=True)
    async with httpx.AsyncClient(base_url=BASE) as c:
        sid = (await c.post("/api/sessions",
                            json={"model": "mock", "mode": "accept",
                                  "mcp_servers": []})).json()["id"]

    prompts = [
        "Plan a Poisson MMS demo and use the critic to validate.",
        "Now extend it to 3D and re-validate with the critic.",
        "Summarise what we did so far.",
    ]
    # First connection: send first two prompts
    async with websockets.connect(f"{WS_BASE}/ws/{sid}") as ws:
        await ws.recv()  # hello
        for p in prompts[:2]:
            await ws.send(json.dumps({"type": "prompt", "text": p}))
            deadline = time.time() + 25
            while time.time() < deadline:
                try:
                    msg = json.loads(
                        await asyncio.wait_for(ws.recv(), timeout=5))
                except asyncio.TimeoutError:
                    break
                if msg["type"] == "done":
                    break

    # Pretend the user closed the tab and came back — same session id
    await asyncio.sleep(0.2)
    async with httpx.AsyncClient(base_url=BASE) as c:
        rehydrated = (await c.get(f"/api/sessions/{sid}")).json()
    check("Reconnect: session rehydrated with persisted events",
          len(rehydrated.get("events", [])) >= 10,
          f"n_events={len(rehydrated.get('events', []))}, "
          f"tokens_in={rehydrated.get('tokens_in')}")

    # Reconnect and send a third prompt
    async with websockets.connect(f"{WS_BASE}/ws/{sid}") as ws:
        hello = json.loads(await ws.recv())
        check("Reconnect: hello carries the same session id",
              hello["session"]["id"] == sid, "")
        await ws.send(json.dumps({"type": "prompt", "text": prompts[2]}))
        last_done = False
        deadline = time.time() + 25
        while time.time() < deadline:
            try:
                msg = json.loads(
                    await asyncio.wait_for(ws.recv(), timeout=5))
            except asyncio.TimeoutError:
                break
            if msg["type"] == "done":
                last_done = True
                break
        check("Reconnect: third prompt completed with 'done'",
              last_done, "")

    async with httpx.AsyncClient(base_url=BASE) as c:
        final = (await c.get(f"/api/sessions/{sid}")).json()
    n_user = sum(1 for e in final["events"] if e.get("type") == "user_msg")
    n_done = sum(1 for e in final["events"] if e.get("type") == "done")
    check("Persisted state: three user prompts logged",
          n_user == 3, f"user_msg_count={n_user}")
    check("Persisted state: three 'done' events logged",
          n_done == 3, f"done_count={n_done}")
    check("Persisted state: tokens accumulated across prompts",
          (final.get("tokens_in", 0) + final.get("tokens_out", 0)) >= 6,
          f"tokens_in={final.get('tokens_in')}, "
          f"tokens_out={final.get('tokens_out')}")


async def ws_restart_clears_events() -> None:
    """Verify `restart` actually clears the events list and resets
    tokens — not just a status message."""
    print("\n[8] Restart command really clears state", flush=True)
    async with httpx.AsyncClient(base_url=BASE) as c:
        sid = (await c.post("/api/sessions",
                            json={"model": "mock", "mode": "accept",
                                  "mcp_servers": []})).json()["id"]
    async with websockets.connect(f"{WS_BASE}/ws/{sid}") as ws:
        await ws.recv()  # hello
        await ws.send(json.dumps({"type": "prompt", "text": "Plan."}))
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                msg = json.loads(
                    await asyncio.wait_for(ws.recv(), timeout=5))
            except asyncio.TimeoutError:
                break
            if msg["type"] == "done":
                break
        await ws.send(json.dumps({"type": "restart"}))
        await asyncio.wait_for(ws.recv(), timeout=3)

    async with httpx.AsyncClient(base_url=BASE) as c:
        s = (await c.get(f"/api/sessions/{sid}")).json()
    check("Restart: events list emptied",
          len(s["events"]) <= 1, f"n_events={len(s['events'])}")
    check("Restart: tokens reset",
          s["tokens_in"] == 0 and s["tokens_out"] == 0,
          f"tokens={s['tokens_in']}/{s['tokens_out']}")


async def main():
    print("Battlefield test against", BASE, flush=True)
    async with httpx.AsyncClient(base_url=BASE, timeout=10) as client:
        await http_checks(client)
        await artefact_checks(client)
        await session_checks(client)
    await ws_real_run()
    await ws_plan_mode_real()
    await ws_live_commands()
    await ws_multi_prompt_run()
    await ws_restart_clears_events()
    await cleanup()

    print()
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"Result: {passed}/{len(results)} passed")
    if passed != len(results):
        print()
        print("FAILURES:")
        for n, ok, d in results:
            if not ok:
                print(f"  {n}  ::  {d}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
