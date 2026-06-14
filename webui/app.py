"""FastAPI app for the OASiS WebUI.

Start with::

    .venv-lg/bin/uvicorn webui.app:app --reload --port 8080

Endpoints:

* ``GET  /`` — single-page UI (HTML)
* ``GET  /api/models`` / ``/api/mcp_servers`` / ``/api/modes``
* ``GET  /api/files?rel=…`` — sandbox directory listing
* ``GET  /api/file?rel=…`` — file text content
* ``GET  /api/viz?rel=…`` — visualization payload
* ``GET  /sandbox-file/{path:path}`` — raw file bytes for vtk.js etc.
* ``GET  /api/sessions`` / ``POST /api/sessions`` / ``DELETE``
* ``GET  /api/extract_params?rel=…`` — heuristic slider extraction
* ``WS   /ws/{session_id}`` — streamed run channel

The WebSocket protocol is symmetric JSON:
``{"type": "...", ...}`` either direction. See :mod:`webui.runner` for
outbound event types and :func:`_handle_inbound` for the inbound set.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config, files, sessions, viz
from .runner import (ApprovalGate, _session_workdir,
                     build_agent_for_session, stream_turn)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("oasis.webui")

app = FastAPI(title="OASiS WebUI", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

STATIC = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC / "index.html").read_text()


# ───────────────────────────────────────────────────────────────────
# Config endpoints
# ───────────────────────────────────────────────────────────────────
@app.get("/api/models")
async def get_models():
    out = []
    for k, m in config.MODELS.items():
        out.append({"id": k, "label": m["label"], "port": m["port"]})
    return {"models": out, "default": config.DEFAULT_MODEL}


@app.get("/api/mcp_servers")
async def get_mcp():
    return {"servers": [
        {"id": k, "label": v["label"], "default_on": v["default_on"]}
        for k, v in config.MCP_SERVERS.items()
    ]}


@app.get("/api/modes")
async def get_modes():
    return {"modes": list(config.MODES), "default": config.DEFAULT_MODE}


# ───────────────────────────────────────────────────────────────────
# Files & viz
# ───────────────────────────────────────────────────────────────────
@app.get("/api/files")
async def api_files(rel: str = ""):
    try:
        return files.list_dir(rel)
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.get("/api/file")
async def api_file(rel: str):
    try:
        return files.read_text(rel)
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.get("/api/viz")
async def api_viz(rel: str):
    try:
        return viz.visualize(rel)
    except PermissionError as e:
        raise HTTPException(403, str(e))


@app.get("/api/extract_params")
async def api_extract_params(rel: str):
    info = files.read_text(rel)
    if "text" not in info:
        return {"params": []}
    return {"params": viz.extract_params(info["text"])}


@app.get("/sandbox-file/{rel:path}")
async def sandbox_file(rel: str):
    try:
        p = files._safe(rel)
    except PermissionError as e:
        raise HTTPException(403, str(e))
    if not p.is_file():
        raise HTTPException(404, "not a file")
    return FileResponse(p)


# ───────────────────────────────────────────────────────────────────
# Sessions
# ───────────────────────────────────────────────────────────────────
@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": sessions.list_sessions()}


@app.post("/api/sessions")
async def new_session(body: dict | None = None):
    body = body or {}
    s = sessions.new_session(
        model=body.get("model"),
        mode=body.get("mode"),
        mcp_servers=body.get("mcp_servers"))
    sessions.save(s)
    return s


@app.get("/api/sessions/{sid}")
async def get_session(sid: str):
    try:
        return sessions.load(sid)
    except FileNotFoundError:
        raise HTTPException(404, "no such session")


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    return {"deleted": sessions.delete(sid)}


# ───────────────────────────────────────────────────────────────────
# WebSocket — interactive runs
# ───────────────────────────────────────────────────────────────────
class WSSession:
    def __init__(self, ws: WebSocket, sid: str):
        self.ws = ws
        self.sid = sid
        self.state = sessions.load(sid)
        self.gate = ApprovalGate()
        self.agent = None
        self.workdir = _session_workdir(sid)

    def mode(self) -> str:
        return self.state.get("mode", config.DEFAULT_MODE)

    async def emit(self, event: dict):
        self.state["events"].append(event)
        if event.get("type") == "token_count":
            self.state["tokens_in"] = (self.state.get("tokens_in", 0)
                                       + (event.get("input") or 0))
            self.state["tokens_out"] = (self.state.get("tokens_out", 0)
                                        + (event.get("output") or 0))
        try:
            await self.ws.send_text(json.dumps(event, default=str))
        except Exception:
            pass

    def ensure_agent(self):
        if self.agent is None:
            self.agent = build_agent_for_session(
                model=self.state.get("model", config.DEFAULT_MODEL),
                mcp_on="oasis" in self.state.get("mcp_servers", []),
                workdir=self.workdir,
                emitter=self.emit,
                get_mode=self.mode,
                gate=self.gate,
            )
        return self.agent


@app.websocket("/ws/{sid}")
async def ws_endpoint(ws: WebSocket, sid: str):
    await ws.accept()
    try:
        ses = WSSession(ws, sid)
    except FileNotFoundError:
        await ws.send_text(json.dumps({"type": "error",
                                       "message": f"no such session: {sid}"}))
        await ws.close()
        return
    try:
        await ses.emit({"type": "status", "message": "connected",
                        "session": ses.state})
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ses.emit({"type": "error", "message": "bad json"})
                continue
            await _handle_inbound(ses, msg)
    except WebSocketDisconnect:
        log.info("client disconnected: %s", sid)
    except Exception as e:
        log.exception("ws error")
        try:
            await ws.send_text(json.dumps({"type": "error",
                                           "message": str(e)}))
        except Exception:
            pass
    finally:
        try:
            sessions.save(ses.state)
        except Exception:
            pass


async def _handle_inbound(ses: WSSession, msg: dict):
    t = msg.get("type")
    if t == "prompt":
        text = msg.get("text", "").strip()
        if not text:
            return
        await ses.emit({"type": "user_msg", "text": text})
        ses.ensure_agent()
        try:
            await stream_turn(agent=ses.agent, user_text=text,
                              emitter=ses.emit)
        except Exception:
            pass
        sessions.save(ses.state)
    elif t == "approve":
        ses.gate.resolve(msg["call_id"], True)
    elif t == "reject":
        ses.gate.resolve(msg["call_id"], False, msg.get("reason", ""))
    elif t == "set_mode":
        ses.state["mode"] = msg.get("mode", ses.state["mode"])
        sessions.save(ses.state)
        await ses.emit({"type": "status",
                        "message": f"mode → {ses.state['mode']}"})
    elif t == "set_model":
        ses.state["model"] = msg["model"]
        ses.agent = None
        sessions.save(ses.state)
        await ses.emit({"type": "status",
                        "message": f"model → {ses.state['model']}"})
    elif t == "set_mcp":
        ses.state["mcp_servers"] = msg.get("servers", [])
        ses.agent = None
        sessions.save(ses.state)
        await ses.emit({"type": "status",
                        "message": "MCP servers updated; agent will "
                        "rebuild on next prompt"})
    elif t == "restart":
        ses.state["events"] = []
        ses.state["tokens_in"] = 0
        ses.state["tokens_out"] = 0
        ses.agent = None
        sessions.save(ses.state)
        await ses.emit({"type": "status", "message": "session restarted"})
    else:
        await ses.emit({"type": "error",
                        "message": f"unknown inbound type: {t}"})
