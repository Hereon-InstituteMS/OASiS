"""Minimal in-process OpenAI-compatible server for offline scaffold tests.

Purpose: drive create_react_agent end-to-end without vLLM/Qwen, so we can
verify the parent → spawn_subagent → sub-agent → return chain actually
fires through every wiring layer.

Behaviour is deterministic and routes by system-prompt fingerprint:

* If the system message contains "ruthlessly critical reviewer", the
  request is from a CRITIC sub-agent — reply with a final assistant
  message "APPROVED: <reason>".
* Otherwise it is the PARENT. On the first call (no tool message in
  history yet), reply with a tool_call to ``spawn_subagent`` asking the
  critic to approve. On the second call (history now contains the
  critic's verdict), reply with a final assistant message.

Implementation uses only the stdlib so no extra deps land in .venv-lg.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


_LOG: list[dict] = []
_LOCK = threading.Lock()


def _is_critic_system(messages: list[dict]) -> bool:
    for m in messages:
        if m.get("role") == "system" and \
           "ruthlessly critical reviewer" in (m.get("content") or ""):
            return True
    return False


def _has_tool_message(messages: list[dict]) -> bool:
    return any(m.get("role") == "tool" for m in messages)


def _build_response(messages: list[dict]) -> dict:
    is_critic = _is_critic_system(messages)
    has_tool = _has_tool_message(messages)

    if is_critic:
        content = "APPROVED: setup and units look consistent; proceed."
        msg = {"role": "assistant", "content": content}
    elif not has_tool:
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": "spawn_subagent",
                    "arguments": json.dumps({
                        "role": "critic",
                        "task": "Verify the setup is consistent before proceeding.",
                        "context": "Parent considering a Poisson MMS run on a 32x32 grid."
                    }),
                },
            }],
        }
    else:
        msg = {"role": "assistant",
               "content": "Parent: critic approved; recording result."}

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "mock-qwen",
        "choices": [{"index": 0, "message": msg, "finish_reason":
                     "tool_calls" if msg.get("tool_calls") else "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"{}")
        messages = body.get("messages", [])
        is_critic = _is_critic_system(messages)
        with _LOCK:
            _LOG.append({
                "path": self.path,
                "is_critic": is_critic,
                "n_messages": len(messages),
                "had_tool_message": _has_tool_message(messages),
                "n_tools_available": len(body.get("tools") or []),
            })
        resp = _build_response(messages)
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a, **kw):
        return  # silence stdout


def start(port: int = 18234) -> tuple[ThreadingHTTPServer, threading.Thread]:
    srv = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, th


def get_log() -> list[dict]:
    with _LOCK:
        return list(_LOG)


def reset_log() -> None:
    with _LOCK:
        _LOG.clear()
