"""Static configuration for the WebUI.

Only constants live here. Anything you'd actually edit at runtime (active
model, mode, MCP toggles) lives in a Session object on the server side.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SANDBOX_ROOT = REPO / "eval_interactive"
SESSION_DIR = REPO / "data" / "webui_sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Each entry: id used in the API → human label, vLLM endpoint, on-disk weights.
MODELS = {
    "qwen2.5-7b":  {"label": "Qwen 2.5 7B",  "port": 8000,
                    "weights": "/media/alexander/PortableSSD/AstroNet/models/qwen2.5-7b"},
    "qwen2.5-14b": {"label": "Qwen 2.5 14B", "port": 8001,
                    "weights": "/media/alexander/PortableSSD/AstroNet/models/qwen2.5-14b"},
    "qwen2.5-32b": {"label": "Qwen 2.5 32B", "port": 8002,
                    "weights": "/media/alexander/PortableSSD/AstroNet/models/qwen2.5-32b"},
    "mock":        {"label": "Mock LLM (no GPU)", "port": None,
                    "weights": None},
}

# MCP servers selectable in the UI. The OASiS server is the main one;
# additional rows are placeholders for future plug-ins.
MCP_SERVERS = {
    "oasis": {
        "label": "OASiS — Open Agentic Simulation System",
        "command": str(REPO / ".venv/bin/python"),
        "args": ["-m", "server"],
        "cwd": str(REPO / "src"),
        "env_extra": {
            "PYTHONPATH": str(REPO / "src"),
            "FOURC_ROOT": os.environ.get(
                "FOURC_ROOT", str(Path.home() / "4C")),
            "FOURC_BINARY": os.environ.get(
                "FOURC_BINARY", str(Path.home() / "4C/build/4C")),
            "LD_LIBRARY_PATH": os.environ.get(
                "LD_LIBRARY_PATH", "/opt/4C-dependencies/lib"),
        },
        "default_on": True,
    },
}

MODES = ("plan", "accept", "autonomous")
DEFAULT_MODE = "accept"
DEFAULT_MODEL = "mock"
