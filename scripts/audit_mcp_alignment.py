"""Audit script — surface-of-truth vs in-process MCP catalog alignment.

Catches a class of operational misalignment found during the
2026-06-01 rigorous-testing campaign:

The MCP server (running as a child process of Claude Code or any
long-lived host) imports each backend's KNOWLEDGE dict at startup
and never reloads it. Catalog edits in
src/backends/<be>/generators/<physics>.py are committed but the
already-running MCP keeps serving the pre-edit pitfalls.

This script does NOT call the MCP — it cannot, because the MCP is
running in another process. What it DOES is print the catalog
exactly as a fresh Python process sees it, so a human (or a CI
job) can diff that against what the user-facing MCP returns.

Usage::

    .venv/bin/python scripts/audit_mcp_alignment.py [backend [physics]]

If both args given, dumps just that (backend, physics) entry. If
backend only, dumps every physics for that backend. With no args,
dumps every (backend, physics) entry for every registered backend.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "data"))


def main(argv: list[str]) -> int:
    os.chdir(REPO_ROOT)
    from core.registry import get_backend, load_all_backends

    load_all_backends()
    # _backends is module-level dict but private; iterate via
    # the explicit list-known-backends approach.
    known = ["fenics", "fourc", "dealii", "ngsolve", "skfem",
             "kratos", "dune", "febio"]

    want_backend = argv[1] if len(argv) > 1 else None
    want_physics = argv[2] if len(argv) > 2 else None

    out: dict = {}
    for be in known:
        if want_backend and be != want_backend:
            continue
        backend = get_backend(be)
        if backend is None:
            out[be] = {"_unavailable": True}
            continue
        be_out: dict = {}
        for cap in backend.supported_physics():
            if want_physics and cap.name != want_physics:
                continue
            k = backend.get_knowledge(cap.name) or {}
            pitfalls = k.get("pitfalls", []) or []
            be_out[cap.name] = {
                "pitfall_count": len(pitfalls),
                "first_pitfall_prefix": (
                    pitfalls[0][:80] if pitfalls else None),
            }
        out[be] = be_out

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
