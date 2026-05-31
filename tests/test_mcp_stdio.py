"""
End-to-end MCP-protocol verification.

Every other test in this repo exercises the backend Python API directly
(`b.generate_input(...)`, `b.run(...)`).  That covers the implementation
the MCP server wraps, but **not** the JSON-RPC stdio layer the server
actually exposes to external agents.

This file is the missing third layer: it spawns the real MCP server as
a subprocess and drives it through `mcp.client.stdio.stdio_client`,
verifying that the tools exposed over stdio respond with well-formed
results — i.e. it would catch a regression where the server config
breaks, a tool decorator drops a parameter, or the protocol envelope
is mis-formatted, none of which the direct-API tests would notice.

The verification is intentionally minimal and behavioural:
  - `list_tools` returns the expected set of tool names
  - `discover()` returns *something* non-empty
  - `prepare_simulation('skfem', 'poisson')` returns a payload that
    contains a `template` and a `knowledge` section
  - one full `run_simulation` round-trip for skfem-Poisson returns a
    completed job with a non-empty `output_files` listing

Anything beyond those four checks belongs in a per-tool test rather
than a smoke probe.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _server_params():
    """Build the same `StdioServerParameters` the user's `~/.claude.json`
    uses to launch this server, so the test exercises the real launch
    path agents would.
    """
    from mcp import StdioServerParameters

    # `**os.environ` already brings every parent-process variable
    # (including FENICS_PYTHON / DEALII_ROOT / DEAL_II_DIR /
    # FOURC_ROOT / FOURC_BINARY when the user has set them), so no
    # separate per-backend propagation loop is needed on top of it.
    # We then layer the test-specific PYTHONPATH and PYVISTA_OFF_SCREEN
    # overrides on top so the spawned server runs without a display.
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "PYVISTA_OFF_SCREEN": "true",
    }
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "server"],
        env=env,
        cwd=str(REPO_ROOT / "src"),
    )


async def _list_tools_async() -> list[str]:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _server_params()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.list_tools()
            return [t.name for t in resp.tools]


async def _call_tool_async(name: str, arguments: dict) -> dict:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _server_params()
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await session.call_tool(name, arguments=arguments)
            # call_tool returns a CallToolResult with `content` list of
            # TextContent/ImageContent; flatten to text+isError.
            text = "\n".join(
                c.text for c in resp.content if hasattr(c, "text")
            )
            return {"isError": bool(resp.isError), "text": text}


# ── tests ────────────────────────────────────────────────────────────────


def test_list_tools_includes_canonical_set():
    """The MCP server must expose at least the core toolset every
    agent uses.  A missing name here means a tool registration was
    silently dropped on the server side.
    """
    try:
        tools = asyncio.run(_list_tools_async())
    except ModuleNotFoundError as e:
        pytest.skip(f"mcp client SDK not importable: {e}")

    # Canonical names from the user's documented workflow + the names
    # in src/server.py register_*_tools() calls.  Not exhaustive (more
    # tools exist) — listed names are the contractually-stable subset
    # external agents rely on.
    expected = {
        "prepare_simulation",
        "run_simulation",
        "discover",
        "knowledge",
        "examples",
        "developer",
        "session_insights",
    }
    missing = expected - set(tools)
    assert not missing, (
        f"MCP server is missing expected tools: {sorted(missing)}; "
        f"server exposed: {sorted(tools)}"
    )


def test_discover_returns_non_empty():
    """discover() with an empty query should return *something* —
    typically the list of available backends or capabilities.
    Failing this almost certainly means the backend registry failed
    to load inside the spawned server."""
    try:
        r = asyncio.run(_call_tool_async("discover", {"query": ""}))
    except ModuleNotFoundError as e:
        pytest.skip(f"mcp client SDK not importable: {e}")

    assert not r["isError"], f"discover() raised: {r['text'][:300]}"
    assert r["text"].strip(), "discover() returned empty text"


def test_prepare_simulation_returns_template_and_knowledge():
    """prepare_simulation for the simplest cell that is known to work
    on every machine (skfem Poisson) must return a payload that
    contains the words `template` and `knowledge` — those are the
    two sections the server's docstring promises.
    """
    try:
        r = asyncio.run(_call_tool_async(
            "prepare_simulation",
            {"solver": "skfem", "physics": "poisson"},
        ))
    except ModuleNotFoundError as e:
        pytest.skip(f"mcp client SDK not importable: {e}")

    assert not r["isError"], f"prepare_simulation raised: {r['text'][:300]}"
    text = r["text"].lower()
    assert "template" in text, (
        f"prepare_simulation did not return a `template` section; "
        f"got: {r['text'][:400]}"
    )
    assert "knowledge" in text, (
        f"prepare_simulation did not return a `knowledge` section; "
        f"got: {r['text'][:400]}"
    )


def test_run_simulation_skfem_poisson_round_trip(tmp_path):
    """Drive a complete simulation through the MCP-stdio path.

    Uses skfem-Poisson because it is the cheapest fully-working cell
    in the layer-3 probe (run time ~0.4 s, no external compiler).
    The minimal script below is the canonical scikit-fem Poisson
    on the unit square -∇²u=1 with u=0 on the boundary; if the
    server's `run_simulation` correctly walks the validate → run
    → post-process path, the returned text must contain `completed`
    and reference a `.vtu` artefact.
    """
    script = (
        '"""Poisson skfem smoke."""\n'
        "from skfem import MeshQuad, ElementQuad1, Basis, solve, condense\n"
        "from skfem.models.poisson import laplace, unit_load\n"
        "import meshio, numpy as np\n"
        "m = MeshQuad.init_tensor(np.linspace(0,1,17), np.linspace(0,1,17))\n"
        "ib = Basis(m, ElementQuad1())\n"
        "K = laplace.assemble(ib); f = unit_load.assemble(ib)\n"
        "D = ib.get_dofs().flatten()\n"
        "u = solve(*condense(K, f, D=D))\n"
        "meshio.Mesh(np.column_stack([m.p.T, np.zeros(m.p.shape[1])]),\n"
        "            [('quad', m.t.T)], point_data={'phi': u}).write('result.vtu')\n"
    )
    try:
        r = asyncio.run(_call_tool_async(
            "run_simulation",
            {
                "solver": "skfem",
                "input_content": script,
                "job_name": "mcp_stdio_smoke_skfem_poisson",
                # critic_approved=True bypasses the in-tool reminder
                # that an independent critic should pre-approve real
                # simulations — fine here because the script is a
                # fixed smoke probe, not a research run.
                "critic_approved": True,
            },
        ))
    except ModuleNotFoundError as e:
        pytest.skip(f"mcp client SDK not importable: {e}")

    assert not r["isError"], f"run_simulation raised: {r['text'][:300]}"
    text = r["text"].lower()
    assert "completed" in text, (
        f"run_simulation did not report `completed` in its response; "
        f"got: {r['text'][:400]}"
    )
    assert ".vtu" in text, (
        f"run_simulation did not reference any .vtu artefact; "
        f"got: {r['text'][:400]}"
    )
