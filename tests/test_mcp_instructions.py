"""Regression tests for the MCP `instructions` string (issue #45).

A client that truncates the server's instructions when folding them into the
model's context dropped the mid-string MANDATORY CRITIC safety block. It now
leads the string; these tests guard that it stays there, appears once, respects
the ablation toggle, and only references tools that actually exist.
"""
import functools
import os
import pathlib
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "src")


@functools.lru_cache(maxsize=None)
def _instructions(disable_critic: bool = False) -> str:
    """Build the server's instructions in a subprocess so env toggles read at
    import time (OFA_DISABLE_CRITIC) take effect. Cached so we import at most twice."""
    env = dict(os.environ, PYTHONPATH=SRC)
    env.pop("OFA_DISABLE_CRITIC", None)
    if disable_critic:
        env["OFA_DISABLE_CRITIC"] = "1"
    code = "import sys; import server; sys.stdout.write(server.mcp.instructions)"
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout


_BLOCK = "For every major step, spawn a sub-agent"  # unique to _CRITIC_BLOCK


def test_critic_block_leads_instructions():
    instr = _instructions()
    assert "MANDATORY CRITIC" in instr
    # Must come before the (long) backends section so prefix truncation keeps it.
    assert instr.index("MANDATORY CRITIC") < instr.index("## Available Backends")
    assert instr.index(_BLOCK) < instr.index("## Available Backends")


def test_critic_block_appears_exactly_once():
    assert _instructions().count(_BLOCK) == 1


def test_ablation_removes_critic_block():
    instr = _instructions(disable_critic=True)
    assert instr.count(_BLOCK) == 0
    assert "MANDATORY CRITIC" not in instr


def test_critic_block_references_only_real_tools():
    instr = _instructions()
    # parameter_study is not a registered tool — must not be referenced.
    assert "parameter_study" not in instr
    # The recommended coupling tools (which now carry critic_approved) are named.
    assert "couple" in instr and "couple_precice" in instr


def test_all_sections_present():
    instr = _instructions()
    for sec in ["## Available Backends", "## Workflow", "## Key Principles",
                "## Solver Selection", "## Cross-Solver Coupling",
                "## Developer Mode", "## Session Knowledge"]:
        assert sec in instr, f"missing section {sec}"


# ── Schema-level guards (issue #45 second audit) ─────────────────────────
# The instructions blob is truncatable; the tool SCHEMAS are always delivered.
# These assert that every tool the critic block names is really registered and
# really carries critic_approved, so the fix can't silently regress with the
# text-only tests above still green.

_CRITIC_TOOLS = ["run_simulation", "run_with_generator", "coupled_solve",
                 "couple", "couple_precice"]


def _registered_tools() -> dict:
    """{name: Tool} for the live MCP registry, resolved in a subprocess so the
    heavy backend imports don't run in the test process."""
    code = (
        "import asyncio, json, sys; import server;\n"
        "tools = asyncio.new_event_loop().run_until_complete(server.mcp.list_tools());\n"
        "sys.stdout.write(json.dumps({t.name: t.inputSchema for t in tools}))"
    )
    env = dict(os.environ, PYTHONPATH=SRC)
    env.pop("OFA_DISABLE_CRITIC", None)
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True, timeout=240)
    assert r.returncode == 0, r.stderr[-2000:]
    import json
    return json.loads(r.stdout)


def test_every_critic_block_tool_is_registered_with_critic_approved():
    schemas = _registered_tools()
    assert "parameter_study" not in schemas  # the phantom tool must be gone
    for name in _CRITIC_TOOLS:
        assert name in schemas, f"critic block names unregistered tool {name}"
        props = schemas[name].get("properties", {})
        assert "critic_approved" in props, f"{name} lost critic_approved"
        assert props["critic_approved"].get("type") == "boolean"


def test_coupling_tools_expose_critic_approved():
    # The recommended coupling tools carried no critic surface before the fix.
    schemas = _registered_tools()
    for name in ("couple", "couple_precice"):
        assert "critic_approved" in schemas[name].get("properties", {})
def test_every_critic_gated_tool_actually_READS_critic_approved():
    """A declared parameter that is never read is worse than no parameter.

    An audit found `coupled_solve` accepting `critic_approved` and never
    referencing it in its body, so an unreviewed run was indistinguishable
    from a reviewed one — while the test above passed, because it only checked
    that the parameter EXISTS. Declaring the gate is not enforcing it.
    """
    import ast
    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "tools" / "consolidated.py"
    tree = ast.parse(src.read_text())
    bodies = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = [a.arg for a in (*node.args.args, *node.args.kwonlyargs)]
            if "critic_approved" not in params:
                continue
            reads = sum(
                1 for n in ast.walk(node)
                if isinstance(n, ast.Name) and n.id == "critic_approved"
                and isinstance(n.ctx, ast.Load))
            bodies[node.name] = reads
    dead = [name for name, reads in bodies.items() if reads == 0]
    assert not dead, (
        "these tools declare critic_approved and never read it, so the "
        "critic requirement is unenforced for them: " + ", ".join(dead))
