"""Regression tests for the MCP `instructions` string (issue #45).

A client that truncates the server's instructions when folding them into the
model's context dropped the mid-string MANDATORY CRITIC safety block. It now
leads the string; these tests guard that it stays there, appears once, respects
the ablation toggle, and only references tools that actually exist.
"""
import functools
import os
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
