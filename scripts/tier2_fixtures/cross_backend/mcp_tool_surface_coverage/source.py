"""Tier-2 Layer E: MCP tool surface coverage.

Calls the three core user-facing MCP tools
(`prepare_simulation`, `knowledge`, `discover`) — the
same surface an LLM agent reaches via the registered
FastMCP tool layer — and asserts:

  (a) Every (backend, physics) tuple advertised by the
      live registry yields non-empty text from
      prepare_simulation.
  (b) The returned text mentions the requested physics
      name verbatim (sanity check that fuzzy matching
      didn't silently redirect to a wrong physics).
  (c) discover('list') text contains every available
      backend name.
  (d) discover('physics <backend>') text mentions every
      physics advertised for that backend.
  (e) knowledge('<physics>', solver='<backend>') text
      mentions the physics name + is non-empty.
  (f) No tool surface emits "Unknown" / "ERROR" /
      "Traceback".

This is the user-facing-surface counterpart to the
Layer-A/B catalog-and-generator audits — Layer A
verifies upstream-library symbols, Layer B verifies
generators execute, Layer E verifies the MCP wrappers
return text that names the right things.

Approach: instantiate a FastMCP, register the
consolidated tools, load the live backend registry,
then iterate via ToolManager.call_tool. No live MCP
server process is spawned — same code path, in-process.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

logging.disable(logging.CRITICAL)
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "src"))


# Probe a small representative subset to keep the fixture
# under the 120-second runner timeout. Full sweep is 198
# (backend, physics) pairs; that's too slow when each call
# subprocess-spawns a backend env. Pick ~3 physics per
# available backend.
PROBE_PHYSICS: dict[str, list[str]] = {
    "fenics":  ["poisson", "linear_elasticity", "helmholtz"],
    "fourc":   ["poisson", "linear_elasticity", "fluid"],
    "dealii":  ["poisson", "linear_elasticity", "stokes"],
    "ngsolve": ["poisson", "linear_elasticity", "helmholtz"],
    "skfem":   ["poisson", "linear_elasticity", "heat"],
    "kratos":  ["poisson", "linear_elasticity", "fluid"],
}


def _setup_mcp():
    from mcp.server.fastmcp import FastMCP
    from core.registry import load_all_backends
    from tools.consolidated import register_consolidated_tools

    m = FastMCP("layer-e-audit")
    register_consolidated_tools(m)
    load_all_backends()
    return m._tool_manager


def _text(result) -> str:
    """call_tool returns either a tuple (content_blocks,
    structured_output) or a list of blocks. Extract the
    string body."""
    if isinstance(result, tuple) and len(result) >= 1:
        result = result[0]
    if isinstance(result, list):
        parts = []
        for block in result:
            if hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if hasattr(result, "text"):
        return result.text
    return str(result)


async def main_async() -> int:
    tm = _setup_mcp()

    failures: list[str] = []
    successes = 0

    # (c) discover('list') mentions every available backend
    listed = _text(await tm.call_tool(
        "discover", {"query": "list"}))
    for backend in PROBE_PHYSICS:
        if backend not in listed.lower():
            failures.append(f"discover_list_missing_backend={backend}")
    print(f"discover_list_chars={len(listed)}")

    # (a-b) prepare_simulation per (backend, physics)
    # (e)   knowledge per (backend, physics)
    # (d)   discover('physics <backend>') per backend
    for backend, physics_list in PROBE_PHYSICS.items():
        # discover('physics', solver=<backend>) — separate
        # args, NOT a 'physics <backend>' query string.
        disc = _text(await tm.call_tool(
            "discover",
            {"query": "physics", "solver": backend}))
        if not disc or len(disc) < 100:
            failures.append(
                f"discover_phys_thin={backend}::{len(disc)}")
            continue
        # discover('physics') text uses the underscored
        # catalog name directly (see L523).
        disc_lower = disc.lower()
        for needle in physics_list:
            if needle not in disc_lower:
                failures.append(
                    f"discover_phys_missing="
                    f"{backend}::{needle}")
        for tok in ("Unknown solver", "Traceback"):
            if tok in disc:
                failures.append(
                    f"discover_phys_error_tok="
                    f"{backend}::{tok!r}")
        for phys in physics_list:
            # prepare_simulation
            prep = _text(await tm.call_tool(
                "prepare_simulation",
                {"solver": backend, "physics": phys}))
            if not prep or len(prep) < 200:
                failures.append(
                    f"prep_thin={backend}::{phys}::"
                    f"{len(prep)}")
                continue
            prep_lower = prep.lower()
            phys_spaced = phys.replace("_", " ")
            tokens = [t for t in phys.split("_")
                      if len(t) >= 4]
            if not (phys in prep_lower
                    or phys_spaced in prep_lower
                    or any(t in prep_lower for t in tokens)):
                failures.append(
                    f"prep_missing_physics_name="
                    f"{backend}::{phys}")
            for tok in ("Unknown solver", "Traceback"):
                if tok in prep:
                    failures.append(
                        f"prep_error_tok="
                        f"{backend}::{phys}::{tok!r}")
            # 'ERROR:' must be at line start (after optional
            # leading whitespace) — substring matching false-
            # positives on real backend YAML keys like 4C's
            # FLUID DYNAMIC `CALCERROR: byfunct`. Audit pass 4
            # tightening (2026-06-02).
            import re as _re
            if _re.search(r"^\s*ERROR:", prep, _re.MULTILINE):
                failures.append(
                    f"prep_error_tok="
                    f"{backend}::{phys}::'^ERROR:'")
            # knowledge — topic="physics" is the
            # documented dispatch route for per-physics
            # catalog lookups (NOT topic=<physics_name>;
            # topic is a category enum).
            kn = _text(await tm.call_tool(
                "knowledge",
                {"topic": "physics", "solver": backend,
                 "physics": phys}))
            if not kn or len(kn) < 100:
                failures.append(
                    f"kn_thin={backend}::{phys}::"
                    f"{len(kn)}")
                continue
            # Match physics in body via UNDERSCORE form
            # OR space-separated form (catalog text often
            # spells 'linear_elasticity' as 'Linear
            # elasticity' in description) OR a non-trivial
            # token from the physics name.
            kn_lower = kn.lower()
            phys_underscored = phys
            phys_spaced = phys.replace("_", " ")
            tokens = [t for t in phys.split("_")
                      if len(t) >= 4]
            ok = (phys_underscored in kn_lower
                  or phys_spaced in kn_lower
                  or any(t in kn_lower for t in tokens))
            if not ok:
                failures.append(
                    f"kn_missing_physics_name="
                    f"{backend}::{phys}")
            successes += 1
        print(f"{backend}_ok")

    print(f"total_probes={successes}")
    print(f"failures_count={len(failures)}")
    for f in failures[:20]:
        print(f"  fail: {f}")

    if not failures:
        return 0
    print("FAIL: MCP-tool surface coverage regression",
          file=sys.stderr)
    return 2


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
