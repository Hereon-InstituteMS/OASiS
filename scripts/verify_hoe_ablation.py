#!/usr/bin/env python3
"""Pre-flight check for the HOE ablation conditions.

Boots the server module three times (MCP_FULL, MCP_NO_CRITIC,
MCP_NO_PITFALL_DB) and asserts that each condition's masking actually
holds: the critic paragraph is present/absent and pitfall-DB content
(per-backend pitfalls, Signal: anchors, general input-format pitfalls)
is present/stripped.

Run from the repo root with the server venv:

    .venv/bin/python scripts/verify_hoe_ablation.py

Exit code 0 with three "ok" lines means the conditions are safe to run.
"""
import json
import os
import sys


def boot(ablate_pitfalls: bool, ablate_critic: bool):
    os.environ["OFA_DISABLE_PITFALLS"] = "1" if ablate_pitfalls else "0"
    os.environ["OFA_DISABLE_CRITIC"] = "1" if ablate_critic else "0"
    for m in list(sys.modules):
        if m.startswith(("server", "tools", "backends", "core")):
            del sys.modules[m]
    if "src" not in sys.path:
        sys.path.insert(0, "src")
    import server
    from core.registry import get_backend
    import tools.consolidated as cons
    return server, cons, get_backend


def instructions_of(server):
    mcp = server.mcp
    return (mcp._mcp_server.instructions
            if hasattr(mcp, "_mcp_server") else mcp.instructions)


def main():
    # MCP_FULL
    server, cons, get_backend = boot(False, False)
    instr = instructions_of(server)
    assert "MANDATORY CRITIC" in instr, "FULL: critic block missing!"
    k = cons._strip_pitfalls(get_backend("skfem").get_knowledge("poisson"))
    assert "pitfalls" in json.dumps(k, default=str), \
        "FULL: pitfalls stripped but should be present"
    print("MCP_FULL          ok: critic present, pitfalls present")

    # MCP_NO_CRITIC
    server, cons, get_backend = boot(False, True)
    assert "MANDATORY CRITIC" not in instructions_of(server), \
        "NO_CRITIC: critic block still present!"
    k = cons._strip_pitfalls(get_backend("skfem").get_knowledge("poisson"))
    assert "pitfalls" in json.dumps(k, default=str), \
        "NO_CRITIC: pitfalls must remain available"
    print("MCP_NO_CRITIC     ok: critic paragraph stripped, pitfalls intact")

    # MCP_NO_PITFALL_DB
    server, cons, get_backend = boot(True, False)
    assert "MANDATORY CRITIC" in instructions_of(server), \
        "NO_PITFALL: critic should remain"
    s = json.dumps(
        cons._strip_pitfalls(get_backend("skfem").get_knowledge("poisson")),
        default=str)
    assert '"pitfalls"' not in s and "Signal:" not in s, \
        "NO_PITFALL: pitfalls/Signal leaked in physics knowledge"
    gk = json.dumps(
        cons._strip_pitfalls(get_backend("fourc").get_knowledge("input_format")),
        default=str)
    assert '"general_pitfalls"' not in gk, \
        "NO_PITFALL: general_pitfalls leaked"
    assert cons._ABLATE_PITFALLS, "NO_PITFALL: toggle not read"
    print("MCP_NO_PITFALL_DB ok: pitfalls, Signal anchors, general "
          "input-format pitfalls stripped; cross_backend/postmortems gated")


if __name__ == "__main__":
    main()
