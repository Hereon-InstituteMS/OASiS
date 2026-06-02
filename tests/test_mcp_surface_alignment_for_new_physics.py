"""Regression: every audit-driven physics added via the upstream-
demo audit loop must be fully visible through the MCP tool surface
(discover + knowledge + prepare_simulation + examples).

Context: when adding a new physics, six things need to line up:
  1. PhysicsCapability registered in backend.py supported_physics()
  2. Generator wired in generators/__init__.py
  3. KNOWLEDGE entry merged into the registry dict
  4. Tier-0/Tier-1 pitfall verification (real symbols, observable
     vocab) — verified by scripts/verify_signal_clauses.py
  5. Layer-F end-to-end execution gate — verified by
     scripts/tier2_fixtures/cross_backend/catalog_template_executes
  6. END-USER MCP-TOOL VISIBILITY — discover, knowledge,
     prepare_simulation, examples all return the new physics
     and its content, not "not found"

This gate covers (6). The other layers cover (1)-(5).

Concretely, an LLM client trying to use a newly-added physics
hits these four MCP tools in sequence; if any returns a
generic "no such physics" or an empty knowledge dict, the
practical UX is broken even though the in-code catalog is fine.

The test is intentionally NOT generic — it pins down the
exact set of audit-driven physics from the 2026-06-02 sweep.
When new audit-driven physics ship, EXTEND `_AUDIT_DRIVEN_PHYSICS`
in lock-step. The test failing on a new addition means the
new physics is in-code-canonical but MCP-tool-invisible: probably
a wiring miss in backend.py or the generator __init__.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


# The audit-driven physics shipped in this session, sourced
# from the upstream_demo_audit.py gap report and verified
# end-to-end at commit time. When extending: add (backend,
# physics) — and ALSO add a Layer-F row + Tier-0/1 verified
# pitfalls before this gate will let you through.
_AUDIT_DRIVEN_PHYSICS: list[tuple[str, str]] = [
    ("skfem",  "wave"),                  # ex09 / ex36 / ex44
    ("skfem",  "adaptive_poisson"),      # ex11 / ex22
    ("skfem",  "point_source"),          # ex17 / ex38
    ("skfem",  "schrodinger"),           # ex39
    ("skfem",  "contact"),               # ex04
    ("skfem",  "hydraulic_resistance"),  # ex29
    ("fenics", "matrix_free_poisson"),   # demo_poisson_matrix_free.py
]


class TestMcpSurfaceAlignmentForNewPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from mcp.server.fastmcp import FastMCP
        from tools.consolidated import register_consolidated_tools
        from core.registry import load_all_backends
        load_all_backends()
        cls.mcp = FastMCP("alignment-test")
        register_consolidated_tools(cls.mcp)
        cls.tools = cls.mcp._tool_manager._tools

    def test_discover_lists_each_new_physics(self) -> None:
        """`discover(query='physics', solver=<be>)` output must
        include each new physics name. This is the LLM's first
        check ('does this backend support X?')."""
        discover_fn = self.tools["discover"].fn
        seen_per_solver: dict[str, set[str]] = {}
        for backend, physics in _AUDIT_DRIVEN_PHYSICS:
            out = seen_per_solver.setdefault(
                backend, set())
            if physics not in out:
                # Lazy fetch per solver.
                output = discover_fn(query="physics",
                                     solver=backend)
                # Extract all physics-name tokens from the output.
                for be, ph in _AUDIT_DRIVEN_PHYSICS:
                    if be == backend and ph in output:
                        out.add(ph)
        for backend, physics in _AUDIT_DRIVEN_PHYSICS:
            self.assertIn(
                physics, seen_per_solver[backend],
                f"discover(physics, {backend!r}) does NOT list "
                f"{physics!r}. The PhysicsCapability is probably "
                f"missing from {backend}/backend.py "
                "supported_physics() — every audit-driven "
                "physics must surface here.")

    def test_knowledge_returns_pitfalls_for_each(self) -> None:
        """`knowledge(topic='physics', solver=<be>,
        physics=<ph>)` must return a description + at least one
        pitfall. Empty / 'not found' replies break the LLM's
        ability to learn the API traps."""
        knowledge_fn = self.tools["knowledge"].fn
        for backend, physics in _AUDIT_DRIVEN_PHYSICS:
            out = knowledge_fn(topic="physics",
                               solver=backend,
                               physics=physics)
            self.assertIn(
                "description", out,
                f"knowledge({backend!r}, {physics!r}) returned "
                f"no 'description' field — KNOWLEDGE dict is "
                f"missing the entry or get_knowledge falls "
                f"through to the wrong source. Got: "
                f"{out[:200]!r}")
            self.assertIn(
                "pitfall", out.lower(),
                f"knowledge({backend!r}, {physics!r}) returned "
                f"no 'pitfall' content — every audit-driven "
                f"physics must ship at least one Tier-0/1 "
                f"verified pitfall.")
            self.assertGreater(
                len(out), 500,
                f"knowledge({backend!r}, {physics!r}) returned "
                f"only {len(out)} bytes — too thin. A real "
                f"audit-driven physics has 4-10 KB of "
                f"description + pitfalls. Suspect a truncation "
                f"or empty-dict fallback.")

    def test_prepare_simulation_returns_template(self) -> None:
        """`prepare_simulation(solver=<be>, physics=<ph>)`
        must return BOTH knowledge AND a runnable template
        code block. This is the LLM's primary workflow tool."""
        prep_fn = self.tools["prepare_simulation"].fn
        for backend, physics in _AUDIT_DRIVEN_PHYSICS:
            out = prep_fn(solver=backend, physics=physics)
            self.assertIn(
                "Knowledge", out,
                f"prepare_simulation({backend!r}, {physics!r}) "
                f"did NOT return a 'Knowledge' section — the "
                f"merge in the workflow is broken or "
                f"get_knowledge returned empty.")
            self.assertIn(
                "```", out,
                f"prepare_simulation({backend!r}, {physics!r}) "
                f"did NOT return a code template block. The LLM "
                f"won't have anything runnable to emit.")
            self.assertGreater(
                len(out), 2000,
                f"prepare_simulation({backend!r}, {physics!r}) "
                f"returned only {len(out)} bytes — too thin. "
                f"Full flow should produce 5-12 KB.")

    def test_examples_search_finds_each(self) -> None:
        """`examples(action='search', keyword=<physics>)` must
        find the new physics. This is the LLM's fallback when
        prepare_simulation gives ambiguous results."""
        examples_fn = self.tools["examples"].fn
        for backend, physics in _AUDIT_DRIVEN_PHYSICS:
            out = examples_fn(keyword=physics)
            self.assertTrue(
                physics in out or backend in out,
                f"examples('search', {physics!r}) returned no "
                f"hits mentioning either {physics!r} or "
                f"{backend!r}. The search index missed the "
                f"new physics — examples_search keyword resolver "
                f"may need to be re-built or the fuzzy-match "
                f"map extended.")


if __name__ == "__main__":
    unittest.main()
