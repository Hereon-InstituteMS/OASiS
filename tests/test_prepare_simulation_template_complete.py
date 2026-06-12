"""Regression: prepare_simulation's '## Template' block must
contain the FULL generated template, not a 3000-char prefix.

Audit 2026-06-01: prepare_simulation truncated the template body
at 3000 chars (alongside the json knowledge block which now has
its own carve-out for pitfalls). On the harder physics the
truncation cut off the solver / output / summary code:

  fenics::navier_stokes::2d         3763 chars
  ngsolve::hdivdiv::2d              3239 chars
  ngsolve::nonlinear_elasticity::2d 3380 chars

So the LLM got a template that imports + builds the form but
never solves or writes results. The fix raises the limit to
12000 chars (enough for any reasonable Layer F-class template,
typically 2-5KB) and adds an explicit "[truncated N chars]"
marker on the rare overflow.

This test pins: the last 100 chars of each backend.generate_input
output appear verbatim in the prepare_simulation response.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


class _StubMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco

    def resource(self, *args, **kwargs):
        def deco(fn):
            return fn
        return deco

    def prompt(self, *args, **kwargs):
        def deco(fn):
            return fn
        return deco


def _prepare_simulation_fn():
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _StubMCP()
    register_consolidated_tools(mcp)
    return mcp.tools["prepare_simulation"]


class TestTemplateBodyNotTruncated(unittest.TestCase):
    """The template body in prepare_simulation must contain
    the full backend.generate_input output for the worst-case
    long Layer F templates."""

    # Picked from the catalog audit: each was > 3000 chars and
    # therefore HAD its solver/output block silently chopped
    # before the fix.
    LONG_TEMPLATES = [
        ("fenics",  "navier_stokes"),
        ("ngsolve", "hdivdiv"),
        ("ngsolve", "nonlinear_elasticity"),
    ]

    def test_long_templates_are_complete(self) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        prepare = _prepare_simulation_fn()
        for solver, physics in self.LONG_TEMPLATES:
            with self.subTest(solver=solver, physics=physics):
                backend = get_backend(solver)
                self.assertIsNotNone(backend)
                cap = next(c for c in backend.supported_physics()
                           if c.name == physics)
                variant = cap.template_variants[0]
                content = backend.generate_input(physics, variant, {})
                self.assertGreater(
                    len(content), 3000,
                    f"{solver}::{physics}::{variant} is not "
                    "long enough to exercise the truncation "
                    "bug — pick a longer template.")
                response = prepare(solver=solver, physics=physics)
                self.assertIn(
                    content[-200:], response,
                    f"prepare_simulation('{solver}', "
                    f"'{physics}') response does NOT contain "
                    "the last 200 chars of the generated "
                    "template — the template was truncated. "
                    "Raise the TEMPLATE_LIMIT in "
                    "src/tools/consolidated.py.")


def _examples_fn():
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _StubMCP()
    register_consolidated_tools(mcp)
    return mcp.tools["examples"]


class TestKnowledgeJsonNotTruncated(unittest.TestCase):
    """Parallel contract for the knowledge JSON section in
    prepare_simulation. After the pitfalls carve-out the
    remaining JSON is small for most backends but fourc has
    ~12 KB of rich materials/plasticity_models/typical_
    experiments per physics — those used to disappear at the
    3000-char cap."""

    LARGE_KNOWLEDGE = [
        ("fourc", "solid_mechanics"),
    ]

    def test_knowledge_body_renders_in_full(self) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        prepare = _prepare_simulation_fn()
        for solver, physics in self.LARGE_KNOWLEDGE:
            with self.subTest(solver=solver, physics=physics):
                backend = get_backend(solver)
                self.assertIsNotNone(backend)
                k = backend.get_knowledge(physics)
                self.assertIsInstance(k, dict)
                self.assertGreater(
                    len(str(k)), 3000,
                    f"{solver}::{physics} not large enough — "
                    "pick a richer KNOWLEDGE block.")
                response = prepare(solver=solver, physics=physics)
                # Hit a couple of keys that live past the
                # 3000-char cut in the JSON dump.
                for marker in ("typical_experiments",
                               "plasticity_pitfalls"):
                    self.assertIn(
                        marker, response,
                        f"prepare_simulation('{solver}', "
                        f"'{physics}') response does NOT "
                        f"contain '{marker}' — knowledge JSON "
                        "was truncated. Match the "
                        "KNOWLEDGE_LIMIT to TEMPLATE_LIMIT "
                        "in src/tools/consolidated.py.")


class TestExamplesSearchTemplateNotTruncated(unittest.TestCase):
    """Parallel contract for examples(action='search'): the
    matched template body must NOT be truncated mid-code. The
    two surfaces share the same TEMPLATE_LIMIT — if either
    regresses to 3000 this test catches the divergence."""

    LONG_TEMPLATES = [
        ("fenics",  "navier_stokes"),
        ("ngsolve", "hdivdiv"),
        ("ngsolve", "nonlinear_elasticity"),
    ]

    def test_examples_search_renders_full_template(self) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        ex = _examples_fn()
        for solver, physics in self.LONG_TEMPLATES:
            with self.subTest(solver=solver, physics=physics):
                backend = get_backend(solver)
                self.assertIsNotNone(backend)
                cap = next(c for c in backend.supported_physics()
                           if c.name == physics)
                full = backend.generate_input(
                    physics, cap.template_variants[0], {})
                self.assertGreater(
                    len(full), 3000,
                    f"{solver}::{physics} not long enough to "
                    "exercise the truncation bug — pick a "
                    "longer template.")
                resp = ex(keyword=physics, solver=solver,
                          action="search")
                self.assertIn(
                    full[-200:], resp,
                    f"examples('{physics}', solver='{solver}', "
                    "action='search') response does NOT contain "
                    "the last 200 chars of the generated "
                    "template — the truncation regressed. The "
                    "examples and prepare_simulation truncation "
                    "limits must stay in sync.")


if __name__ == "__main__":
    unittest.main()
