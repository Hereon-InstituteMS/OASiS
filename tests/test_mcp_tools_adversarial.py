"""Regression: MCP tools handle adversarial inputs gracefully.

Adversarial-fuzzing pass (task #40) — probe corner cases that an
LLM might emit by accident. Coverage:

  prepare_simulation:
    - empty / whitespace-only physics    — must NOT silently
                                            fuzzy-match the first
                                            physics (task #136 root
                                            cause).
    - unknown solver                     — clear "Unknown solver: X"
                                            error.
    - canonical aliases (Poisson, elasticity)
                                         — matched via fuzzy
                                            resolver with *Note:*
                                            breadcrumb.
  discover:
    - query='recommend' with empty solver/physics — task #150
                                            rejection contract.
  examples:
    - empty keyword                      — task #137 guard.

The gate calls registered handlers directly (NOT the MCP wire)
so the test passes regardless of MCP server staleness.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _get_tool(tool_name: str):
    """Re-register all consolidated tools on a fresh FastMCP
    instance and return the underlying callable for `tool_name`."""
    from mcp.server.fastmcp import FastMCP  # type: ignore
    from tools.consolidated import register_consolidated_tools
    from core.registry import load_all_backends
    load_all_backends()

    mcp = FastMCP("test")
    register_consolidated_tools(mcp)
    return mcp._tool_manager._tools[tool_name].fn


def _invoke_prepare_simulation(solver: str, physics: str) -> str:
    return _get_tool("prepare_simulation")(solver=solver,
                                            physics=physics)


class TestPrepareSimulationAdversarial(unittest.TestCase):
    def test_empty_physics_surfaces_available(self) -> None:
        result = _invoke_prepare_simulation("fenics", "")
        self.assertIn("Empty physics query", result,
                      "Empty physics must surface a usage hint, "
                      "NOT silently fuzzy-match the first physics. "
                      f"Got: {result[:200]}")
        self.assertIn("poisson", result.lower(),
                      "The error message must list available physics "
                      "so the LLM can self-correct.")

    def test_whitespace_physics_surfaces_available(self) -> None:
        result = _invoke_prepare_simulation("fenics", "   ")
        self.assertIn("Empty physics query", result,
                      "Whitespace-only physics must be rejected the "
                      "same way as truly empty input.")

    def test_unknown_solver_clean_error(self) -> None:
        result = _invoke_prepare_simulation("does_not_exist", "poisson")
        self.assertIn("Unknown solver", result,
                      "Unknown solver must produce a clean error, "
                      f"not crash. Got: {result[:200]}")

    def test_uppercase_physics_fuzzy_matches(self) -> None:
        result = _invoke_prepare_simulation("fenics", "Poisson")
        self.assertIn("matched to 'poisson'", result,
                      "Case-insensitive physics names must be fuzzy-"
                      "matched with a *Note:* breadcrumb so the LLM "
                      "sees what canonical name was used.")

    def test_alias_physics_fuzzy_matches(self) -> None:
        result = _invoke_prepare_simulation("fenics", "elasticity")
        self.assertIn("matched to 'linear_elasticity'", result,
                      "Known aliases (elasticity → linear_elasticity) "
                      "must surface a *Note:* breadcrumb.")


class TestDiscoverAdversarial(unittest.TestCase):
    """discover('recommend') with empty payload (task #150)."""

    def test_recommend_empty_physics_rejected(self) -> None:
        discover = _get_tool("discover")
        result = discover(query="recommend", solver="")
        self.assertIn("Empty physics", result,
                      "discover(query='recommend', solver='') must "
                      "be rejected — task #150. Got: "
                      f"{result[:200]}")


class TestExamplesAdversarial(unittest.TestCase):
    """examples with empty keyword (task #137)."""

    def test_empty_keyword_rejected(self) -> None:
        examples = _get_tool("examples")
        result = examples(keyword="", solver="fenics", action="search")
        # The guard wording may vary; the contract is that the
        # tool surfaces a usage hint or available-examples list
        # rather than silently returning random matches via the
        # fuzzy-resolver's '' substring match.
        lower = result.lower()
        self.assertTrue(
            ("usage" in lower) or ("provide" in lower)
            or ("specify" in lower) or ("empty" in lower)
            or ("keyword" in lower),
            "examples(keyword='') must guard against the empty "
            "substring match (task #137). Got: "
            f"{result[:200]}")


class TestKnowledgeAdversarial(unittest.TestCase):
    """knowledge tool corner cases."""

    def test_empty_topic_shows_usage(self) -> None:
        knowledge = _get_tool("knowledge")
        result = knowledge(topic="")
        self.assertIn("Topics:", result,
                      "knowledge(topic='') must surface a usage "
                      "hint listing all valid topics.")
        # Defense in depth: the postmortems topic was added in
        # task #147 and the usage hint must include it.
        self.assertIn("postmortems", result,
                      "knowledge usage hint must list "
                      "'postmortems' topic (task #147).")

    def test_unknown_topic_shows_usage(self) -> None:
        knowledge = _get_tool("knowledge")
        result = knowledge(topic="nonsense_topic")
        # Unknown topic should NOT silently return empty —
        # surface the same usage hint so the LLM can pick.
        self.assertIn("Topics:", result,
                      "knowledge(topic='nonsense') must fall back "
                      "to the usage hint.")

    def test_unknown_solver_clean_error(self) -> None:
        knowledge = _get_tool("knowledge")
        result = knowledge(topic="physics", solver="ghost",
                            physics="poisson")
        self.assertIn("Unknown solver", result,
                      f"Expected clean 'Unknown solver' error. "
                      f"Got: {result[:200]}")

    def test_overview_surfaces_general_catalog(self) -> None:
        """knowledge(topic='overview', solver=<6-of-8>) must
        return the backend-level _general reference catalog as
        JSON. Before 2026-06-02 this content was reachable only
        via the internal get_knowledge('_general') call — LLMs
        had no MCP-visible path to discover it. (Task #52.)"""
        knowledge = _get_tool("knowledge")
        # The 6 backends with substantive _general content.
        # dealii is the largest (~5.2 KB); the others are 1-2 KB.
        for solver in ("dealii", "fenics", "ngsolve", "skfem",
                       "kratos", "dune"):
            result = knowledge(topic="overview", solver=solver)
            self.assertIn(f'"{solver}"', result,
                f"knowledge(topic='overview', solver={solver!r}) "
                f"must return JSON keyed by solver name. "
                f"Got: {result[:200]}")
            self.assertIn("description", result,
                f"overview for {solver} must contain the "
                f"'description' field from _general. Got: "
                f"{result[:200]}")

    def test_overview_missing_solver(self) -> None:
        knowledge = _get_tool("knowledge")
        result = knowledge(topic="overview", solver="")
        # No solver → fall through to usage hint.
        self.assertIn("Topics:", result,
            "knowledge(topic='overview') without solver must "
            "fall back to usage hint, not silently return empty.")
        self.assertIn("overview", result,
            "knowledge usage hint must list 'overview' topic "
            "(task #52).")

    def test_overview_unknown_solver_clean_error(self) -> None:
        knowledge = _get_tool("knowledge")
        result = knowledge(topic="overview", solver="ghost")
        self.assertIn("Unknown solver", result,
            f"overview with bogus solver must clean-error. "
            f"Got: {result[:200]}")

    def test_overview_empty_general_clean_message(self) -> None:
        """fourc has empty _general; the branch must emit a
        self-explanatory message instead of crashing or
        returning '{}'. (febio used to be on this list, but the
        FEBio refactor passes populated febio _general with real
        embedding/AMR content — verify by inverse: febio's
        overview MUST now return substantive JSON.)"""
        knowledge = _get_tool("knowledge")
        # Direct branch: fourc _general still empty.
        result_fourc = knowledge(topic="overview", solver="fourc")
        self.assertIn("No backend-level overview catalog", result_fourc,
            f"overview for fourc (empty _general) must emit a clear "
            f"no-content message. Got: {result_fourc[:200]}")
        # Inverse anchor: if a future refactor empties febio _general,
        # this assertion fails first, prompting an update here AND in
        # whichever generator was edited.
        result_febio = knowledge(topic="overview", solver="febio")
        self.assertNotIn("No backend-level overview catalog",
                         result_febio,
            "febio _general was populated by the FEBio refactor; "
            "overview must now return substantive JSON, not the "
            "empty-catalog message. If this fails, restore the "
            "febio _general content or move febio back into the "
            "empty-catalog loop.")
        self.assertGreater(
            len(result_febio), 5000,
            f"febio overview should be substantial (>5k chars); "
            f"got {len(result_febio)} chars.")


class TestDeveloperAdversarial(unittest.TestCase):
    """developer tool corner cases."""

    def test_architecture_returns_json(self) -> None:
        dev = _get_tool("developer")
        result = dev(action="architecture", solver="fenics")
        self.assertIn("\"root\"", result,
                      "developer('architecture') must return JSON "
                      "with at least the 'root' key for the requested "
                      "solver. Got: " + result[:200])

    def test_unknown_action_shows_usage(self) -> None:
        dev = _get_tool("developer")
        result = dev(action="nonsense_action", solver="fenics")
        self.assertIn("Usage", result,
                      "developer('nonsense_action') must fall back "
                      "to a usage hint listing valid actions. Got: "
                      + result[:200])


class TestIntrospectionToolsSmoke(unittest.TestCase):
    """Smoke gates for the introspection tools — confirm they
    return a non-empty markdown response and don't crash on
    default arguments."""

    def test_rediscover_backends_default(self) -> None:
        rb = _get_tool("rediscover_backends")
        result = rb(confirm=False)
        # The tool must enumerate backends found / missing and
        # list at least one of each. Confirms the discovery
        # probe path doesn't silently return empty.
        self.assertIn("Backend Discovery", result,
                      "rediscover_backends must surface the standard "
                      "'Backend Discovery' header.")
        self.assertTrue(
            "Available" in result or "Not found" in result,
            "rediscover_backends must list at least the "
            "Available / Not found sections.")

    def test_reload_catalog_default(self) -> None:
        rl = _get_tool("reload_catalog")
        result = rl()
        self.assertIn("reload_catalog", result,
                      "reload_catalog must echo its own name in "
                      "the summary line.")
        self.assertIn("modules reloaded", result,
                      "reload_catalog summary must report a "
                      "'modules reloaded' count.")
        # Reloads should not fail for a healthy tree.
        self.assertIn("0 failed", result,
                      "reload_catalog reported failures — module "
                      "import is broken somewhere. "
                      f"Got: {result[:300]}")


class TestSimulationToolsUnknownSolver(unittest.TestCase):
    """Smoke gates for the run / coupled_solve tools — each
    must produce a clean 'Unknown solver' / 'Backend not found'
    error rather than crashing when handed an unrecognised
    solver name. Full real-backend exercises are covered by
    Layer-D / Layer-F suites; this gate just defends the
    error path."""

    def test_run_simulation_unknown_solver(self) -> None:
        import asyncio
        rs = _get_tool("run_simulation")
        result = asyncio.run(rs(solver="ghost_solver",
                                 input_content="print('ok')"))
        self.assertIn("Unknown solver", str(result),
                      "run_simulation must produce a clean "
                      "'Unknown solver' error on an unrecognised "
                      f"solver. Got: {str(result)[:200]}")

    def test_run_with_generator_unknown_solver(self) -> None:
        import asyncio
        rwg = _get_tool("run_with_generator")
        result = asyncio.run(rwg(solver="ghost_solver",
                                  generator_script="print('ok')"))
        self.assertIn("Unknown solver", str(result),
                      "run_with_generator must produce a clean "
                      "'Unknown solver' error on an unrecognised "
                      f"solver. Got: {str(result)[:200]}")

    def test_coupled_solve_unknown_backend(self) -> None:
        import asyncio
        cs = _get_tool("coupled_solve")
        result = asyncio.run(cs(problem="poisson_dd",
                                 solver_a="ghost",
                                 solver_b="fenics"))
        self.assertIn("not found", str(result).lower(),
                      "coupled_solve must surface a clean "
                      "'Backend not found' error when either side "
                      "is unrecognised. Got: " + str(result)[:200])


if __name__ == "__main__":
    unittest.main()
