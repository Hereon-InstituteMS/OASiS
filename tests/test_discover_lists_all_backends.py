"""Regression: discover('list') and discover('capabilities') must
expose EVERY registered backend with its actual availability
status, not silently hide the not-yet-installed ones.

Caught 2026-06-02:
  Both surfaces iterated `available_backends()`, which filters by
  check_availability() == AVAILABLE. With dune-fem and febio not
  yet installed locally, the LLM-facing discover output showed
  only 6 of the 8 backends the FastMCP server instructions
  advertise. An LLM that wanted DUNE-fem or FEBio had no way to
  learn:
    1. that those backends are KNOWN to this MCP, and
    2. how to install them (the install hint that
       check_availability returns).

The user-facing impact: an LLM trying to satisfy a "do biomechanics
with FEBio" request would see no FEBio entry, no install hint, and
no path forward — a worst-class alignment failure because the
response *looked* like FEBio simply was not supported.

Fixes (commit ed303aa+):
  - core/registry.py gains all_backends()
  - consolidated.py imports all_backends and uses it in
    discover('list'), discover('physics'), discover('capabilities')
  - the not-available rows include the install hint inline

This test pins those contracts.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _registered_tools():
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise unittest.SkipTest(f"FastMCP not installed: {exc}")
    from core.registry import load_all_backends
    from tools.consolidated import register_consolidated_tools

    load_all_backends()
    mcp = FastMCP("test")
    register_consolidated_tools(mcp)
    return mcp._tool_manager._tools  # type: ignore[attr-defined]


# Backends the MCP is shipped to know about. If the user removes
# one of these, the test should fail loudly so the discover surface
# stays in sync.
_EXPECTED_BACKENDS = ("fourc", "fenics", "dealii", "ngsolve",
                      "skfem", "kratos", "dune", "febio")


class TestDiscoverListsAllBackends(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.tools = _registered_tools()

    def test_list_includes_every_registered_backend(self) -> None:
        result = self.tools["discover"].fn(query="list")
        for name in _EXPECTED_BACKENDS:
            self.assertIn(
                f"({name})", result,
                f"discover('list') is missing backend {name!r}. "
                f"All registered backends — including not-installed "
                f"ones — must appear so LLMs can learn they exist.")

    def test_list_shows_install_hint_for_unavailable(self) -> None:
        """If a backend is not_installed, the inline install hint
        from check_availability() must be visible so the LLM does
        not need a separate tool call to learn how to install.

        The current discover('list') format is:
            - **Display** (name): not_installed — fmt input
              *hint message*

        Coarser global check: if any 'not_installed' is in the
        output, at least one italic-marked hint must appear.
        """
        result = self.tools["discover"].fn(query="list")
        if "not_installed" in result:
            self.assertRegex(
                result, r"\*[^*]+\*",
                "discover('list') has a not_installed row but no "
                "install hint anywhere in the output.")

    def test_capabilities_includes_every_registered_backend(self) -> None:
        result = self.tools["discover"].fn(query="capabilities")
        # capabilities returns a markdown table — match by
        # display name in the | Solver | column.
        # We check that the table has at least 8 data rows
        # (the 2 header rows + 8 data rows for the 8 backends).
        data_rows = [
            ln for ln in result.splitlines()
            if ln.startswith("| ") and not ln.startswith("| Solver")
            and not ln.startswith("|---")
        ]
        self.assertGreaterEqual(
            len(data_rows), len(_EXPECTED_BACKENDS),
            f"discover('capabilities') has {len(data_rows)} rows; "
            f"expected at least {len(_EXPECTED_BACKENDS)} (one per "
            f"registered backend). Output:\n{result}")

    def test_physics_includes_unavailable_backends(self) -> None:
        result = self.tools["discover"].fn(query="physics")
        # discover('physics') groups by backend display name. The
        # dune backend (display: DUNE-fem) and febio (display:
        # FEBio) are likely not_installed in the test env; both
        # MUST appear as section headers.
        for display in ("DUNE-fem", "FEBio"):
            # If the backend is somehow installed, the section
            # appears without the [not_installed] tag — still fine.
            self.assertIn(
                f"## {display}", result,
                f"discover('physics') missing section for "
                f"{display!r}. Unavailable backends must still "
                f"have their physics enumerated so LLMs know what "
                f"the backend offers before installing.")


if __name__ == "__main__":
    unittest.main()
