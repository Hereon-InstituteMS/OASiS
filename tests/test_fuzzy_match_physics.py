"""Regression: _fuzzy_match_physics must route short canonical
shorthands (ns, em, pd, cfd) to the right physics via the synonym
map BEFORE falling through to a loose substring scan.

Caught 2026-06-02:
  _fuzzy_match_physics ran loose substring matches on physics
  name AND description first; the synonym map was consulted only
  as a fall-through. Short tokens collided constantly:

    'ns'         -> 'heat'        (substring of 'transient')
    'em'         -> 'eigenvalue'  (substring of 'eigenvalue')
    'pd'         -> 'nonlinear_pde' (substring of 'pde')
    'mechanics'  -> 'fracture'    (description: 'fracture mechanics')

  An LLM that typed the canonical shorthand for navier_stokes,
  maxwell, or peridynamics silently received the wrong physics
  block and the wrong template — a worst-class alignment failure
  because the response *looked* successful.

This test pins:
  1. Each canonical-shorthand synonym (ns, em, cfd, vibration,
     thermal, ...) routes to the synonym target when that target
     exists in the backend's catalog.
  2. Exact physics-name queries still match exactly.
  3. Long descriptive queries ('incompressible', 'magnetostatics')
     still route via the synonym map.
  4. A nonsense query falls through to itself so the caller can
     surface a "no information found" message.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


class TestFuzzyMatchPhysics(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.fenics = get_backend("fenics")
        cls.ngsolve = get_backend("ngsolve")
        if cls.fenics is None or cls.ngsolve is None:
            raise unittest.SkipTest(
                "fenics or ngsolve backend not registered.")

    def _fm(self, backend, query: str) -> str:
        from tools.consolidated import _fuzzy_match_physics
        return _fuzzy_match_physics(backend, query)

    # ── short canonical shorthands ────────────────────────

    def test_ns_routes_to_navier_stokes(self) -> None:
        self.assertEqual(self._fm(self.fenics, "ns"),
                         "navier_stokes")
        self.assertEqual(self._fm(self.ngsolve, "ns"),
                         "navier_stokes")

    def test_em_routes_to_maxwell(self) -> None:
        self.assertEqual(self._fm(self.fenics, "em"),
                         "maxwell")
        self.assertEqual(self._fm(self.ngsolve, "em"),
                         "maxwell")

    def test_cfd_routes_to_navier_stokes(self) -> None:
        self.assertEqual(self._fm(self.fenics, "cfd"),
                         "navier_stokes")

    # ── longer-form shorthands ────────────────────────────

    def test_thermal_routes_to_heat(self) -> None:
        self.assertEqual(self._fm(self.fenics, "thermal"),
                         "heat")

    def test_mechanics_routes_to_linear_elasticity(self) -> None:
        # Before fix: matched 'fracture' because the fracture
        # description contained 'fracture mechanics'.
        self.assertEqual(self._fm(self.fenics, "mechanics"),
                         "linear_elasticity")

    def test_incompressible_routes_to_navier_stokes(self) -> None:
        self.assertEqual(self._fm(self.fenics, "incompressible"),
                         "navier_stokes")

    def test_magnetostatics_exact_match(self) -> None:
        # fenics has magnetostatics as an explicit physics name.
        # The synonym map points 'magnetostatics' to 'maxwell' but
        # the exact-name path runs first; we expect the exact
        # name to win.
        self.assertEqual(
            self._fm(self.fenics, "magnetostatics"),
            "magnetostatics")

    # ── empty / nonsense ──────────────────────────────────

    def test_empty_query_returns_empty(self) -> None:
        self.assertEqual(self._fm(self.fenics, ""), "")
        self.assertEqual(self._fm(self.fenics, "   "), "")

    def test_nonsense_query_passes_through(self) -> None:
        # No exact match, no synonym, no substring — caller
        # decides what to do with it. Must NOT silently map to
        # a coincidental substring physics.
        result = self._fm(self.fenics,
                          "totally_nonexistent_physics_xyz")
        self.assertEqual(result, "totally_nonexistent_physics_xyz")

    # ── name-exact match still wins over synonym ──────────

    def test_exact_name_wins_over_synonym(self) -> None:
        # 'navier_stokes' is itself an exact name; the synonym
        # map also contains 'ns' -> 'navier_stokes' / etc. The
        # exact-name dispatch must run first.
        self.assertEqual(
            self._fm(self.fenics, "navier_stokes"),
            "navier_stokes")
        self.assertEqual(
            self._fm(self.fenics, "linear_elasticity"),
            "linear_elasticity")


if __name__ == "__main__":
    unittest.main()
