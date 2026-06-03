"""Regression: the _PHYSICS_SYNONYMS map must route natural-English
terminology to the right canonical physics in each backend.

Built 2026-06-03 after Layer G audit surfaced that users searching for
common terms (heat_transfer, structural_2d, amr, plane_strain) silently
got zero pitfalls because the catalog used different canonical names
(heat, linear_elasticity, hp_adaptive). The catalog content was there;
the discovery path was broken.

Pins:
  1. Zero orphan aliases — every entry's target exists in at least one
     backend's supported_physics().
  2. Each of the originally-broken queries routes correctly per backend.
  3. Cross-backend uniformity — common terms route the same way across
     every backend that supports the canonical target.
"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


class TestPhysicsAliasMap(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, all_backends
        load_all_backends()
        cls.backends = {b.name(): b for b in all_backends()}
        cls.all_canonical: set[str] = set()
        for b in cls.backends.values():
            cls.all_canonical.update(p.name for p in b.supported_physics())

    def _fm(self, backend_name: str, query: str) -> str:
        from tools.consolidated import _fuzzy_match_physics
        b = self.backends.get(backend_name)
        if b is None:
            self.skipTest(f"backend {backend_name!r} not registered")
        return _fuzzy_match_physics(b, query)

    # ── 1. Map-level integrity ─────────────────────────────────────

    def test_no_orphan_aliases(self) -> None:
        """Every alias must route to a canonical name that exists in at
        least one backend. Orphans would be silent UX failures: the
        resolver consults the synonym, looks up the canonical in
        backend.supported_physics(), doesn't find it, falls through to
        substring matching which usually fails for short queries."""
        from tools.consolidated import _PHYSICS_SYNONYMS
        orphans = [
            (alias, target) for alias, target in _PHYSICS_SYNONYMS.items()
            if target not in self.all_canonical
        ]
        if orphans:
            preview = "\n".join(
                f"  {a!r} -> {t!r}" for a, t in orphans[:10]
            )
            self.fail(
                f"{len(orphans)} alias(es) point to canonical names "
                f"that don't exist in any backend:\n{preview}"
            )

    # ── 2. The originally-broken queries (Layer G audit) ──────────

    def test_heat_transfer_routes_to_heat(self) -> None:
        for be in ("kratos", "fenics", "ngsolve", "dealii",
                   "skfem", "febio", "dune"):
            with self.subTest(backend=be):
                self.assertEqual(self._fm(be, "heat_transfer"), "heat")

    def test_thermal_routes_to_heat(self) -> None:
        for be in ("kratos", "fenics", "ngsolve", "dealii",
                   "skfem", "febio", "dune"):
            with self.subTest(backend=be):
                self.assertEqual(self._fm(be, "thermal"), "heat")

    def test_heat_conduction_routes_to_heat(self) -> None:
        for be in ("kratos", "fenics", "ngsolve"):
            with self.subTest(backend=be):
                self.assertEqual(self._fm(be, "heat_conduction"), "heat")

    def test_structural_2d_routes_to_linear_elasticity(self) -> None:
        """Originally a fourc::structural_2d query returned 0 pitfalls
        because the canonical key is fourc::structural / structural_mechanics
        / solid / solid_mechanics / linear_elasticity. The alias map
        routes all four naming variants to linear_elasticity (which exists
        across all 8 backends, so this lifts every backend uniformly)."""
        self.assertEqual(self._fm("fourc", "structural_2d"),
                         "linear_elasticity")

    def test_plane_strain_routes_to_linear_elasticity(self) -> None:
        for be in ("fourc", "fenics", "ngsolve", "skfem"):
            with self.subTest(backend=be):
                self.assertEqual(self._fm(be, "plane_strain"),
                                 "linear_elasticity")

    def test_plane_stress_routes_to_linear_elasticity(self) -> None:
        for be in ("fourc", "fenics", "ngsolve", "skfem"):
            with self.subTest(backend=be):
                self.assertEqual(self._fm(be, "plane_stress"),
                                 "linear_elasticity")

    def test_amr_routes_to_hp_adaptive_in_dealii(self) -> None:
        """deal.II is the main backend for adaptive refinement;
        the canonical physics key is hp_adaptive (not the knowledge
        key 'adaptive_refinement' which is not exposed via
        supported_physics())."""
        self.assertEqual(self._fm("dealii", "amr"), "hp_adaptive")
        self.assertEqual(self._fm("dealii", "h_refinement"), "hp_adaptive")
        self.assertEqual(self._fm("dealii", "adaptive"), "hp_adaptive")
        self.assertEqual(self._fm("dealii", "refinement"), "hp_adaptive")

    def test_stokes_flow_routes_to_stokes(self) -> None:
        for be in ("fenics", "ngsolve", "skfem", "dune"):
            with self.subTest(backend=be):
                self.assertEqual(self._fm(be, "stokes_flow"), "stokes")

    # ── 3. Cross-backend uniformity of major synonyms ─────────────

    def test_cfd_routes_to_navier_stokes_everywhere(self) -> None:
        for be in ("fenics", "ngsolve", "skfem", "dune", "dealii"):
            with self.subTest(backend=be):
                self.assertEqual(self._fm(be, "cfd"), "navier_stokes")

    def test_em_routes_to_maxwell_everywhere(self) -> None:
        """deal.II is excluded — it doesn't expose maxwell in
        supported_physics() (it ships matrix_free / multigrid /
        obstacle / topology_opt / compressible_euler as advanced
        physics keys but not a standalone maxwell)."""
        for be in ("fenics", "ngsolve", "dune"):
            with self.subTest(backend=be):
                self.assertEqual(self._fm(be, "em"), "maxwell")

    def test_large_strain_routes_to_hyperelasticity(self) -> None:
        for be in ("fenics", "ngsolve", "dealii", "skfem", "febio", "dune"):
            with self.subTest(backend=be):
                self.assertEqual(self._fm(be, "large_strain"),
                                 "hyperelasticity")

    def test_peridynamics_routes_to_particle_pd(self) -> None:
        """Only fourc exposes particle_pd in supported_physics().
        kratos has peridynamics-adjacent work under pfem_* keys
        but no particle_pd canonical, so the alias falls through
        in kratos (correct behaviour — no false routing)."""
        self.assertEqual(self._fm("fourc", "peridynamics"),
                         "particle_pd")

    def test_nosbpd_routes_to_particle_pd(self) -> None:
        """User-specific term (the user's own work uses NOSBPD).
        Confirms the map covers domain-of-practice abbreviations.
        Same backend scope as peridynamics."""
        self.assertEqual(self._fm("fourc", "nosbpd"), "particle_pd")

    def test_vibration_routes_to_eigenvalue(self) -> None:
        for be in ("fenics", "ngsolve", "dealii", "skfem", "dune"):
            with self.subTest(backend=be):
                self.assertEqual(self._fm(be, "vibration"), "eigenvalue")


if __name__ == "__main__":
    unittest.main()
