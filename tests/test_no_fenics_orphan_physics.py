"""Regression: every deep_knowledge fenics physics is reachable from discover.

Closes the orphan-knowledge gap discovered 2026-06-01: helmholtz,
maxwell, and nearly_incompressible_elasticity each had a full
pitfalls list in src/tools/deep_knowledge.py:_FENICS_KNOWLEDGE
but were missing from src/backends/fenics/backend.py's
_PHYSICS_CAPABILITIES list. Users browsing
  discover(physics, solver='fenics')
saw 19 backends and never knew the catalog had those three.

This test fails the moment a new physics key is added to
_FENICS_KNOWLEDGE without also being added to the
PhysicsCapability list.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "data"))


# These _FENICS_KNOWLEDGE keys are reference catalogs, not
# physics types — they intentionally have no PhysicsCapability
# entry. Whitelist them so the test doesn't false-positive.
_REFERENCE_KEYS: set[str] = {
    "element_catalog", "mesh_catalog", "solver_catalog",
    "boundary_conditions", "io_catalog", "ufl_reference",
    "scalar_transport", "solid_mechanics", "fluid", "fsi",
    "beams", "contact", "structural_dynamics", "particle_pd",
    "particle_sph",
    # ↑ These all belong to 4C-side terminology that lives in
    # _FENICS_KNOWLEDGE for historical reasons but is not part
    # of the fenics physics surface.
    "complex_valued", "parallel_computing", "api_changes",
    # ↑ Genuine reference material about DOLFINx itself — not
    # a PDE / physics type.
}


class TestNoFenicsOrphanPhysics(unittest.TestCase):
    def test_every_fenics_knowledge_physics_is_exposed(self) -> None:
        from core.registry import get_backend, load_all_backends
        from tools.deep_knowledge import _FENICS_KNOWLEDGE

        load_all_backends()
        backend = get_backend("fenics")
        assert backend is not None, (
            "fenics backend not registered — test cannot run")
        exposed = {c.name for c in backend.supported_physics()}

        dk_physics = {
            k for k, v in _FENICS_KNOWLEDGE.items()
            if isinstance(v, dict) and "pitfalls" in v
            and k not in _REFERENCE_KEYS
        }

        orphans = dk_physics - exposed
        self.assertEqual(
            orphans, set(),
            f"deep_knowledge has physics not exposed via "
            f"supported_physics: {sorted(orphans)}. "
            f"Add PhysicsCapability entries in "
            f"src/backends/fenics/backend.py or whitelist as "
            f"reference catalogs.")


if __name__ == "__main__":
    unittest.main()
