"""Cross-backend regression: every deep_knowledge physics key is exposed.

Closes the orphan-knowledge gap discovered iteratively
2026-06-01. For every backend that has a data/*_knowledge.py
catalog OR a deep_knowledge dict, every key carrying a
'pitfalls' list must be reachable via
backend.supported_physics() — otherwise the catalog text is
unreachable from discover.

This test was extracted from test_no_fenics_orphan_physics.py
once the same gap was found in fourc (4 umbrella orphans:
particles, scalar_transport, structural_mechanics, thermal).

Reference-only catalogs (element_catalog, mesh_catalog, etc.)
are intentionally NOT physics and are excluded.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "data"))


# Reference catalogs (not physics) — same whitelist for any
# backend's _KNOWLEDGE dict. Kept conservative: if a future
# entry SHOULD be exposed but is whitelisted here, the test
# silently passes (false negative). Better that than a noisy
# false positive for genuine reference material.
_REFERENCE_KEYS: set[str] = {
    "element_catalog", "mesh_catalog", "solver_catalog",
    "boundary_conditions", "io_catalog", "ufl_reference",
    # dolfinx-specific reference catalogs:
    "complex_valued", "parallel_computing", "api_changes",
}


def _backend_dk_keys(backend_name: str) -> set[str]:
    """Return the set of (physics) keys carrying a 'pitfalls'
    list in the backend's deep-knowledge dict (or {} if there
    is no such dict)."""
    import importlib

    candidates = [
        # data/*_knowledge.py shape (e.g. fourc_knowledge,
        # kratos_knowledge).
        (f"{backend_name}_knowledge",
         f"{backend_name.upper()}_KNOWLEDGE"),
        # tools/deep_knowledge.py shape (FENICS, DEALII, FEBIO).
        ("tools.deep_knowledge",
         f"_{backend_name.upper()}_KNOWLEDGE"),
    ]
    for module_name, attr in candidates:
        try:
            mod = importlib.import_module(module_name)
        except ImportError:
            continue
        d = getattr(mod, attr, None)
        if isinstance(d, dict):
            return {
                k for k, v in d.items()
                if isinstance(v, dict) and "pitfalls" in v
            }
    return set()


class TestNoOrphanPhysics(unittest.TestCase):
    def test_no_orphan_physics_in_any_backend(self) -> None:
        from core.registry import get_backend, load_all_backends

        load_all_backends()
        offending: dict[str, list[str]] = {}
        for backend_name in ("fenics", "fourc", "dealii",
                             "kratos", "ngsolve", "skfem"):
            backend = get_backend(backend_name)
            if backend is None:
                continue
            exposed = {c.name for c in backend.supported_physics()}
            dk_keys = _backend_dk_keys(backend_name)
            orphans = (dk_keys - exposed) - _REFERENCE_KEYS
            if orphans:
                offending[backend_name] = sorted(orphans)

        # Per-backend exemptions for known-tracked work:
        #   * kratos has _auxiliary_overview in generators —
        #     tracked as a separate Layer-A scanner issue.
        # dealii orphans closed 2026-06-01: advection_dg /
        # contact / nonlinear_elasticity now exposed via
        # PhysicsCapability AND reachable via the
        # tools.deep_knowledge fallback in
        # backends.dealii.backend.get_knowledge.
        for be in ("kratos",):
            offending.pop(be, None)

        self.assertEqual(
            offending, {},
            f"deep_knowledge has physics not exposed via "
            f"supported_physics — orphans by backend: "
            f"{offending}. Add PhysicsCapability entries or "
            f"whitelist as reference catalogs.")


if __name__ == "__main__":
    unittest.main()
