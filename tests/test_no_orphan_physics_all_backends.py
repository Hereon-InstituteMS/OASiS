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
    list anywhere reachable from the backend — both the
    deep-knowledge dicts and the per-physics generator
    KNOWLEDGE dicts."""
    import importlib

    keys: set[str] = set()

    # 1. data/*_knowledge.py shape (fourc_knowledge,
    #    kratos_knowledge).
    try:
        mod = importlib.import_module(f"{backend_name}_knowledge")
    except ImportError:
        mod = None
    if mod is not None:
        d = getattr(mod, f"{backend_name.upper()}_KNOWLEDGE", None)
        if isinstance(d, dict):
            keys.update(
                k for k, v in d.items()
                if isinstance(v, dict) and "pitfalls" in v
            )

    # 2. tools/deep_knowledge.py shape (FENICS, DEALII, FEBIO).
    try:
        dk_mod = importlib.import_module("tools.deep_knowledge")
        d = getattr(
            dk_mod, f"_{backend_name.upper()}_KNOWLEDGE", None)
        if isinstance(d, dict):
            keys.update(
                k for k, v in d.items()
                if isinstance(v, dict) and "pitfalls" in v
            )
    except ImportError:
        pass

    # 3. Per-physics generator KNOWLEDGE dicts (one .py per
    #    physics under backends.<be>.generators). Some keys
    #    (e.g. kratos _auxiliary_overview) only live here.
    try:
        gen_pkg = importlib.import_module(
            f"backends.{backend_name}.generators")
        gen_dir = Path(gen_pkg.__file__).parent
        for fname in sorted(gen_dir.iterdir()):
            if (fname.suffix == ".py"
                    and not fname.name.startswith("_")
                    and "base" not in fname.name):
                modname = (f"backends.{backend_name}."
                           f"generators.{fname.stem}")
                try:
                    m = importlib.import_module(modname)
                except Exception:  # noqa: BLE001
                    continue
                K = getattr(m, "KNOWLEDGE", None)
                if isinstance(K, dict):
                    keys.update(
                        k for k, v in K.items()
                        if isinstance(v, dict) and "pitfalls" in v
                    )
    except ImportError:
        pass

    return keys


class TestNoOrphanPhysics(unittest.TestCase):
    def test_no_orphan_physics_in_any_backend(self) -> None:
        from core.registry import get_backend, load_all_backends

        load_all_backends()
        offending: dict[str, list[str]] = {}
        for backend_name in ("fenics", "fourc", "dealii",
                             "kratos", "ngsolve", "skfem",
                             "dune", "febio"):
            backend = get_backend(backend_name)
            if backend is None:
                continue
            exposed = {c.name for c in backend.supported_physics()}
            # Recognise the convention where a backend exposes
            # a leading-underscore key (e.g. kratos
            # _auxiliary_overview) as the public name with the
            # underscore stripped (auxiliary_overview).
            exposed |= {f"_{n}" for n in exposed}
            dk_keys = _backend_dk_keys(backend_name)
            orphans = (dk_keys - exposed) - _REFERENCE_KEYS
            if orphans:
                offending[backend_name] = sorted(orphans)

        # All orphans tracked in this campaign now closed
        # (fenics 2026-06-01 commit 3f75b23, fourc f19e6ae,
        # dealii 45b9d3f, kratos auxiliary_overview this commit).
        # Exemptions list is intentionally empty — any new
        # orphan must be wired up or whitelisted before this
        # test goes green again.

        self.assertEqual(
            offending, {},
            f"deep_knowledge has physics not exposed via "
            f"supported_physics — orphans by backend: "
            f"{offending}. Add PhysicsCapability entries or "
            f"whitelist as reference catalogs.")


if __name__ == "__main__":
    unittest.main()
