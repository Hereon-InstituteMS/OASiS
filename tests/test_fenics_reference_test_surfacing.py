"""Regression: prepare_simulation must surface real FEniCS demos.

Catalog audit 2026-06-01: _find_reference_test_files('fenics', 'poisson')
returned 0 chars because the fenics-demo lookup was double-broken:

  1. Hardcoded path ~/miniconda3/envs/fenics/share/dolfinx/demo did
     not exist (conda-forge moved demos under
     etc/conda/test-files/fenics-dolfinx/0/python/demo and the env
     on this machine is named ofa-fenicsx, not fenics).

  2. The search_terms map biases to 4C / deal.II filename
     conventions (poisson -> scatra, heat -> thermo) — those keys
     do NOT match dolfinx demo filenames (demo_poisson.py,
     demo_helmholtz.py).

Fix probes multiple candidate paths AND falls through to the
raw physics name + hyphenated variants + common-prefix-stripped
forms (linear_elasticity -> elasticity).

This test pins the contract: if the fenics conda env is present,
prepare_simulation('fenics', X) surfaces at least one demo file
for every X in a small canonical set.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _fenics_demo_dir_exists() -> bool:
    """Heuristic: any local conda env carrying fenics demos."""
    conda_envs = Path.home() / "miniconda3" / "envs"
    if not conda_envs.is_dir():
        return False
    for env in conda_envs.iterdir():
        if "fenics" not in env.name.lower():
            continue
        for candidate in (
            env / "share" / "dolfinx" / "demo",
            env / "etc" / "conda" / "test-files"
            / "fenics-dolfinx" / "0" / "python" / "demo",
        ):
            if candidate.is_dir():
                return True
    return False


class TestFenicsReferenceTestSurfacing(unittest.TestCase):
    """If the fenics conda env is locally available, every
    canonical physics must surface at least one demo."""

    CANONICAL_PHYSICS = (
        "poisson", "helmholtz", "stokes", "navier_stokes",
        "biharmonic", "mixed_poisson", "cahn_hilliard",
        "linear_elasticity",
    )

    def test_demos_surface(self) -> None:
        if not _fenics_demo_dir_exists():
            self.skipTest("No fenics conda env on this machine — "
                          "skipping reference-test surfacing check.")
        from core.registry import load_all_backends
        load_all_backends()
        from tools.knowledge import _find_reference_test_files
        empty: list[str] = []
        for ph in self.CANONICAL_PHYSICS:
            out = _find_reference_test_files("fenics", ph)
            if not out or len(out) < 100:
                empty.append(ph)
        if empty:
            self.fail(
                f"fenics reference-test surfacing returns empty for "
                f"{len(empty)}/{len(self.CANONICAL_PHYSICS)} "
                f"canonical physics: {empty}. The dolfinx demos "
                "directory is present but the search-term resolver "
                "fails to find matching demo_<X>.py files. Check "
                "src/tools/knowledge.py:_find_reference_test_files "
                "— the keyword resolution probably regressed back "
                "to a 4C / deal.II-only search_terms map.")


if __name__ == "__main__":
    unittest.main()
