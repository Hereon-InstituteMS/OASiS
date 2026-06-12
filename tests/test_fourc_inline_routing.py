"""Regression: fourc inline-mesh routing for previously-broken catalog rows.

Created 2026-06-12 (task #29). The probe sweep found 34 fourc templates
aborting in 4C's MatchTree because generate_input() fell through to
generator templates carrying literal <placeholder> scalars
(`YOUNG: <Young_modulus>`, `FILE: <mesh_file.e>`) — unrunnable as
emitted. Five rows were closed by routing them to the proven
self-contained inline-mesh inputs (no external mesh files, all
parameters defaulted):

    solid_mechanics/linear_2d         -> matched_elasticity_input
    scalar_transport/poisson_2d       -> matched_poisson_input
    scalar_transport/heat_transient_2d-> matched_heat_transient_input (NEW)
    structural_dynamics/genalpha_2d   -> matched_elasticity_genalpha_input (NEW)
    solid_mechanics/nonlinear_3d      -> matched_elasticity_3d_nonlinear_input (NEW)

All five verified live against the 4C binary 2026-06-12: rc=0 with
VTU output (transient rows produce one file per step).

These tests are GEN-ONLY (no 4C binary needed) so they run in CI:
they pin that generate_input for each row returns self-contained
YAML with zero un-substituted placeholders and an inline mesh.
The live execution gate is the probe sweep
(benchmarks/probe_all_templates.py --backend fourc).
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# Matches generator-template placeholders like <Young_modulus> or
# <mesh_file.e> but NOT YAML flow content or comparison operators.
_PLACEHOLDER = re.compile(r"<[A-Za-z_][A-Za-z0-9_. ]*>")

ROUTED_ROWS = [
    ("solid_mechanics", "linear_2d"),
    ("scalar_transport", "poisson_2d"),
    ("scalar_transport", "heat_transient_2d"),
    ("structural_dynamics", "genalpha_2d"),
    ("solid_mechanics", "nonlinear_3d"),
]


class TestFourcInlineRouting(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def _gen(self, physics: str, variant: str) -> str:
        return self.backend.generate_input(physics, variant, {})

    def test_routed_rows_have_no_placeholders(self) -> None:
        """The five routed rows must emit fully-substituted YAML —
        a literal <...> scalar in the output is exactly the bug that
        made 4C MPI_Abort in MatchTree (probe 2026-06-12)."""
        for physics, variant in ROUTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant)
                hits = _PLACEHOLDER.findall(content)
                self.assertFalse(
                    hits,
                    f"{physics}/{variant} emits un-substituted "
                    f"placeholders {hits[:5]} — it regressed from the "
                    f"inline-mesh route back to a raw generator "
                    f"template.")

    def test_routed_rows_are_self_contained(self) -> None:
        """Inline-mesh inputs embed NODE COORDS — no external mesh
        file reference that the user's work dir won't have."""
        for physics, variant in ROUTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant)
                self.assertIn(
                    "NODE COORDS", content,
                    f"{physics}/{variant} lost its inline mesh.")
                self.assertNotIn(
                    "FILE:", content,
                    f"{physics}/{variant} references an external "
                    f"mesh file — not self-contained.")

    def test_transient_heat_is_actually_transient(self) -> None:
        content = self._gen("scalar_transport", "heat_transient_2d")
        self.assertIn("One_Step_Theta", content,
                      "heat_transient_2d must use a transient "
                      "integrator, not Stationary.")
        m = re.search(r"NUMSTEP:\s*(\d+)", content)
        self.assertIsNotNone(m)
        self.assertGreater(int(m.group(1)), 1,
                           "transient run needs NUMSTEP > 1")

    def test_genalpha_is_dynamic_with_mass(self) -> None:
        content = self._gen("structural_dynamics", "genalpha_2d")
        self.assertIn("GenAlpha", content)
        m = re.search(r"DENS:\s*([0-9.eE+-]+)", content)
        self.assertIsNotNone(m)
        self.assertGreater(float(m.group(1)), 0.0,
                           "GenAlpha dynamics with DENS=0 has no "
                           "inertia — the transient is meaningless.")

    def test_nonlinear_3d_is_finite_strain_hex(self) -> None:
        content = self._gen("solid_mechanics", "nonlinear_3d")
        self.assertIn("KINEM nonlinear", content)
        self.assertIn("SOLID HEX8", content)
        self.assertIn("DIM: 3", content)

    def test_params_override_defaults(self) -> None:
        """params must flow through the routing lambdas."""
        content = self.backend.generate_input(
            "solid_mechanics", "linear_2d", {"E": 123.5})
        self.assertIn("123.5", content)
        content = self.backend.generate_input(
            "scalar_transport", "heat_transient_2d", {"numstep": 7})
        self.assertRegex(content, r"NUMSTEP:\s*7\b")


if __name__ == "__main__":
    unittest.main()
