"""Regression: fourc inline-mesh routing for the mixture/mixture_3d and
constraint/constraint_3d catalog rows.

Created 2026-06-12. Both rows' 4C generator templates
(src/backends/fourc/generators/mixture.py, constraint.py) returned a
ONE-LINE comment, not a YAML dict, so validate_input() failed with
"Input is not a YAML dictionary" and the probe never reached the run
stage. They are now routed to self-contained inline-mesh structural
inputs (no external mesh files, all parameters defaulted):

    mixture/mixture_3d       -> matched_mixture_3d_input   (NEW)
    constraint/constraint_3d -> matched_constraint_3d_input(NEW)

mixture/mixture_3d is a unit-cube tension test whose material is the 4C
Mixture toolbox (MAT_Mixture -> MIX_Rule_Simple ->
MIX_Constituent_ElastHyper -> ELAST_CoupLogNeoHooke). constraint/
constraint_3d is a unit-cube tension test with a DESIGN POINT COUPLING
CONDITION (multi-point coupling) tying the loaded face's transverse DOFs
together. Both verified live against the 4C binary 2026-06-12: rc=0,
"processor 0 finished normally", VTU output written, < 1 s each.

These tests are GEN-ONLY (no 4C binary needed) so they run in CI: they
pin that generate_input for each row returns a self-contained YAML dict
that validate_input accepts, with zero un-substituted placeholders, an
inline mesh, and the expected physics sections — exactly the original
regression. The live execution gate is the probe sweep
(benchmarks/probe_all_templates.py --backend fourc).
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# Matches generator-template placeholders like <Young_modulus> or
# <mesh_file.e> but NOT YAML flow content or comparison operators.
_PLACEHOLDER = re.compile(r"<[A-Za-z_][A-Za-z0-9_. ]*>")

# The probe params the routing lambdas must tolerate (nz=16 must NOT
# inflate the 3D cube mesh — resolution is keyed off "n").
_PROBE = {
    "kappa": 1.0, "nx": 16, "ny": 16, "nz": 16, "E": 1000.0, "nu": 0.3,
    "rho": 1.0, "Re": 100, "mu": 1.0, "refinements": 4,
    "T_end": 0.01, "dt": 0.001,
}

ROUTED_ROWS = [
    ("mixture", "mixture_3d"),
    ("constraint", "constraint_3d"),
]


class TestFourcInlineMixtureConstraint(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def _gen(self, physics: str, variant: str, params: dict | None = None) -> str:
        return self.backend.generate_input(physics, variant, params or {})

    def test_rows_parse_as_yaml_dict_and_validate(self) -> None:
        """The original bug: the generator templates returned a one-line
        comment, not a YAML dict, so validate_input reported "Input is
        not a YAML dictionary". Pin that the routed inputs parse as a
        dict and validate_input() returns [] — under the probe params."""
        for physics, variant in ROUTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant, _PROBE)
                data = yaml.safe_load(content)
                self.assertIsInstance(
                    data, dict,
                    f"{physics}/{variant} is not a YAML dictionary — "
                    f"the exact original regression.")
                self.assertEqual(
                    self.backend.validate_input(content), [],
                    f"{physics}/{variant} failed validate_input.")

    def test_rows_have_no_placeholders(self) -> None:
        for physics, variant in ROUTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant, _PROBE)
                hits = _PLACEHOLDER.findall(content)
                self.assertFalse(
                    hits,
                    f"{physics}/{variant} emits un-substituted "
                    f"placeholders {hits[:5]}.")

    def test_rows_are_self_contained(self) -> None:
        """Inline-mesh inputs embed NODE COORDS — no external mesh
        file reference the user's work dir won't have."""
        for physics, variant in ROUTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant, _PROBE)
                self.assertIn(
                    "NODE COORDS", content,
                    f"{physics}/{variant} lost its inline mesh.")
                self.assertNotIn(
                    "FILE:", content,
                    f"{physics}/{variant} references an external "
                    f"mesh file — not self-contained.")

    def test_mixture_uses_mat_mixture_toolbox(self) -> None:
        content = self._gen("mixture", "mixture_3d", _PROBE)
        self.assertIn("MAT_Mixture", content)
        self.assertIn("MIX_Rule_Simple", content)
        self.assertIn("MIX_Constituent_ElastHyper", content)
        self.assertIn("SOLID HEX8", content)
        self.assertIn("DIM: 3", content)

    def test_constraint_has_coupling_condition(self) -> None:
        content = self._gen("constraint", "constraint_3d", _PROBE)
        self.assertIn("DESIGN POINT COUPLING CONDITIONS", content)
        # the coupling must actually gather nodes into a design point set
        self.assertIn("DNODE-NODE TOPOLOGY", content)
        self.assertIn("SOLID HEX8", content)
        self.assertIn("DIM: 3", content)

    def test_3d_mesh_not_inflated_by_nz(self) -> None:
        """The probe passes nz=16; the lambdas key resolution off "n"
        (capped <= 6), so the cube must stay small — a 16^3 cube would
        blow past the runtime budget. Count NODE COORDS lines and pin
        them well below what nz=16 (>= 17^3 = 4913 nodes) would give."""
        for physics, variant in ROUTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant, _PROBE)
                n_nodes = len(re.findall(r"NODE \d+ COORD", content))
                self.assertGreater(n_nodes, 0)
                self.assertLess(
                    n_nodes, 1000,
                    f"{physics}/{variant} has {n_nodes} nodes — nz=16 "
                    f"inflated the cube (should cap off 'n' <= 6, "
                    f"i.e. <= 7^3 = 343 nodes).")

    def test_params_override_defaults(self) -> None:
        """params must flow through the routing lambdas."""
        content = self.backend.generate_input(
            "mixture", "mixture_3d", {"E": 222.5})
        self.assertIn("222.5", content)
        content = self.backend.generate_input(
            "constraint", "constraint_3d", {"nu": 0.111})
        self.assertIn("0.111", content)


if __name__ == "__main__":
    unittest.main()
