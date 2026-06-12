"""Regression: fourc inline-mesh routing for the thermo catalog rows.

Created 2026-06-12 (layer-a probe sweep). thermo/thermo_2d and
thermo/thermo_3d were worse than the placeholder rows: the generator's
get_template() returned a ONE-LINE comment
("# Thermal template — use THERMO QUAD4/HEX8 elements with MAT_Fourier"),
not even a YAML dict, so validate_input() failed with "Input is not a
YAML dictionary" and the probe never reached the run stage.

Both rows are now routed in backend._generate_inline() to genuine
PROBLEMTYPE "Thermo" inline-mesh inputs:

    thermo/thermo_2d -> matched_thermo_2d_input  (NEW)
        [0,1]^2 QUAD4, THERMO QUAD4 + MAT_Fourier, steady (Statics),
        T_left=100 / T_right=0 Dirichlet on opposite edges.
    thermo/thermo_3d -> matched_thermo_3d_input  (NEW)
        unit cube HEX8, THERMO HEX8 + MAT_Fourier, transient
        (OneStepTheta), resolution keyed off "n" (NOT nx/ny/nz) so a
        generic parameter sweep cannot inflate the mesh.

Both verified live against the 4C binary 2026-06-12: rc=0
("processor 0 finished normally"), < 1 s each.

These tests are GEN-ONLY (no 4C binary needed) so they run in CI:
they pin that generate_input for each row returns a self-contained
YAML *dictionary* (the original failure mode) with zero
un-substituted placeholders, an inline mesh, and real Thermo physics.
The live execution gate is the probe sweep
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

# Probe parameter set the routing lambdas must tolerate (note nz=16:
# it must NOT inflate the 3D mesh, which is keyed off "n" instead).
PROBE_PARAMS = {
    "kappa": 1.0, "nx": 16, "ny": 16, "nz": 16,
    "E": 1000.0, "nu": 0.3, "rho": 1.0, "Re": 100, "mu": 1.0,
    "refinements": 4, "T_end": 0.01, "dt": 0.001,
}

ROUTED_ROWS = [
    ("thermo", "thermo_2d"),
    ("thermo", "thermo_3d"),
]


class TestFourcInlineThermo(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def _gen(self, physics: str, variant: str, params: dict | None = None) -> str:
        return self.backend.generate_input(physics, variant,
                                           dict(params or PROBE_PARAMS))

    def test_rows_are_yaml_dicts_and_validate(self) -> None:
        """THE regression for these rows: get_template() used to return
        a one-line comment, so validate_input() failed with 'Input is
        not a YAML dictionary' before the run stage."""
        for physics, variant in ROUTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant)
                parsed = yaml.safe_load(content)
                self.assertIsInstance(
                    parsed, dict,
                    f"{physics}/{variant} does not emit a YAML "
                    f"dictionary — regressed to the comment-only "
                    f"template.")
                self.assertEqual(
                    self.backend.validate_input(content), [],
                    f"{physics}/{variant} fails validate_input().")

    def test_rows_have_no_placeholders(self) -> None:
        for physics, variant in ROUTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant)
                hits = _PLACEHOLDER.findall(content)
                self.assertFalse(
                    hits,
                    f"{physics}/{variant} emits un-substituted "
                    f"placeholders {hits[:5]}.")

    def test_rows_are_self_contained(self) -> None:
        """Inline-mesh inputs embed NODE COORDS — no external mesh
        file reference that the user's work dir won't have."""
        for physics, variant in ROUTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant)
                self.assertIn("NODE COORDS", content,
                              f"{physics}/{variant} lost its inline mesh.")
                self.assertNotIn("FILE:", content,
                                 f"{physics}/{variant} references an "
                                 f"external mesh file.")

    def test_rows_are_genuine_thermo_physics(self) -> None:
        """Must be the real Thermo problemtype with THERMO elements and
        MAT_Fourier — not a scatra-based heat surrogate."""
        expected_elem = {"thermo_2d": "THERMO QUAD4",
                         "thermo_3d": "THERMO HEX8"}
        for physics, variant in ROUTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant)
                parsed = yaml.safe_load(content)
                self.assertEqual(
                    parsed["PROBLEM TYPE"]["PROBLEMTYPE"], "Thermo",
                    f"{physics}/{variant} is not PROBLEMTYPE Thermo.")
                self.assertIn("THERMAL DYNAMIC", parsed)
                self.assertIn(expected_elem[variant], content)
                self.assertIn("MAT_Fourier", content)
                self.assertNotIn("MAT_scatra", content)

    def test_3d_mesh_not_inflated_by_probe_nz(self) -> None:
        """3D resolution is keyed off 'n' (default 6, cap 8); the
        probe's nx=ny=nz=16 must not blow up the cube mesh."""
        content = self._gen("thermo", "thermo_3d")  # PROBE_PARAMS, nz=16
        parsed = yaml.safe_load(content)
        n_elems = len(parsed["THERMO ELEMENTS"])
        self.assertEqual(n_elems, 6 ** 3,
                         "probe nz=16 inflated the 3D thermo mesh — "
                         "resolution must come from 'n', not nx/ny/nz.")
        # the cap on n itself:
        content = self._gen("thermo", "thermo_3d", {"n": 50})
        n_elems = len(yaml.safe_load(content)["THERMO ELEMENTS"])
        self.assertLessEqual(n_elems, 8 ** 3, "'n' cap (<=8) is gone.")

    def test_params_override_defaults(self) -> None:
        """params must flow through the routing lambdas."""
        content = self._gen("thermo", "thermo_2d", {"nx": 4, "ny": 4})
        parsed = yaml.safe_load(content)
        self.assertEqual(len(parsed["THERMO ELEMENTS"]), 16)

        content = self._gen("thermo", "thermo_2d", {"kappa": 2.5})
        self.assertRegex(content, r"constant:\s*\[2\.5\]")

        content = self._gen("thermo", "thermo_3d", {"n": 3})
        parsed = yaml.safe_load(content)
        self.assertEqual(len(parsed["THERMO ELEMENTS"]), 27)

        content = self._gen("thermo", "thermo_3d",
                            {"T_end": 0.01, "dt": 0.001})
        self.assertRegex(content, r"NUMSTEP:\s*10\b")


if __name__ == "__main__":
    unittest.main()
