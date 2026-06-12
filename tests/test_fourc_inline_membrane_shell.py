"""Regression: fourc inline-mesh routing for membrane + shell rows.

Created 2026-06-12. Both rows previously fell through to a one-line
*comment* template in their generators (generators/membrane.py and
generators/shell.py returned "# Membrane template ..." / "# Shell
template ..."). A bare comment is not a YAML mapping, so
validate_input() failed with the literal message

    "Input is not a YAML dictionary"

before the run stage was ever reached — the probe never executed 4C.

The fix routes both rows to NEW self-contained inline-mesh structural
inputs in backends/fourc/inline_mesh.py:

    membrane/membrane_2d -> matched_membrane_2d_input
        flat MEMBRANE4 QUAD4 patch under a prescribed uniaxial stretch
        (membranes have zero bending stiffness, so the corpus stabilises
        them by prescribing the full nodal displacement field via
        Dirichlet — no free DOF is left to buckle / go singular).
        Material MAT_Membrane_ElastHyper + ELAST_IsoNeoHooke.

    shell/shell_3d -> matched_shell_3d_input
        flat SHELL7P QUAD4 clamped cantilever under a transverse
        orthopressure. Material MAT_ElastHyper + ELAST_CoupNeoHooke.

Both verified live against the 4C binary 2026-06-12: rc=0, all 10
time steps finalised, VTU output written (membrane wct ~1 s, shell
~6 s).

These tests are GEN-ONLY (no 4C binary needed) so they run in CI:
they pin that generate_input for each row returns self-contained YAML
that parses as a dict, validates clean (the original regression), has
no un-substituted placeholders, an inline mesh (NODE COORDS, no FILE:),
the right physics tokens (membrane element + membrane material; shell
element), and that probe params flow through the routing lambdas.
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

# Full probe parameter set the routing lambdas must tolerate.
PROBE_PARAMS = {
    "kappa": 1.0, "nx": 16, "ny": 16, "nz": 16,
    "E": 1000.0, "nu": 0.3, "rho": 1.0, "Re": 100, "mu": 1.0,
    "refinements": 4, "T_end": 0.01, "dt": 0.001,
}

ROWS = [
    ("membrane", "membrane_2d"),
    ("shell", "shell_3d"),
]


class TestFourcInlineMembraneShell(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def _gen(self, physics: str, variant: str,
             params: dict | None = None) -> str:
        return self.backend.generate_input(
            physics, variant, params if params is not None else {})

    def test_rows_parse_as_yaml_dict_and_validate_clean(self) -> None:
        """The original regression: a one-line comment template made
        validate_input return ["Input is not a YAML dictionary"]. Each
        routed row must now parse as a YAML mapping AND validate clean
        ([]) under the full probe params."""
        import yaml
        for physics, variant in ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant, PROBE_PARAMS)
                data = yaml.safe_load(content)
                self.assertIsInstance(
                    data, dict,
                    f"{physics}/{variant} is not a YAML dictionary — "
                    f"the exact original failure.")
                errors = self.backend.validate_input(content)
                self.assertEqual(
                    errors, [],
                    f"{physics}/{variant} validate_input returned "
                    f"{errors}; expected [].")

    def test_rows_have_no_placeholders(self) -> None:
        for physics, variant in ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant, PROBE_PARAMS)
                hits = _PLACEHOLDER.findall(content)
                self.assertFalse(
                    hits,
                    f"{physics}/{variant} emits un-substituted "
                    f"placeholders {hits[:5]}.")

    def test_rows_are_self_contained(self) -> None:
        """Inline-mesh inputs embed NODE COORDS — no external FILE:."""
        for physics, variant in ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant, PROBE_PARAMS)
                self.assertIn(
                    "NODE COORDS", content,
                    f"{physics}/{variant} lost its inline mesh.")
                self.assertNotIn(
                    "FILE:", content,
                    f"{physics}/{variant} references an external mesh.")

    def test_membrane_physics(self) -> None:
        """Membrane row uses the MEMBRANE element token AND a genuine
        membrane material (NOT a plain solid)."""
        content = self._gen("membrane", "membrane_2d", PROBE_PARAMS)
        self.assertIn("MEMBRANE4 QUAD4", content,
                      "membrane row must use MEMBRANE4 QUAD4 elements.")
        self.assertIn("MAT_Membrane_ElastHyper", content,
                      "membrane row must use a membrane material.")
        self.assertIn("plane_stress", content,
                      "membranes carry plane-stress only.")
        # Membranes are surface elements in 3D ambient space — forcing
        # PROBLEM SIZE DIM: 2 makes 4C SIGFPE in initialize_elements().
        self.assertNotIn("PROBLEM SIZE", content,
                         "membrane input must NOT pin PROBLEM SIZE DIM.")

    def test_shell_physics(self) -> None:
        """Shell row uses the SHELL7P element token + a shell material."""
        content = self._gen("shell", "shell_3d", PROBE_PARAMS)
        self.assertIn("SHELL7P QUAD4", content,
                      "shell row must use SHELL7P QUAD4 elements.")
        self.assertIn("USE_ANS true", content,
                      "shell element suffix must carry the ANS flag "
                      "copied from the corpus element line.")
        self.assertIn("MAT_ElastHyper", content,
                      "shell row must use a hyperelastic shell material.")

    def test_params_override_defaults(self) -> None:
        """E from params must flow through the routing lambdas."""
        m = self.backend.generate_input(
            "membrane", "membrane_2d", {"E": 2222.0})
        # membrane material MUE = E/(2*(1+nu)) = 2222/2.6 ~ 854.6
        self.assertRegex(m, r"constant:\s*854\.",
                         "membrane MUE must derive from the E param.")
        s = self.backend.generate_input(
            "shell", "shell_3d", {"E": 3333.0})
        self.assertIn("3333.0", s,
                      "shell YOUNG must derive from the E param.")

    def test_probe_params_do_not_inflate_mesh(self) -> None:
        """The probe passes nx=ny=16; the caps must keep the mesh
        small enough to run in well under 30 s. A QUAD4 grid of
        nx*ny elements: 16*16 = 256 elements is fine."""
        for physics, variant in ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant, PROBE_PARAMS)
                n_elems = content.count("QUAD4 ")
                self.assertLessEqual(
                    n_elems, 16 * 16 + 1,
                    f"{physics}/{variant} mesh inflated to {n_elems} "
                    f"elements under the probe params.")


if __name__ == "__main__":
    unittest.main()
