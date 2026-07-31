"""Gen-only tests for the Kratos curved-geometry MMS family (annulus, Gmsh-meshed).

These tests exercise ONLY the generator layer — no KratosMultiphysics import
happens at test time (the pip Kratos in the repo .venv is GLIBC-broken; the
runnable install lives under /usr/bin/python3, see KNOWLEDGE['curved_mms']).

Covered:
  * registry surfaces curved_mms_annulus_2d in GENERATORS and curved_mms in KNOWLEDGE
  * backend surfaces the physics via supported_physics / generate_input / get_knowledge
  * generated script compiles, honors parameters, has no unresolved placeholders,
    prints the machine-readable L2_ERROR line, and passes backend.validate_input
  * module-level validate_parameters accepts good params and rejects bad ones

Live convergence evidence (run outside pytest, /usr/bin/python3, 2026-08-01):
h=0.2/0.1/0.05/0.025 gave L2 = 6.363e-2/1.674e-2/4.226e-3/1.063e-3,
observed orders 1.93/1.99/1.99 (theoretical: 2).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "data"))

# Dev-draw parameters (deliberately distinctive, NOT the template defaults)
GOOD_PARAMS = {
    "r_inner": 0.37,
    "r_outer": 1.93,
    "coeff_a": 0.7,
    "coeff_b": -1.3,
    "coeff_c": 0.85,
    "coeff_d": 0.41,
    "mode_m": 4,
    "radial_p": 2.5,
    "conductivity": 3.25,
    "mesh_size": 0.07,
}


class TestCurvedMMSRegistry(unittest.TestCase):
    def test_generator_and_knowledge_registered(self) -> None:
        from backends.kratos.generators import GENERATORS, KNOWLEDGE
        self.assertIn("curved_mms_annulus_2d", GENERATORS)
        self.assertIn("curved_mms", KNOWLEDGE)
        kn = KNOWLEDGE["curved_mms"]
        self.assertTrue(kn.get("description"))
        self.assertTrue(kn.get("pitfalls"))
        self.assertEqual(kn.get("theoretical_l2_order"), 2.0)
        # The verified environment facts must be recorded
        joined = " ".join(kn["pitfalls"])
        self.assertIn("GLIBC", joined)
        self.assertIn("/usr/bin/python3", joined)

    def test_backend_surfaces_physics(self) -> None:
        from backends.kratos.backend import KratosBackend
        backend = KratosBackend()
        caps = {c.name: c for c in backend.supported_physics()}
        self.assertIn("curved_mms", caps)
        self.assertIn("annulus_2d", caps["curved_mms"].template_variants)
        self.assertTrue(backend.get_knowledge("curved_mms"))
        script = backend.generate_input("curved_mms", "annulus_2d", dict(GOOD_PARAMS))
        self.assertIn("LaplacianElement2D3N", script)
        # Honesty guard of the backend must accept the script as a real solve
        self.assertEqual(backend.validate_input(script), [])


class TestCurvedMMSGeneratedScript(unittest.TestCase):
    def _gen(self, params: dict) -> str:
        from backends.kratos.generators.curved_mms import _curved_mms_annulus_2d
        return _curved_mms_annulus_2d(params)

    def test_compiles_and_no_unresolved_placeholders(self) -> None:
        script = self._gen(dict(GOOD_PARAMS))
        compile(script, "<curved_mms>", "exec")  # must not raise
        for token in ("{r_i}", "{r_o}", "{A}", "{B}", "{C}", "{D}", "{m}",
                      "{pexp}", "{kappa}", "{h}", "{msh_file}"):
            self.assertNotIn(token, script, f"unresolved placeholder {token}")

    def test_defaults_work(self) -> None:
        script = self._gen({})
        compile(script, "<curved_mms>", "exec")
        self.assertIn("L2_ERROR", script)

    def test_parameters_are_honored(self) -> None:
        script = self._gen(dict(GOOD_PARAMS))
        self.assertIn("R_INNER = 0.37", script)
        self.assertIn("R_OUTER = 1.93", script)
        self.assertIn("COEFF_A = 0.7", script)
        self.assertIn("COEFF_B = -1.3", script)
        self.assertIn("COEFF_C = 0.85", script)
        self.assertIn("COEFF_D = 0.41", script)
        self.assertIn("MODE_M = 4", script)
        self.assertIn("RADIAL_P = 2.5", script)
        self.assertIn("KAPPA = 3.25", script)
        self.assertIn("MESH_SIZE = 0.07", script)

    def test_msh_file_route_is_honored(self) -> None:
        params = dict(GOOD_PARAMS, msh_file="/some/agent/mesh.msh")
        script = self._gen(params)
        self.assertIn('MSH_FILE = r"/some/agent/mesh.msh"', script)
        self.assertIn("gmsh.open(MSH_FILE)", script)

    def test_machine_readable_output_and_real_solve(self) -> None:
        script = self._gen(dict(GOOD_PARAMS))
        self.assertIn('L2_ERROR = {l2_error:.12e}', script)  # machine-readable line
        self.assertIn("import gmsh", script)
        self.assertIn("import KratosMultiphysics", script)
        self.assertIn("ConvectionDiffusionApplication", script)
        self.assertIn("ResidualBasedLinearStrategy", script)
        self.assertIn("strategy.Solve()", script)
        self.assertIn("results_summary.json", script)
        # exact manufactured source (sympy-verified): f = -kappa*D*p^2*r^(p-2)
        self.assertIn("-KAPPA * COEFF_D * RADIAL_P**2 * r**(RADIAL_P - 2.0)", script)

    def test_bad_params_raise(self) -> None:
        with self.assertRaises(ValueError):
            self._gen({"r_inner": -1.0})


class TestValidateParameters(unittest.TestCase):
    def _validate(self, params: dict) -> list[str]:
        from backends.kratos.generators.curved_mms import validate_parameters
        return validate_parameters(params)

    def test_good_params_pass(self) -> None:
        self.assertEqual(self._validate(dict(GOOD_PARAMS)), [])
        self.assertEqual(self._validate({}), [])  # defaults are valid

    def test_bad_radii(self) -> None:
        self.assertTrue(any("r_inner" in e for e in self._validate({"r_inner": 0.0})))
        self.assertTrue(any("r_inner" in e for e in self._validate({"r_inner": -0.5})))
        self.assertTrue(any("r_outer" in e for e in
                            self._validate({"r_inner": 1.0, "r_outer": 0.5})))
        self.assertTrue(any("r_outer" in e for e in
                            self._validate({"r_inner": 1.0, "r_outer": 1.0})))

    def test_bad_mode_number(self) -> None:
        self.assertTrue(any("mode_m" in e for e in self._validate({"mode_m": 0})))
        self.assertTrue(any("mode_m" in e for e in self._validate({"mode_m": -2})))
        self.assertTrue(any("mode_m" in e for e in self._validate({"mode_m": 2.5})))
        self.assertTrue(any("mode_m" in e for e in self._validate({"mode_m": True})))

    def test_non_numeric_coefficients(self) -> None:
        for key in ("coeff_a", "coeff_b", "coeff_c", "coeff_d", "radial_p",
                    "conductivity", "mesh_size", "r_inner", "r_outer"):
            errs = self._validate({key: "not-a-number"})
            self.assertTrue(any(key in e for e in errs), f"{key} not rejected")

    def test_bad_conductivity_and_mesh_size(self) -> None:
        self.assertTrue(any("conductivity" in e for e in self._validate({"conductivity": 0.0})))
        self.assertTrue(any("mesh_size" in e for e in self._validate({"mesh_size": -0.1})))

    def test_bad_msh_file(self) -> None:
        self.assertTrue(any("msh_file" in e for e in self._validate({"msh_file": 42})))


if __name__ == "__main__":
    unittest.main()
