"""Gen-only regression: fourc thermo_transient_mms/temporal_mms_2d.

Unsteady-heat MMS extension family on 4C graded on TEMPORAL
convergence order: manufactured solution
u*(x,y,t) = temp_offset + (amp*sin(kx x)*sin(ky y) + grad_amp*x/lx)*f(t)
on a FIXED QUAD4 mesh, exact volumetric source rho*c*du*/dt -
kappa*lap(u*) as a SYMBOLIC_FUNCTION_OF_SPACE_TIME body load,
time-dependent Dirichlet u* on the whole boundary, initial field
u*(x,0), One-Step-Theta time integration.

The live gate is a dt-halving study against a built 4C binary: the
Richardson orders ||u_dt - u_dt/2|| must approach the theoretical
One-Step-Theta rates (2 at theta=0.5, 1 at theta=1.0). The error
against the exact solution saturates at the spatial Q1 floor, so it is
Richardson -- not error-vs-exact -- that grades the time integrator.
The measured orders are outputs of that run and are deliberately not
recorded here.

The load-bearing 4C fact these decks encode: the volumetric source in
STANDALONE Thermo must use the PLAIN "DESIGN SURF NEUMANN CONDITIONS"
(2D domain condition = the elements themselves). The THERMO-prefixed
"DESIGN SURF/VOL THERMO NEUMANN CONDITIONS" are registered under
"ThermoSurfaceNeumann"/"ThermoVolumeNeumann" and are SILENTLY dropped
by both Discretization::evaluate_neumann and TemperImpl::radiation()
(probed live: rc=0, no warning, field converged to the source-free
solution). These tests pin the plain section so the deck never
regresses to the silently-ignored variant.

Tests are GEN-ONLY (no 4C binary needed) so they run in CI; the live
execution gate is the separate dt-halving study.
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

_PLACEHOLDER = re.compile(r"<[A-Za-z_][A-Za-z0-9_. ]*>")

PHYSICS = "thermo_transient_mms"
VARIANT = "temporal_mms_2d"

# Probe-style parameter set the routing lambda must tolerate.
PROBE_PARAMS = {
    "kappa": 1.0, "rho": 1.0, "c": 1.0, "n": 16,
    "theta": 0.5, "dt": 0.02, "T_end": 0.4,
    "nx": 16, "ny": 16, "nz": 16, "E": 1000.0, "nu": 0.3,
}


class TestFourcThermoTransientMMS(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")
        from backends.fourc.generators import get_generator
        cls.gen = get_generator(PHYSICS)

    def _gen(self, params: dict | None = None) -> str:
        return self.backend.generate_input(PHYSICS, VARIANT,
                                           dict(params or PROBE_PARAMS))

    # ── deck validity ──────────────────────────────────────────────

    def test_row_is_yaml_dict_and_validates(self) -> None:
        content = self._gen()
        parsed = yaml.safe_load(content)
        self.assertIsInstance(parsed, dict,
                              "temporal_mms_2d does not emit a YAML "
                              "dictionary.")
        self.assertEqual(self.backend.validate_input(content), [],
                         "temporal_mms_2d fails validate_input().")

    def test_row_has_no_placeholders(self) -> None:
        content = self._gen()
        hits = _PLACEHOLDER.findall(content)
        self.assertFalse(hits, f"un-substituted placeholders {hits[:5]}")

    def test_row_is_self_contained(self) -> None:
        content = self._gen()
        self.assertIn("NODE COORDS", content, "lost its inline mesh")
        self.assertNotIn("FILE:", content,
                         "references an external mesh file")

    def test_row_is_genuine_transient_thermo_mms(self) -> None:
        content = self._gen()
        parsed = yaml.safe_load(content)
        self.assertEqual(parsed["PROBLEM TYPE"]["PROBLEMTYPE"], "Thermo")
        self.assertEqual(parsed["THERMAL DYNAMIC"]["DYNAMICTYPE"],
                         "OneStepTheta")
        self.assertIn("THERMO QUAD4", content)
        self.assertIn("MAT_Fourier", content)
        # MMS ingredients: exact solution (FUNCT1), source (FUNCT2),
        # initial field by function, whole-boundary Dirichlet driven
        # by FUNCT 1.
        self.assertEqual(parsed["THERMAL DYNAMIC"]["INITIALFIELD"],
                         "field_by_function")
        self.assertEqual(parsed["THERMAL DYNAMIC"]["INITFUNCNO"], 1)
        self.assertIn("FUNCT1", parsed)
        self.assertIn("FUNCT2", parsed)
        for fid in ("FUNCT1", "FUNCT2"):
            self.assertIn("SYMBOLIC_FUNCTION_OF_SPACE_TIME",
                          parsed[fid][0])
        dirich = parsed["DESIGN LINE DIRICH CONDITIONS"][0]
        self.assertEqual(dirich["FUNCT"], [1],
                         "Dirichlet no longer driven by the exact-"
                         "solution space-time function.")

    def test_source_uses_plain_neumann_section(self) -> None:
        """THE 4C trap this family exists to encode: the volumetric
        source must go through the PLAIN Neumann section. The
        THERMO-prefixed one is silently ignored in standalone Thermo
        (verified live: rc=0, no warning, and a field that converged to
        the SOURCE-FREE solution until the plain section was used)."""
        content = self._gen()
        parsed = yaml.safe_load(content)
        self.assertIn("DESIGN SURF NEUMANN CONDITIONS", parsed)
        self.assertNotIn("DESIGN SURF THERMO NEUMANN CONDITIONS", parsed)
        neumann = parsed["DESIGN SURF NEUMANN CONDITIONS"][0]
        self.assertEqual(neumann["FUNCT"], [2],
                         "source no longer wired to FUNCT2")
        # the domain condition must cover every node (body load)
        n_nodes = len(parsed["NODE COORDS"])
        self.assertEqual(len(parsed["DSURF-NODE TOPOLOGY"]), n_nodes,
                         "DSURF body-load condition no longer covers "
                         "the whole domain.")

    # ── parameters honored ─────────────────────────────────────────

    def test_params_override_defaults(self) -> None:
        content = self._gen({"n": 4, "dt": 0.01, "T_end": 0.05,
                             "theta": 1.0, "kappa": 2.5, "rho": 2.0,
                             "c": 3.0})
        parsed = yaml.safe_load(content)
        self.assertEqual(len(parsed["THERMO ELEMENTS"]), 16)
        self.assertAlmostEqual(
            parsed["THERMAL DYNAMIC"]["TIMESTEP"], 0.01)
        self.assertEqual(parsed["THERMAL DYNAMIC"]["NUMSTEP"], 5)
        self.assertAlmostEqual(
            parsed["THERMAL DYNAMIC"]["MAXTIME"], 0.05)
        self.assertAlmostEqual(
            parsed["THERMAL DYNAMIC/ONESTEPTHETA"]["THETA"], 1.0)
        self.assertRegex(content, r"constant:\s*\[2\.5\]")
        # CAPA = rho * c
        self.assertAlmostEqual(
            parsed["MATERIALS"][0]["MAT_Fourier"]["CAPA"], 6.0)

    def test_dt_halving_same_t_end(self) -> None:
        """The temporal-convergence workflow: halving dt doubles
        NUMSTEP while MAXTIME stays fixed."""
        for dt, steps in ((0.04, 10), (0.02, 20), (0.01, 40),
                          (0.005, 80)):
            parsed = yaml.safe_load(
                self._gen({"dt": dt, "T_end": 0.4}))
            self.assertEqual(parsed["THERMAL DYNAMIC"]["NUMSTEP"],
                             steps, f"dt={dt}")
            self.assertAlmostEqual(
                parsed["THERMAL DYNAMIC"]["MAXTIME"], 0.4,
                msg=f"dt={dt}: dt levels end at different times")

    def test_mesh_cap_and_probe_nx_ignored(self) -> None:
        """Resolution is keyed off 'n' (cap 96); generic sweep params
        nx/ny/nz must not inflate the mesh."""
        parsed = yaml.safe_load(
            self._gen({"n": 8, "nx": 64, "ny": 64, "nz": 64}))
        self.assertEqual(len(parsed["THERMO ELEMENTS"]), 64)
        parsed = yaml.safe_load(self._gen({"n": 500}))
        self.assertLessEqual(len(parsed["THERMO ELEMENTS"]), 96 ** 2)

    def test_time_profile_exp(self) -> None:
        content = self._gen({"time_profile": "exp", "omega": 2.0,
                             "dt": 0.02, "T_end": 0.4})
        self.assertIn("exp(-2*t)", content)
        self.assertNotIn("cos(", content.split("FUNCT1")[1])

    # ── validate_parameters coverage ───────────────────────────────

    def test_validate_parameters_accepts_good(self) -> None:
        self.assertEqual(
            self.gen.validate_parameters(
                {"theta": 0.5, "dt": 0.02, "T_end": 0.4, "kappa": 1.0,
                 "rho": 1.0, "c": 1.0, "n": 48, "mode_x": 1,
                 "mode_y": 2, "omega": 3.0, "time_profile": "cos",
                 "amp": 1.0, "grad_amp": 0.5, "lx": 1.0, "ly": 1.0}),
            [])

    def test_validate_parameters_rejects_bad_theta(self) -> None:
        for theta in (0.0, -0.5, 1.5):
            issues = self.gen.validate_parameters({"theta": theta})
            self.assertTrue(any("theta" in i for i in issues),
                            f"theta={theta} not rejected")
        # theta = 1.0 (backward Euler) is legal
        self.assertEqual(self.gen.validate_parameters({"theta": 1.0}),
                         [])

    def test_validate_parameters_rejects_bad_timestep(self) -> None:
        self.assertTrue(any(
            "dt" in i for i in
            self.gen.validate_parameters({"dt": 0.0})))
        self.assertTrue(any(
            "dt" in i for i in
            self.gen.validate_parameters({"dt": -0.01})))
        self.assertTrue(any(
            "T_end" in i for i in
            self.gen.validate_parameters({"dt": 0.5, "T_end": 0.1})))
        # non-integer T_end/dt breaks same-final-time comparisons
        self.assertTrue(any(
            "integer multiple" in i for i in
            self.gen.validate_parameters({"dt": 0.03, "T_end": 0.4})))

    def test_validate_parameters_rejects_bad_material(self) -> None:
        for key in ("kappa", "rho", "c"):
            for val in (0.0, -1.0):
                issues = self.gen.validate_parameters({key: val})
                self.assertTrue(any(key in i for i in issues),
                                f"{key}={val} not rejected")

    def test_validate_parameters_rejects_degenerate_mms(self) -> None:
        self.assertTrue(any(
            "degenerates" in i for i in
            self.gen.validate_parameters({"amp": 0.0, "grad_amp": 0.0})))
        self.assertTrue(any(
            "omega" in i for i in
            self.gen.validate_parameters({"omega": 0.0})))
        self.assertTrue(any(
            "time_profile" in i for i in
            self.gen.validate_parameters({"time_profile": "tan"})))
        self.assertTrue(any(
            "mode_x" in i for i in
            self.gen.validate_parameters({"mode_x": 0})))
        self.assertTrue(any(
            "n=" in i for i in
            self.gen.validate_parameters({"n": 1})))
        self.assertTrue(any(
            "lx" in i for i in
            self.gen.validate_parameters({"lx": -1.0})))


if __name__ == "__main__":
    unittest.main()
