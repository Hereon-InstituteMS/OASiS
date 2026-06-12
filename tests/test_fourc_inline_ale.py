"""Regression: fourc inline-mesh routing for the ale/ale_2d catalog row.

Created 2026-06-12. The probe sweep found generate_input("ale",
"ale_2d", ...) falling through to the generator template in
src/backends/fourc/generators/ale.py, which emits literal
<placeholder> scalars and references an external Exodus mesh —
4C aborts in MatchTree. The row is now routed in
backend._generate_inline() to matched_ale_2d_input(): a
self-contained 2D ALE mesh-motion problem on [0,1]² (ALE2 QUAD4,
ALE_TYPE laplace_material, bottom edge fixed, top edge sheared by
a time-ramp FUNCT), mirroring the ale2d_laplace_*.4C.yaml
regression inputs.

Verified live against the 4C binary 2026-06-12: rc=0
("processor 0 finished normally", 10 linear solves, < 1 s).

These tests are GEN-ONLY (no 4C binary needed) so they run in CI:
they pin that generate_input for the row returns self-contained
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

# The exact params dict the probe sweep passes — the routing lambda
# must accept it gracefully (irrelevant keys ignored).
PROBE_PARAMS = {
    "kappa": 1.0, "nx": 16, "ny": 16, "nz": 16,
    "E": 1000.0, "nu": 0.3, "rho": 1.0, "Re": 100, "mu": 1.0,
    "refinements": 4, "T_end": 0.01, "dt": 0.001,
}


class TestFourcInlineAle(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def _gen(self, params: dict | None = None) -> str:
        return self.backend.generate_input("ale", "ale_2d",
                                           params or {})

    def test_no_placeholders(self) -> None:
        """A literal <...> scalar in the output is exactly the bug
        that made 4C MPI_Abort in MatchTree (probe 2026-06-12)."""
        for params in ({}, PROBE_PARAMS):
            with self.subTest(params=bool(params)):
                content = self._gen(params)
                hits = _PLACEHOLDER.findall(content)
                self.assertFalse(
                    hits,
                    f"ale/ale_2d emits un-substituted placeholders "
                    f"{hits[:5]} — it regressed from the inline-mesh "
                    f"route back to the raw generator template.")

    def test_self_contained_inline_mesh(self) -> None:
        """Inline-mesh inputs embed NODE COORDS — no external mesh
        file reference that the user's work dir won't have."""
        content = self._gen(PROBE_PARAMS)
        self.assertIn("NODE COORDS", content,
                      "ale/ale_2d lost its inline mesh.")
        self.assertNotIn("FILE:", content,
                         "ale/ale_2d references an external mesh "
                         "file — not self-contained.")

    def test_is_ale_mesh_motion_problem(self) -> None:
        """Physics pins: pure ALE problem with ALE2 elements and a
        time-dependent moving boundary."""
        content = self._gen(PROBE_PARAMS)
        self.assertIn('PROBLEMTYPE: "Ale"', content)
        self.assertIn("ALE DYNAMIC", content)
        self.assertIn("ALE2 QUAD4", content)
        self.assertIn("ALE ELEMENTS", content)
        # Moving boundary: a FUNCT1 with a time-dependent symbolic
        # function, referenced by a Dirichlet condition.
        self.assertIn("SYMBOLIC_FUNCTION_OF_SPACE_TIME", content)
        self.assertRegex(content, r"FUNCT: \[1, 0\]",
                         "no Dirichlet condition references the "
                         "time-ramp FUNCT — the boundary does not "
                         "move.")
        # Transient: more than one step.
        m = re.search(r"NUMSTEP: (\d+)", content)
        self.assertIsNotNone(m)
        self.assertGreater(int(m.group(1)), 1)

    def test_params_override_defaults(self) -> None:
        """params must flow through the routing lambda."""
        content = self._gen({"nx": 4, "ny": 4,
                             "T_end": 0.02, "dt": 0.005})
        self.assertEqual(content.count("ALE2 QUAD4"), 16,
                         "nx/ny did not reach the mesh generator.")
        self.assertRegex(content, r"NUMSTEP: 4\b",
                         "T_end/dt did not set the step count.")
        self.assertRegex(content, r"TIMESTEP: 0\.005\b")
        # Mesh size stays capped so the probe can't request a huge
        # grid.
        content = self._gen({"nx": 100, "ny": 100})
        self.assertEqual(content.count("ALE2 QUAD4"), 32 * 32,
                         "mesh cap (<= 32x32) regressed.")


if __name__ == "__main__":
    unittest.main()
