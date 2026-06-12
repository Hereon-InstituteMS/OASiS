"""Regression: fourc inline-mesh routing for the beams catalog rows.

Created 2026-06-12. The probe sweep found beams/cantilever_static and
beams/cantilever_dynamic aborting in 4C's MatchTree because
generate_input() fell through to the BeamsGenerator templates carrying
literal <placeholder> scalars (`YOUNG: <Young_modulus>`,
`TIMESTEP: <load_step_size>`) — unrunnable as emitted. Both rows are
now routed to corpus-matched self-contained inline-mesh inputs:

    beams/cantilever_static  -> matched_beam_cantilever_static_input
        (10 BEAM3R LINE2 on [0,10], clamped at x=0, tip force F_z
         ramped over 5 Statics load steps, TangDis predictor; load
         scales with E so the probe's E=1000 override converges)
    beams/cantilever_dynamic -> matched_beam_cantilever_dynamic_input
        (10 BEAM3R LINE3 + HERMITE_CENTERLINE on [0,10],
         GenAlphaLieGroup / MASSLIN rotations / RHO_INF, tip bending
         moment M_y ramped via FUNCT1 over 5 steps)

Both verified live against the 4C binary 2026-06-12: rc=0 (all steps
finalised, beam VTU output) with both the standard probe parameter
dict and empty params.

These tests are GEN-ONLY (no 4C binary needed) so they run in CI:
they pin that generate_input for each row returns self-contained YAML
with zero un-substituted placeholders, an inline beam mesh, and the
shipped physics (BEAM3R + MAT_BeamReissnerElastHyper; Statics for the
static row, GenAlphaLieGroup with NUMSTEP > 1 for the dynamic row).
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
# <load_step_size> but NOT YAML flow content or comparison operators.
_PLACEHOLDER = re.compile(r"<[A-Za-z_][A-Za-z0-9_. ]*>")

ROUTED_ROWS = [
    ("beams", "cantilever_static"),
    ("beams", "cantilever_dynamic"),
]


class TestFourcInlineBeams(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def _gen(self, physics: str, variant: str, params: dict | None = None) -> str:
        return self.backend.generate_input(physics, variant, params or {})

    def test_routed_rows_have_no_placeholders(self) -> None:
        """Both beams rows must emit fully-substituted YAML — a
        literal <...> scalar in the output is exactly the bug that
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
        """Beam inputs MUST embed the mesh inline (NODE COORDS) —
        4C beam elements cannot be read from Exodus files at all."""
        for physics, variant in ROUTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant)
                self.assertIn(
                    "NODE COORDS", content,
                    f"{physics}/{variant} lost its inline mesh.")
                self.assertNotIn(
                    "FILE:", content,
                    f"{physics}/{variant} references an external "
                    f"mesh file — beams cannot use Exodus meshes.")

    def test_rows_ship_reissner_beam_physics(self) -> None:
        """Both rows must keep the shipped beam formulation: BEAM3R
        elements with the hyperelastic Reissner beam material."""
        for physics, variant in ROUTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant)
                self.assertIn("BEAM3R", content)
                self.assertIn("MAT_BeamReissnerElastHyper", content)
                self.assertIn("STRUCTURE ELEMENTS", content)
                self.assertIn("TRIADS", content)

    def test_static_is_statics_line2(self) -> None:
        content = self._gen("beams", "cantilever_static")
        self.assertRegex(content, r'DYNAMICTYPE:\s*"?Statics"?')
        self.assertIn("BEAM3R LINE2", content)
        self.assertNotIn("GenAlphaLieGroup", content)

    def test_dynamic_is_genalpha_liegroup_with_steps(self) -> None:
        content = self._gen("beams", "cantilever_dynamic")
        self.assertIn("GenAlphaLieGroup", content)
        self.assertIn("BEAM3R LINE3", content)
        self.assertIn("HERMITE_CENTERLINE true", content)
        self.assertRegex(content, r'MASSLIN:\s*"?rotations"?')
        self.assertIn("RHO_INF", content)
        m = re.search(r"NUMSTEP:\s*(\d+)", content)
        self.assertIsNotNone(m)
        self.assertGreater(int(m.group(1)), 1,
                           "dynamic run needs NUMSTEP > 1")

    def test_params_override_defaults(self) -> None:
        """params must flow through the routing lambdas."""
        content = self._gen("beams", "cantilever_static",
                            {"E": 1234.5})
        self.assertRegex(content, r"YOUNG:\s*1234\.5\b")
        content = self._gen("beams", "cantilever_dynamic",
                            {"numstep": 7, "E": 1234.5})
        self.assertRegex(content, r"NUMSTEP:\s*7\b")
        self.assertRegex(content, r"YOUNG:\s*1234\.5\b")

    def test_probe_params_are_tolerated(self) -> None:
        """The standard probe dict must generate cleanly (the live
        rc=0 gate ran with exactly these params 2026-06-12)."""
        probe = {"kappa": 1.0, "nx": 16, "ny": 16, "nz": 16,
                 "E": 1000.0, "nu": 0.3, "rho": 1.0, "Re": 100,
                 "mu": 1.0, "refinements": 4, "T_end": 0.01,
                 "dt": 0.001}
        for physics, variant in ROUTED_ROWS:
            with self.subTest(row=f"{physics}/{variant}"):
                content = self._gen(physics, variant, dict(probe))
                self.assertFalse(_PLACEHOLDER.findall(content))
                self.assertIn("NODE COORDS", content)


if __name__ == "__main__":
    unittest.main()
