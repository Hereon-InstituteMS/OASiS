"""Regression: fourc inline-mesh routing for porous_media/single_phase_3d.

Created 2026-06-12. The probe sweep found the porous_media generator
template aborting in 4C's input matcher: it emitted the porofluid mesh
into a "TRANSPORT ELEMENTS" section with element lines carrying a
"TYPE PoroFluidMultiPhase" suffix, and it lacked the CLONING MATERIAL
MAP (porofluid -> structure / MAT_StructPoro) that the
porofluid_pressure_based problem type requires even for a rigid
skeleton.  The row is now routed through
matched_porofluid_single_phase_3d_input (inline_mesh.py): corpus-exact
"FLUID ELEMENTS" / "POROFLUIDMULTIPHASE HEX8 ... MAT 1" lines, full
single-phase material hierarchy, and pressure Dirichlet BCs on the
x=0 / x=1 faces of a unit cube.

Verified live against the real 4C binary 2026-06-12: rc=0 in < 1 s
(10 one-step-theta steps, n=4 hex8 cube).

These tests are GEN-ONLY (no 4C binary needed) so they run in CI.
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

# The probe's parameter soup — nz=16 must NOT inflate the 3D mesh.
PROBE_PARAMS = {
    "kappa": 1.0, "nx": 16, "ny": 16, "nz": 16,
    "E": 1000.0, "nu": 0.3, "rho": 1.0, "Re": 100, "mu": 1.0,
    "refinements": 4, "T_end": 0.01, "dt": 0.001,
}


class TestFourcInlinePorousMedia(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def _gen(self, params: dict | None = None) -> str:
        return self.backend.generate_input(
            "porous_media", "single_phase_3d", params or {})

    def test_no_placeholders(self) -> None:
        content = self._gen(dict(PROBE_PARAMS))
        hits = _PLACEHOLDER.findall(content)
        self.assertFalse(
            hits,
            f"porous_media/single_phase_3d emits un-substituted "
            f"placeholders {hits[:5]} — it regressed from the "
            f"inline-mesh route back to a raw generator template.")

    def test_self_contained_inline_mesh(self) -> None:
        content = self._gen(dict(PROBE_PARAMS))
        self.assertIn("NODE COORDS", content,
                      "row lost its inline mesh.")
        self.assertNotIn("FILE:", content,
                         "row references an external mesh file — "
                         "not self-contained.")

    def test_porofluid_physics_sections(self) -> None:
        content = self._gen(dict(PROBE_PARAMS))
        self.assertIn('PROBLEMTYPE: "porofluid_pressure_based"', content)
        self.assertIn("MAT_FluidPoroSinglePhase", content)
        # Corpus element convention: FLUID ELEMENTS with
        # POROFLUIDMULTIPHASE HEX8 lines and NO "TYPE ..." suffix
        # (the old template's TYPE suffix is what 4C aborted on).
        self.assertIn("FLUID ELEMENTS", content)
        self.assertIn("POROFLUIDMULTIPHASE HEX8", content)
        self.assertNotIn("TYPE PoroFluidMultiPhase", content)
        # porofluid_pressure_based requires a material clone onto the
        # (rigid) structure dis — missing map is a hard 4C error.
        self.assertIn("CLONING MATERIAL MAP", content)
        self.assertIn("MAT_StructPoro", content)

    def test_probe_nz_does_not_inflate_mesh(self) -> None:
        """Resolution is driven by 'n', never by the probe's nz=16."""
        content = self._gen(dict(PROBE_PARAMS))
        n_elems = len(re.findall(r"POROFLUIDMULTIPHASE HEX8", content))
        self.assertLessEqual(
            n_elems, 8 ** 3,
            f"{n_elems} elements — probe's nx/ny/nz=16 leaked into "
            f"the 3D mesh size.")

    def test_params_override_defaults(self) -> None:
        """params must flow through the routing lambda."""
        content = self._gen({"kappa": 123.5})
        self.assertRegex(content, r"PERMEABILITY:\s*123\.5\b")
        content = self._gen({"n": 3})
        n_elems = len(re.findall(r"POROFLUIDMULTIPHASE HEX8", content))
        self.assertEqual(n_elems, 27)


if __name__ == "__main__":
    unittest.main()
