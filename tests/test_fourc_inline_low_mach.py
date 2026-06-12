"""Regression: fourc inline-mesh routing for low_mach/heated_channel_2d.

Created 2026-06-12. The probe sweep found generate_input('low_mach',
'heated_channel_2d', ...) falling through to the generator template in
src/backends/fourc/generators/low_mach.py, which emits literal
<placeholder> scalars and references an external Exodus mesh — 4C
aborts in MatchTree. The row is now routed to
matched_low_mach_heated_channel_input (src/backends/fourc/
inline_mesh.py): a self-contained 2D heated channel with
PROBLEMTYPE Low_Mach_Number_Flow, FLUID QUAD4 (PHYSICAL_TYPE Loma,
MAT_sutherland) coupled to a cloned TRANSP temperature field via
CLONING MATERIAL MAP — section combination mirrors the working corpus
case loma_2d_heated_chan_30x100.4C.yaml with self-contained UMFPACK
solvers.

Verified live against the real 4C binary 2026-06-12: rc=0 in ~1 s,
5 time steps, fluid Newton and scatra nonlinear loops fully converged,
VTU output written per step.

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

# Params the 2026-06-12 probe sweep passes to every row — generation
# must accept them (using what applies, ignoring the rest).
PROBE_PARAMS = {
    "kappa": 1.0, "nx": 16, "ny": 16, "nz": 16,
    "E": 1000.0, "nu": 0.3, "rho": 1.0, "Re": 100, "mu": 1.0,
    "refinements": 4, "T_end": 0.01, "dt": 0.001,
}


class TestFourcInlineLowMach(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def _gen(self, params: dict | None = None) -> str:
        return self.backend.generate_input(
            "low_mach", "heated_channel_2d", params or {})

    def test_no_placeholders(self) -> None:
        """A literal <...> scalar in the output is exactly the bug
        that made 4C MPI_Abort in MatchTree (probe 2026-06-12)."""
        for params in ({}, PROBE_PARAMS):
            with self.subTest(params=bool(params)):
                content = self._gen(params)
                hits = _PLACEHOLDER.findall(content)
                self.assertFalse(
                    hits,
                    f"low_mach/heated_channel_2d emits un-substituted "
                    f"placeholders {hits[:5]} — it regressed from the "
                    f"inline-mesh route back to the raw generator "
                    f"template.")

    def test_self_contained_inline_mesh(self) -> None:
        """Inline-mesh inputs embed NODE COORDS — no external mesh
        file reference that the user's work dir won't have."""
        content = self._gen()
        self.assertIn("NODE COORDS", content,
                      "low_mach/heated_channel_2d lost its inline mesh.")
        self.assertNotIn("FILE:", content,
                         "low_mach/heated_channel_2d references an "
                         "external mesh file — not self-contained.")

    def test_is_real_low_mach_physics(self) -> None:
        """The routed input must be genuine Loma physics: variable-
        density fluid coupled to a cloned scalar (temperature) field."""
        content = self._gen()
        self.assertIn('PROBLEMTYPE: "Low_Mach_Number_Flow"', content)
        self.assertIn('PHYSICAL_TYPE: "Loma"', content)
        self.assertIn("MAT_sutherland", content,
                      "Loma needs a temperature-dependent fluid "
                      "material (MAT_sutherland).")
        self.assertIn("CLONING MATERIAL MAP", content,
                      "the temperature field is a TRANSP discretization "
                      "cloned from the fluid — without the cloning map "
                      "there is no scatra field.")
        self.assertIn("FLUID QUAD4", content)

    def test_params_override_defaults(self) -> None:
        """params must flow through the routing lambda."""
        content = self._gen({"numstep": 7, "T_wall": 333.25})
        self.assertRegex(content, r"NUMSTEP:\s*7\b")
        self.assertIn("333.25", content)
        # probe's nx/ny are honored (nx=16 -> 16 elements per row;
        # with ny=16 the mesh has 17x17 = 289 nodes)
        content = self._gen(PROBE_PARAMS)
        self.assertIn("16x16", content)


if __name__ == "__main__":
    unittest.main()
