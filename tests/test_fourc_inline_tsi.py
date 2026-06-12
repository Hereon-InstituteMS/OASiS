"""Regression: fourc inline-mesh routing for tsi/monolithic_3d.

Created 2026-06-12. The probe sweep found the tsi/monolithic_3d
catalog row aborting in 4C's MatchTree because generate_input() fell
through to the generator template (src/backends/fourc/generators/
tsi.py) carrying literal <placeholder> scalars and an external Exodus
mesh reference. The row is now routed to
matched_tsi_monolithic_3d_input (src/backends/fourc/inline_mesh.py):
a self-contained SOLIDSCATRA HEX8 cube with genuinely MONOLITHIC
two-way thermo-structure coupling (COUPALGO tsi_monolithic, merged
TSI block matrix + direct UMFPACK solver, layout copied from the 4C
corpus example tsi_lincompression_monolithic_mergeTSImatrix.4C.yaml).

Verified live against the 4C binary 2026-06-12: rc=0 in ~6 s with the
probe params (nx=ny=nz=16 capped to 8^3), TSI::Monolithic::Evaluate in
the timing summary and 10 converged coupled steps.

These tests are GEN-ONLY (no 4C binary needed) so they run in CI.
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

# The exact params the template probe passes to every row.
PROBE_PARAMS = {
    "kappa": 1.0, "nx": 16, "ny": 16, "nz": 16,
    "E": 1000.0, "nu": 0.3, "rho": 1.0, "Re": 100, "mu": 1.0,
    "refinements": 4, "T_end": 0.01, "dt": 0.001,
}


class TestFourcInlineTsiMonolithic(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def _gen(self, params: dict | None = None) -> str:
        return self.backend.generate_input(
            "tsi", "monolithic_3d", params or {})

    def test_no_placeholders(self) -> None:
        """A literal <...> scalar in the output is exactly the bug
        that made 4C MPI_Abort in MatchTree (probe 2026-06-12)."""
        content = self._gen(dict(PROBE_PARAMS))
        hits = _PLACEHOLDER.findall(content)
        self.assertFalse(
            hits,
            f"tsi/monolithic_3d emits un-substituted placeholders "
            f"{hits[:5]} — it regressed from the inline-mesh route "
            f"back to the raw generator template.")

    def test_self_contained_inline_mesh(self) -> None:
        """Inline-mesh inputs embed NODE COORDS — no external mesh
        file reference that the user's work dir won't have."""
        content = self._gen()
        self.assertIn("NODE COORDS", content,
                      "tsi/monolithic_3d lost its inline mesh.")
        self.assertNotIn("FILE:", content,
                         "tsi/monolithic_3d references an external "
                         "mesh file — not self-contained.")

    def test_physics_is_monolithic_tsi(self) -> None:
        """The row is named monolithic_3d — it must ship the genuine
        monolithic coupling algorithm, not a one-way solve."""
        content = self._gen()
        self.assertIn('PROBLEMTYPE: "Thermo_Structure_Interaction"',
                      content)
        self.assertIn('COUPALGO: "tsi_monolithic"', content,
                      "tsi/monolithic_3d must use the monolithic "
                      "coupling algorithm — anything else is "
                      "mislabeled physics.")
        self.assertIn("TSI DYNAMIC/MONOLITHIC", content,
                      "monolithic TSI needs its TSI DYNAMIC/MONOLITHIC "
                      "section (coupled Newton tolerances + solver).")
        # Merged block matrix is what lets the direct UMFPACK solver
        # handle the coupled system without Belos/Teko XML files.
        self.assertIn("MERGE_TSI_BLOCK_MATRIX: true", content)
        self.assertIn("SOLIDSCATRA HEX8", content)
        self.assertIn("CLONING MATERIAL MAP", content)
        self.assertIn("DIM: 3", content)

    def test_params_override_defaults(self) -> None:
        """params must flow through the routing lambda."""
        content = self._gen({"E": 123.5, "T_end": 0.005, "dt": 0.001})
        self.assertIn("123.5", content)
        self.assertRegex(content, r"NUMSTEP:\s*5\b")

    def test_mesh_capped_at_8(self) -> None:
        """The probe passes nx=ny=nz=16; a 16^3 SOLIDSCATRA monolithic
        solve is too big, so the route caps each direction at 8
        (9^3 = 729 nodes)."""
        content = self._gen(dict(PROBE_PARAMS))
        node_count = len(re.findall(r'"NODE \d+ COORD', content))
        self.assertEqual(node_count, 9 ** 3,
                         "nx/ny/nz=16 must be capped to 8 per "
                         "direction (729 nodes), got a different "
                         "mesh size.")
        elem_count = len(re.findall(r"SOLIDSCATRA HEX8", content))
        self.assertEqual(elem_count, 8 ** 3)


if __name__ == "__main__":
    unittest.main()
