"""Regression: fourc inline-mesh routing for electrochemistry/nernst_planck_3d.

Created 2026-06-12. The probe sweep found this row falling through to
the generator template in backends/fourc/generators/electrochemistry.py,
which emits literal <placeholder> scalars and references an external
Exodus mesh — 4C aborts in MatchTree. The row is now routed in
backend._generate_inline() to matched_nernst_planck_3d_input():
a self-contained Nernst-Planck binary electrolyte on [0,1]^3 with two
MAT_ion species (valences +1/-1), electroneutrality closure
(ELCH CONTROL EQUPOT "ENC", element TYPE ElchNP), cation concentration
pinned on the x=0 / x=1 faces, and the potential grounded at a corner
node. Verified live against the 4C binary 2026-06-12: rc=0 in ~1.4 s
(10 one-step-theta steps, Newton converged every step).

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

# The probe's standard parameter dict — the routing lambda must
# tolerate every key here (and must NOT wire nx/ny/nz into the
# 3D mesh resolution: nz=16 would inflate the 3-species nonlinear
# solve far past the runtime budget).
PROBE_PARAMS = {
    "kappa": 1.0, "nx": 16, "ny": 16, "nz": 16,
    "E": 1000.0, "nu": 0.3, "rho": 1.0, "Re": 100,
    "mu": 1.0, "refinements": 4, "T_end": 0.01, "dt": 0.001,
}


class TestFourcInlineElectrochemistry(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def _gen(self, params: dict | None = None) -> str:
        return self.backend.generate_input(
            "electrochemistry", "nernst_planck_3d",
            dict(PROBE_PARAMS) if params is None else params)

    def test_no_placeholders(self) -> None:
        """A literal <...> scalar is exactly the bug that made 4C
        MPI_Abort in MatchTree (probe 2026-06-12)."""
        content = self._gen()
        hits = _PLACEHOLDER.findall(content)
        self.assertFalse(
            hits,
            f"electrochemistry/nernst_planck_3d emits un-substituted "
            f"placeholders {hits[:5]} — it regressed from the "
            f"inline-mesh route back to the raw generator template.")

    def test_self_contained_inline_mesh(self) -> None:
        """Inline-mesh input embeds NODE COORDS — no external mesh
        file reference that the user's work dir won't have."""
        content = self._gen()
        self.assertIn("NODE COORDS", content,
                      "nernst_planck_3d lost its inline mesh.")
        self.assertNotIn("FILE:", content,
                         "nernst_planck_3d references an external "
                         "mesh file — not self-contained.")

    def test_nernst_planck_physics(self) -> None:
        """Two ionic species + ENC potential closure on ElchNP
        transport elements — the corpus-authoritative Nernst-Planck
        setup (elch_Kwok_Wu_BDF2.4C.yaml)."""
        content = self._gen()
        self.assertIn("ELCH CONTROL", content)
        self.assertIn('EQUPOT: "ENC"', content)
        self.assertEqual(content.count("MAT_ion"), 2,
                         "binary electrolyte needs exactly two "
                         "MAT_ion species")
        self.assertIn("TYPE ElchNP", content,
                      "transport elements must use the Nernst-Planck "
                      "element type token")
        self.assertIn('PROBLEMTYPE: "Electrochemistry"', content)

    def test_probe_mesh_size_not_inflated(self) -> None:
        """Resolution comes from 'n' (default 4), not the probe's
        nx/ny/nz=16: a 16^3 three-field nonlinear solve would blow
        the runtime budget."""
        content = self._gen()
        n_elements = len(re.findall(r"TRANSP HEX8", content))
        self.assertLessEqual(n_elements, 6 ** 3,
                             "mesh resolution leaked from nx/ny/nz — "
                             "must stay at n<=6 per direction")
        n = re.findall(r'"NODE \d+ COORD', content)
        self.assertEqual(len(n), 5 ** 3,
                         "expected the default n=4 mesh (125 nodes)")

    def test_params_override_defaults(self) -> None:
        """params must flow through the routing lambda."""
        params = dict(PROBE_PARAMS)
        params.update({"n": 3, "c_left": 7.25, "numstep": 6,
                       "dt": 0.0005})
        content = self._gen(params)
        self.assertEqual(len(re.findall(r'"NODE \d+ COORD', content)),
                         4 ** 3, "n=3 must give a 4^3-node mesh")
        self.assertIn("7.25", content)
        self.assertRegex(content, r"NUMSTEP:\s*6\b")
        self.assertRegex(content, r"TIMESTEP:\s*0.0005\b")


if __name__ == "__main__":
    unittest.main()
