"""Regression: fourc inline-mesh lubrication slider-bearing row.

Created 2026-06-12. The catalog row lubrication/slider_bearing_2d
fell through to the generator template, which emitted literal
<placeholder> scalars plus an external Exodus mesh (FILE: ...) —
unrunnable as emitted, aborting 4C's MatchTree (probe 2026-06-12).

The row is now routed to matched_lubrication_slider_bearing_input,
an inline-mesh port of the authoritative 4C corpus case
tests/input_files/lubrication_sb_2d.4C.yaml (PURE_LUB Reynolds
solve, LUBRICATION QUAD4 elements, MAT_lubrication). Verified live
against the 4C binary 2026-06-12: rc=0 in ~0.5 s.

These tests are GEN-ONLY (no 4C binary needed) so they run in CI:
they pin that generate_input returns self-contained YAML with zero
un-substituted placeholders, an inline mesh, the right lubrication
physics keys, and that params flow through the routing lambda.
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

_PHYSICS = "lubrication"
_VARIANT = "slider_bearing_2d"


class TestFourcInlineLubrication(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def _gen(self, params: dict | None = None) -> str:
        return self.backend.generate_input(_PHYSICS, _VARIANT, params or {})

    def test_no_placeholders(self) -> None:
        """A literal <...> scalar is exactly the bug that made 4C
        MPI_Abort in MatchTree (probe 2026-06-12)."""
        content = self._gen()
        hits = _PLACEHOLDER.findall(content)
        self.assertFalse(
            hits,
            f"{_PHYSICS}/{_VARIANT} emits un-substituted placeholders "
            f"{hits[:5]} — it regressed from the inline-mesh route "
            f"back to a raw generator template.")

    def test_self_contained_inline_mesh(self) -> None:
        """Inline-mesh input embeds NODE COORDS and references no
        external mesh file the user's work dir won't have."""
        content = self._gen()
        self.assertIn("NODE COORDS", content,
                      "lost its inline mesh.")
        self.assertNotIn("FILE:", content,
                         "references an external mesh file — not "
                         "self-contained.")
        self.assertIn("LUBRICATION ELEMENTS", content)
        self.assertIn("LUBRICATION QUAD4", content)

    def test_lubrication_physics_keys(self) -> None:
        """The physics must be a genuine Reynolds-equation solve:
        PROBLEMTYPE Lubrication, MAT_lubrication, LUBRICATION
        DYNAMIC — the keys the corpus case is built on."""
        content = self._gen()
        self.assertIn('PROBLEMTYPE: "Lubrication"', content)
        self.assertIn("LUBRICATION DYNAMIC", content)
        self.assertIn("MAT_lubrication", content)
        # film-height + surface-velocity functions and the pressure
        # Dirichlet line that make the converging-film pressure build.
        self.assertIn("HEIGHTFEILD", content)
        self.assertIn("PURE_LUB", content)
        self.assertIn("DESIGN LINE DIRICH CONDITIONS", content)

    def test_no_superlu(self) -> None:
        """This build's Superlu segfaults on the Reynolds solve;
        the generator must pin UMFPACK instead."""
        content = self._gen()
        self.assertNotIn("Superlu", content)
        self.assertIn('SOLVER: "UMFPACK"', content)

    def test_params_override_defaults(self) -> None:
        """params (nx) must flow through the routing lambda and the
        mesh must stay capped small for a < 30 s run."""
        content = self._gen({"nx": 8, "ny": 1})
        # 8 columns x 1 row -> 8 LUBRICATION QUAD4 elements.
        elems = re.findall(r"LUBRICATION QUAD4", content)
        self.assertEqual(len(elems), 8)
        # nx is capped at 32 even if the probe passes a large value.
        big = self._gen({"nx": 999, "ny": 999})
        n_big = len(re.findall(r"LUBRICATION QUAD4", big))
        self.assertLessEqual(n_big, 32 * 4)


if __name__ == "__main__":
    unittest.main()
