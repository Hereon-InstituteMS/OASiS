"""Regression: fourc inline-mesh routing for level_set/advection_2d.

Created 2026-06-12. The probe sweep found generate_input("level_set",
"advection_2d") falling through to the generator template carrying
literal <placeholder> scalars and an external Exodus mesh reference —
4C aborted in MatchTree. The row is now routed to
matched_level_set_advection_input (inline_mesh.py): PROBLEMTYPE
Level_Set on a one-element-thick pseudo-2D TRANSP HEX8 / TYPE Ls
layer, signed-distance circle initial field (shifted by +1.0 — the
corpus gaussian-hill workaround for builds without Qhull) advected by
a rigid-rotation velocity FUNCT.

Verified live against the 4C binary 2026-06-12: rc=0 in ~1 s
(10 steps, VTU output via output-scatra.pvd), deterministic on rerun.

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

# The exact parameter dict the probe sweep passes to every row —
# the route must accept it gracefully (irrelevant keys ignored).
_PROBE_PARAMS = {
    "kappa": 1.0, "nx": 16, "ny": 16, "nz": 16, "E": 1000.0,
    "nu": 0.3, "rho": 1.0, "Re": 100, "mu": 1.0, "refinements": 4,
    "T_end": 0.01, "dt": 0.001,
}


class TestFourcInlineLevelSet(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def _gen(self, params: dict | None = None) -> str:
        return self.backend.generate_input(
            "level_set", "advection_2d", dict(params or _PROBE_PARAMS))

    def test_no_placeholders(self) -> None:
        """A literal <...> scalar in the output is exactly the bug
        that made 4C MPI_Abort in MatchTree (probe 2026-06-12)."""
        content = self._gen()
        hits = _PLACEHOLDER.findall(content)
        self.assertFalse(
            hits,
            f"level_set/advection_2d emits un-substituted placeholders "
            f"{hits[:5]} — it regressed from the inline-mesh route back "
            f"to a raw generator template.")

    def test_self_contained_inline_mesh(self) -> None:
        """Inline-mesh input embeds NODE COORDS — no external mesh
        file reference that the user's work dir won't have."""
        content = self._gen()
        self.assertIn("NODE COORDS:", content)
        self.assertIn("TRANSPORT ELEMENTS:", content)
        self.assertNotIn("FILE:", content)
        self.assertNotIn(".e\"", content)

    def test_level_set_physics_sections(self) -> None:
        """Pin the physics: real Level_Set problemtype, signed-distance
        circle initial field via FUNCT2, rigid-rotation velocity via
        FUNCT1, Ls-type transport elements, zero diffusivity."""
        content = self._gen()
        self.assertIn('PROBLEMTYPE: "Level_Set"', content)
        self.assertIn("LEVEL-SET CONTROL:", content)
        # Initial field: signed-distance circle, field_by_function.
        self.assertIn('INITIALFIELD: "field_by_function"', content)
        self.assertIn("sqrt((x-0.5)^2+(y-0.5)^2)-0.25", content)
        # Velocity: prescribed rigid rotation about the center.
        self.assertIn('VELOCITYFIELD: "function"', content)
        self.assertIn('SYMBOLIC_FUNCTION_OF_SPACE_TIME: "-(y-0.5)"',
                      content)
        self.assertIn('SYMBOLIC_FUNCTION_OF_SPACE_TIME: "(x-0.5)"',
                      content)
        # Level-set transport elements (TYPE Ls, not Std) — HEX8
        # because 4C's zero-isosurface capture aborts on QUAD4.
        self.assertIn("TRANSP HEX8", content)
        self.assertIn("MAT 1 TYPE Ls", content)
        self.assertIn("MAT_scatra:", content)

    def test_params_override(self) -> None:
        """nx/ny and the route's own time params must take effect;
        oversized meshes are capped at 32."""
        coarse = self._gen({"nx": 4, "ny": 4, "numstep": 3,
                            "timestep": 0.1})
        fine = self._gen({"nx": 8, "ny": 8})
        self.assertLess(coarse.count('"NODE '), fine.count('"NODE '))
        self.assertIn("NUMSTEP: 3", coarse)
        self.assertIn("TIMESTEP: 0.1", coarse)
        # 4x4x1 pseudo-2D layer -> 5*5*2 = 50 mesh nodes.
        self.assertEqual(coarse.count("COORD "), 50)
        # Cap: nx=999 must clamp to 32 (33*33*2 = 2178 nodes).
        capped = self._gen({"nx": 999, "ny": 999})
        self.assertEqual(capped.count("COORD "), 2178)


if __name__ == "__main__":
    unittest.main()
