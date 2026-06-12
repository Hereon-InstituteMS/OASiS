"""Regression: fourc inline-mesh route for cardiovascular0d/windkessel_3d.

Created 2026-06-12. The cardiovascular0d generator template
(src/backends/fourc/generators/cardiovascular0d.py) returned a ONE-LINE
comment ("# Cardiovascular0D template - use DESIGN SURF CARDIOVASCULAR0D
CONDITIONS"), NOT a YAML dict, so generate_input('cardiovascular0d',
'windkessel_3d', ...) fell through to that comment and validate_input
failed with "Input is not a YAML dictionary" -- the probe never reached
the run stage.

The row is now routed to matched_cardiovascular0d_windkessel_input, a
self-contained inline-mesh port of
tests/input_files/cardiovascular0d_4elementwindkessel_structure_direct_stat.4C.yaml:
a structural HEX8 cube whose x=lx face is coupled to a lumped-parameter
4-element Windkessel model via DESIGN SURF CARDIOVASCULAR 0D conditions.

Verified live against the 4C binary 2026-06-12: rc=0,
"processor 0 finished normally", "Model: 4-element windkessel" active.

This test is GEN-ONLY (no 4C binary needed) so it runs in CI: it pins that
generate_input returns parseable, validate_input-clean YAML with zero
un-substituted placeholders, an inline mesh (no external files), and the
physics actually present (the 0D windkessel condition + 0D-structure
coupling). The live execution gate is the probe sweep
(benchmarks/probe_all_templates.py --backend fourc).
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# Matches generator-template placeholders like <Young_modulus> or
# <mesh_file.e> but NOT YAML flow content or comparison operators.
_PLACEHOLDER = re.compile(r"<[A-Za-z_][A-Za-z0-9_. ]*>")

PHYSICS = "cardiovascular0d"
VARIANT = "windkessel_3d"

# The probe params the routing lambda must tolerate (task #29 sweep).
PROBE_PARAMS = {
    "kappa": 1.0, "nx": 16, "ny": 16, "nz": 16,
    "E": 1000.0, "nu": 0.3, "rho": 1.0, "Re": 100, "mu": 1.0,
    "refinements": 4, "T_end": 0.01, "dt": 0.001,
}


class TestFourcInlineCardiovascular0D(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")
        if cls.backend is None:
            raise unittest.SkipTest("fourc backend not registered")

    def _gen(self, params: dict | None = None) -> str:
        return self.backend.generate_input(PHYSICS, VARIANT, params or {})

    def test_is_yaml_dict_and_validates(self) -> None:
        """The original regression: the template used to be a one-line
        comment, so validate_input failed with 'Input is not a YAML
        dictionary'. It must now parse as a YAML dict and validate clean."""
        content = self._gen(PROBE_PARAMS)
        parsed = yaml.safe_load(content)
        self.assertIsInstance(
            parsed, dict,
            "cardiovascular0d/windkessel_3d must emit a YAML dictionary, "
            "not a bare comment (the original bug).")
        self.assertEqual(
            self.backend.validate_input(content), [],
            "validate_input must return [] for the routed input.")

    def test_no_placeholders(self) -> None:
        content = self._gen(PROBE_PARAMS)
        hits = _PLACEHOLDER.findall(content)
        self.assertFalse(
            hits,
            f"emits un-substituted placeholders {hits[:5]} -- it regressed "
            f"from the inline-mesh route back to a raw generator template.")

    def test_self_contained_inline_mesh(self) -> None:
        """Inline-mesh input embeds NODE COORDS and references no external
        mesh file (FILE:) the user's work dir won't have."""
        content = self._gen(PROBE_PARAMS)
        self.assertIn("NODE COORDS", content, "lost its inline mesh.")
        self.assertNotIn(
            "FILE:", content,
            "references an external mesh file -- not self-contained.")

    def test_physics_is_cardiovascular0d_windkessel(self) -> None:
        """The defining physics must be present: the 0D 4-element
        Windkessel surface condition and the 0D-structure coupling, on a
        structural HEX8 cube."""
        content = self._gen(PROBE_PARAMS)
        self.assertIn(
            "DESIGN SURF CARDIOVASCULAR 0D 4-ELEMENT WINDKESSEL CONDITIONS",
            content,
            "the 4-element windkessel surface condition is the whole point.")
        self.assertIn(
            "DESIGN SURF CARDIOVASCULAR 0D-STRUCTURE COUPLING CONDITIONS",
            content,
            "the 0D model must be coupled to the 3D structural cavity.")
        self.assertIn("CARDIOVASCULAR 0D-STRUCTURE COUPLING", content,
                      "the 0D-3D coupling solver section must be present.")
        self.assertIn("SOLID HEX8", content)
        self.assertIn('PROBLEMTYPE: "Structure"', content)

    def test_params_override_defaults(self) -> None:
        """params must flow through the routing lambda."""
        content = self.backend.generate_input(
            PHYSICS, VARIANT, {"E": 123.5})
        self.assertIn("123.5", content)
        # dt/T_end drive the step count: T_end/dt steps (capped at 10).
        content = self.backend.generate_input(
            PHYSICS, VARIANT, {"T_end": 0.3, "dt": 0.1})
        self.assertRegex(content, r"NUMSTEP:\s*3\b")

    def test_resolution_is_capped(self) -> None:
        """The probe passes nx=ny=nz=16; the route keys off 'n' (capped
        <=4) so the monolithic 0D-3D solve cannot be inflated."""
        content = self._gen(PROBE_PARAMS)
        node_lines = [ln for ln in content.splitlines()
                      if "COORD" in ln]
        # n<=4 cube -> at most 5^3 = 125 nodes.
        self.assertLessEqual(
            len(node_lines), 125,
            f"mesh inflated to {len(node_lines)} nodes -- resolution cap "
            f"failed (probe nx/ny/nz=16 leaked through).")


if __name__ == "__main__":
    unittest.main()
