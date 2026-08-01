"""Gen-only tests for the deal.II poisson_3d_mixed_bc generator.

Created 2026-08-01 (deliverable E1): 3D Poisson on the unit cube with
MIXED Dirichlet-Neumann boundaries, manufactured solution (MMS),
global refinement and per-cycle L2 error output.

These tests do NOT require the deal.II library — they pin the
*generator* contract only:

  * the template key is registered and reachable through the backend's
    generate_input("poisson", "3d_mixed_bc", ...);
  * the generated C++ has no unresolved Python-format placeholders;
  * every parameter (degree, cycles, MMS coefficients, Neumann-face
    set) is honoured in the emitted source;
  * the Neumann/Dirichlet face split is consistent (Neumann set plus
    Dirichlet comment cover all 6 colorized hyper_cube face ids);
  * validate_parameters flags nonsensical inputs and the generator
    refuses to emit code for them.

The live execution gate is the convergence smoke run (deal.II
9.8.0-pre, 2026-08-01): degree=1 observed L2 orders 2.06/1.96/1.99/
2.00, degree=2 observed 1.55(pre-asymptotic)/2.92/2.98/3.00 — the
verified numbers are recorded in the module KNOWLEDGE.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# Dev-draw parameter set exercising every knob (distinct from the
# module defaults on purpose).
DEV_PARAMS = {
    "degree": 3,
    "cycles": 5,
    "initial_refinements": 2,
    "a": 2.0, "b": 3.0, "c": 1.5,
    "d": 0.25, "amp": 2.5,
    "neumann_faces": [0, 2],
}

# Raw Python-format tokens that would betray an f-string escape bug.
_UNRESOLVED_TOKENS = (
    "{degree}", "{cycles}", "{init_ref}", "{a_lit}", "{b_lit}",
    "{c_lit}", "{d_lit}", "{amp_lit}", "{neumann_set_lit}",
    "{neumann_comment}", "{dirichlet_comment}", "None",
)


class TestPoisson3dMixedBcGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("dealii")
        if cls.backend is None:
            raise unittest.SkipTest("dealii backend not registered")

    # ── registration / dispatch ─────────────────────────────────────

    def test_template_key_registered(self) -> None:
        from backends.dealii.generators import list_template_keys
        self.assertIn("poisson_3d_mixed_bc", list_template_keys())

    def test_backend_advertises_variant(self) -> None:
        caps = {c.name: c for c in self.backend.supported_physics()}
        self.assertIn("poisson", caps)
        self.assertIn("3d_mixed_bc", caps["poisson"].template_variants)

    def test_knowledge_registered_and_populated(self) -> None:
        from backends.dealii.generators import get_knowledge
        k = get_knowledge("poisson_mixed_bc")
        self.assertIsInstance(k, dict)
        self.assertTrue(k.get("description"))
        self.assertTrue(k.get("pitfalls"))
        self.assertGreaterEqual(len(k["pitfalls"]), 4)
        # The headline theory claim must be present.
        self.assertIn("k+1", str(k.get("expected_order", "")))

    # ── default generation ──────────────────────────────────────────

    def test_default_generation_is_complete_cpp(self) -> None:
        src = self.backend.generate_input("poisson", "3d_mixed_bc", {})
        self.assertIsInstance(src, str)
        self.assertIn("int main", src)
        self.assertIn("#include <deal.II/fe/fe_values.h>", src)
        self.assertIn("FEFaceValues<3>", src)
        self.assertIn("integrate_difference", src)
        self.assertIn("compute_global_error", src)
        self.assertIn('"cycle "', src)  # machine-readable error line
        for tok in _UNRESOLVED_TOKENS:
            self.assertNotIn(tok, src,
                             f"unresolved placeholder {tok!r} leaked "
                             "into the generated C++")
        # Must also pass the backend's own stub/placeholder detector.
        self.assertEqual(self.backend.validate_input(src), [])

    # ── parameters honoured ─────────────────────────────────────────

    def test_parameters_respected(self) -> None:
        src = self.backend.generate_input(
            "poisson", "3d_mixed_bc", dict(DEV_PARAMS))
        self.assertIn("const unsigned int degree               = 3;", src)
        self.assertIn("const unsigned int n_cycles             = 5;", src)
        self.assertIn("const unsigned int initial_refinements  = 2;", src)
        for lit in ("2.0", "3.0", "1.5", "0.25", "2.5"):
            self.assertIn(lit, src)
        self.assertIn("mms_amp = 2.5", src)
        self.assertIn("mms_d   = 0.25", src)

    def test_neumann_dirichlet_face_split(self) -> None:
        src = self.backend.generate_input(
            "poisson", "3d_mixed_bc", dict(DEV_PARAMS))
        # Neumann ids {0, 2} in the C++ set literal...
        self.assertIn(
            "const std::set<types::boundary_id> neumann_ids = {0, 2};",
            src)
        # ...and the complement documented as Dirichlet.
        self.assertIn("Dirichlet faces: 1, 3, 4, 5", src)
        # colorize is mandatory for per-face ids.
        self.assertIn("/*colorize=*/true", src)

    def test_default_face_split_covers_all_six_ids(self) -> None:
        src = self.backend.generate_input("poisson", "3d_mixed_bc", {})
        self.assertIn(
            "const std::set<types::boundary_id> neumann_ids = {1, 3, 5};",
            src)
        self.assertIn("Dirichlet faces: 0, 2, 4", src)

    # ── validate_parameters ─────────────────────────────────────────

    def test_validate_parameters_accepts_defaults_and_dev_draw(self) -> None:
        from backends.dealii.generators.poisson_mixed_bc import (
            validate_parameters,
        )
        self.assertEqual(validate_parameters({}), [])
        self.assertEqual(validate_parameters(dict(DEV_PARAMS)), [])

    def test_validate_parameters_flags_bad_inputs(self) -> None:
        from backends.dealii.generators.poisson_mixed_bc import (
            validate_parameters,
        )
        bad_cases = {
            "degree_zero":      {"degree": 0},
            "degree_str":       {"degree": "two"},
            "degree_bool":      {"degree": True},
            "cycles_negative":  {"cycles": -1},
            "cycles_float":     {"cycles": 2.5},
            "initref_negative": {"initial_refinements": -2},
            "mesh_blowup":      {"cycles": 6, "initial_refinements": 5},
            "coeff_string":     {"a": "fast"},
            "coeff_nan":        {"b": float("nan")},
            "zero_solution":    {"amp": 0.0, "d": 0.0},
            "face_out_of_range": {"neumann_faces": [1, 7]},
            "face_duplicates":  {"neumann_faces": [1, 1, 3]},
            "all_neumann":      {"neumann_faces": [0, 1, 2, 3, 4, 5]},
            "faces_not_a_list": {"neumann_faces": "left"},
        }
        for label, params in bad_cases.items():
            with self.subTest(case=label):
                problems = validate_parameters(params)
                self.assertTrue(
                    problems,
                    f"validate_parameters accepted bad input {params!r}")
                self.assertTrue(
                    all(isinstance(p, str) and p for p in problems))

    def test_generator_refuses_invalid_parameters(self) -> None:
        from backends.dealii.generators.poisson_mixed_bc import (
            _poisson_3d_mixed_bc,
        )
        with self.assertRaises(ValueError):
            _poisson_3d_mixed_bc({"degree": 0})
        with self.assertRaises(ValueError):
            _poisson_3d_mixed_bc(
                {"neumann_faces": [0, 1, 2, 3, 4, 5]})

    def test_all_dirichlet_is_valid_and_emits_empty_neumann_set(self) -> None:
        """neumann_faces=[] is a legal degenerate case (pure Dirichlet
        MMS study) — the set literal must still be valid C++."""
        from backends.dealii.generators.poisson_mixed_bc import (
            validate_parameters,
        )
        self.assertEqual(validate_parameters({"neumann_faces": []}), [])
        src = self.backend.generate_input(
            "poisson", "3d_mixed_bc", {"neumann_faces": []})
        self.assertIn(
            "const std::set<types::boundary_id> neumann_ids = {};", src)
        self.assertIn("Dirichlet faces: 0, 1, 2, 3, 4, 5", src)


if __name__ == "__main__":
    unittest.main()
