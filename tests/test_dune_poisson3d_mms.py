"""Gen-only tests for the DUNE-fem poisson_3d_varcoeff generator (E5).

Created 2026-08-01 (deliverable E5): 3D VARIABLE-COEFFICIENT Poisson on
[0, L]^3 with manufactured solution, exact Dirichlet data, Lagrange
order as a parameter, uniform refinement and per-level L2/H1 error
lines. This is the first genuinely hard DUNE-fem family (the existing
poisson_2d is a single constant-force instance with no error
measurement).

These tests do NOT require dune.fem — they pin the *generator*
contract only:

  * the template key is registered and reachable through the backend's
    generate_input("poisson_mms", "3d_varcoeff", ...);
  * the generated Python compiles and has no unresolved placeholders;
  * every parameter is honoured in the emitted source;
  * the EMITTED u*/kappa expression text is cross-checked by
    ast-evaluating it at random points against independent reference
    implementations (manufactured_solution_at / diffusion_at) — this
    catches literal-formatting and operator-precedence bugs, the real
    risk in a family whose source term is otherwise exact (UFL
    differentiates u* symbolically, so there is no hand-derived f to
    transcribe);
  * the template uses the NON-deprecated dune-fem 2.10 API
    (dune.fem.integrate, nonlinear.*/linear.* parameter keys);
  * validate_parameters flags nonsensical inputs (including a
    kappa-positivity/ellipticity guard) and the generator refuses to
    emit code for them.

The live execution gate is the convergence run on this install
(conda env dune-fem-env, dune-fem 2.10, 2026-08-01), dev-draw
parameters L=2, a=2, b=1, c=1.5, d=0.5, amp=1.3, kappa=(2, -0.75,
0.5, 1.0), 4 levels:

  order=1 (n0=4): L2 EOCs 1.984 / 1.996 / 1.999 (theory 2),
                  H1 EOCs 1.028 / 1.007 / 1.002 (theory 1)
  order=2 (n0=2): L2 EOCs 1.628(pre-asymptotic) / 2.923 / 2.982
                  (theory 3),
                  H1 EOCs 1.059(pre-asymptotic) / 1.990 / 1.998
                  (theory 2)

(recorded also in the module KNOWLEDGE).
"""
from __future__ import annotations

import ast
import math
import random
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from backends.dune.generators.poisson_mms3d import (  # noqa: E402
    _DEFAULTS, diffusion_at, kappa_min, manufactured_solution_at,
    validate_parameters)

# Dev-draw parameter set exercising every knob (distinct from the
# module defaults on purpose).
DEV_PARAMS = {
    "order": 2,
    "levels": 4,
    "n0": 2,
    "L": 2.0,
    "a": 2.0, "b": 1.0, "c": 1.5,
    "d": 0.5, "amp": 1.3,
    "k0": 2.0, "kx": -0.75, "ky": 0.5, "kz": 1.0,
}


def _extract_exprs(src: str) -> dict:
    """ast-parse the generated script; return compiled u_exact/kappa
    expressions plus the module-level L constant."""
    tree = ast.parse(src)
    found: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("u_exact", "kappa", "L") and name not in found:
                found[name] = compile(ast.Expression(node.value),
                                      "<expr>", "eval")
    return found


def _eval_emitted(src: str, x: float, y: float, z: float) -> tuple:
    exprs = _extract_exprs(src)
    ns = {"sin": math.sin, "pi": math.pi}
    L = eval(exprs["L"], ns)  # noqa: S307 — our own generated literal
    ns.update({"L": L, "x": (x, y, z)})
    return eval(exprs["u_exact"], ns), eval(exprs["kappa"], ns)  # noqa: S307


class TestPoisson3dVarcoeffGenerator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("dune")
        if cls.backend is None:
            raise unittest.SkipTest("dune backend not registered")

    # ── registration / dispatch ─────────────────────────────────────

    def test_template_key_registered(self) -> None:
        from backends.dune.generators import GENERATORS
        self.assertIn("poisson_mms_3d_varcoeff", GENERATORS)

    def test_backend_advertises_variant(self) -> None:
        caps = {c.name: c for c in self.backend.supported_physics()}
        self.assertIn("poisson_mms", caps)
        self.assertIn("3d_varcoeff", caps["poisson_mms"].template_variants)
        self.assertIn(3, caps["poisson_mms"].spatial_dims)

    def test_knowledge_registered_and_populated(self) -> None:
        k = self.backend.get_knowledge("poisson_mms")
        self.assertIsInstance(k, dict)
        self.assertTrue(k.get("description"))
        self.assertGreaterEqual(len(k.get("pitfalls", [])), 4)
        self.assertIn("k+1", str(k.get("expected_order", "")))

    def test_every_pitfall_has_signal_line(self) -> None:
        # dune sits at the 100% Signal-coverage floor — a new pitfall
        # without a Signal: clause would regress the coverage gate.
        k = self.backend.get_knowledge("poisson_mms")
        for i, pit in enumerate(k["pitfalls"]):
            self.assertIn("Signal:", pit,
                          f"pitfall #{i} lacks a Signal: clause")

    # ── default generation ──────────────────────────────────────────

    def test_default_generation_is_complete_python(self) -> None:
        src = self.backend.generate_input("poisson_mms", "3d_varcoeff", {})
        self.assertIsInstance(src, str)
        for needle in ("structuredGrid", "lagrange", "galerkin",
                       "DirichletBC", "SpatialCoordinate",
                       "results_summary.json", "writeVTK", "EOC"):
            self.assertIn(needle, src)
        # machine-readable per-level error line
        self.assertIn('f"level {level} n {n} dofs {space.size} "', src)
        # compiles as Python and passes the backend's own validator
        compile(src, "<gen>", "exec")
        self.assertEqual(self.backend.validate_input(src), [])

    def test_no_unresolved_placeholders(self) -> None:
        src = self.backend.generate_input("poisson_mms", "3d_varcoeff",
                                          DEV_PARAMS)
        for tok in ("{order}", "{levels}", "{n0}", "{L}", "{a}", "{amp}",
                    "{k0}", "{kx}", "None*", "*None"):
            self.assertNotIn(tok, src)

    def test_uses_modern_dune_fem_api(self) -> None:
        # the deprecated spellings were observed live (see KNOWLEDGE);
        # the template must emit the 2.10 replacements.
        src = self.backend.generate_input("poisson_mms", "3d_varcoeff", {})
        self.assertIn("from dune.fem import integrate", src)
        self.assertNotIn("from dune.fem.function import integrate", src)
        self.assertIn("nonlinear.tolerance", src)
        self.assertNotIn("newton.tolerance", src)
        self.assertNotIn("newton.linear", src)

    # ── parameters honoured ─────────────────────────────────────────

    def test_parameters_honoured(self) -> None:
        src = self.backend.generate_input("poisson_mms", "3d_varcoeff",
                                          DEV_PARAMS)
        self.assertIn("order = 2", src)
        self.assertIn("levels = 4", src)
        self.assertIn("n0 = 2", src)
        self.assertIn("L = 2", src)
        for lit in ("1.3", "0.5", "-0.75", "1.5"):
            self.assertIn(lit, src)

    def test_emitted_expressions_match_reference(self) -> None:
        """ast-eval the EMITTED u*/kappa text against the module's
        independent reference evaluators at random points."""
        rng = random.Random(20260801)
        for params in ({}, DEV_PARAMS,
                       {"L": 3.5, "a": -1.0, "b": 0.5, "c": 2.0,
                        "d": -0.25, "amp": 0.7,
                        "k0": 4.0, "kx": -1.5, "ky": -1.0, "kz": -0.5}):
            src = self.backend.generate_input("poisson_mms", "3d_varcoeff",
                                              params)
            L = {**_DEFAULTS, **params}["L"]
            for _ in range(25):
                x, y, z = (rng.uniform(0, L) for _ in range(3))
                u_e, kap_e = _eval_emitted(src, x, y, z)
                self.assertAlmostEqual(
                    u_e, manufactured_solution_at(params, x, y, z),
                    delta=1e-12 * max(1.0, abs(u_e)))
                self.assertAlmostEqual(
                    kap_e, diffusion_at(params, x, y, z),
                    delta=1e-12 * max(1.0, abs(kap_e)))

    def test_kappa_min_is_exact_corner_minimum(self) -> None:
        rng = random.Random(42)
        for _ in range(50):
            params = {"L": rng.uniform(0.5, 4.0),
                      "k0": rng.uniform(0.1, 5.0),
                      "kx": rng.uniform(-2, 2),
                      "ky": rng.uniform(-2, 2),
                      "kz": rng.uniform(-2, 2)}
            L = params["L"]
            corner_min = min(
                diffusion_at(params, i * L, j * L, k * L)
                for i in (0, 1) for j in (0, 1) for k in (0, 1))
            self.assertAlmostEqual(kappa_min(params), corner_min,
                                   places=12)

    # ── validation guards ───────────────────────────────────────────

    def test_defaults_and_dev_params_validate_clean(self) -> None:
        self.assertEqual(validate_parameters({}), [])
        self.assertEqual(validate_parameters(DEV_PARAMS), [])

    def test_validation_rejects_bad_inputs(self) -> None:
        cases = {
            "order 0": {"order": 0},
            "order 7": {"order": 7},
            "order str": {"order": "two"},
            "order bool": {"order": True},
            "levels 0": {"levels": 0},
            "levels 9": {"levels": 9},
            "n0 too small": {"n0": 1},
            "L zero": {"L": 0.0},
            "L negative": {"L": -1.0},
            "nan coefficient": {"a": float("nan")},
            "inf amp": {"amp": float("inf")},
            "zero solution": {"amp": 0.0, "d": 0.0},
            "kappa nonpositive": {"k0": 1.0, "kx": -1.0, "ky": -0.5,
                                  "kz": 0.0},
            "cost guard": {"order": 4, "n0": 8, "levels": 5},
        }
        for label, params in cases.items():
            with self.subTest(label):
                problems = validate_parameters(params)
                self.assertTrue(problems,
                                f"{label}: expected rejection, got OK")
                with self.assertRaises(ValueError):
                    self.backend.generate_input("poisson_mms", "3d_varcoeff",
                                                params)

    def test_cost_guard_boundary(self) -> None:
        # n0 * 2^(levels-1) * order == 96 is the largest admissible
        self.assertEqual(
            validate_parameters({"order": 3, "n0": 4, "levels": 4}), [])
        self.assertTrue(
            validate_parameters({"order": 4, "n0": 4, "levels": 4}))

    def test_ellipticity_guard_message_names_the_minimum(self) -> None:
        problems = validate_parameters(
            {"k0": 0.5, "kx": -1.0, "ky": 0.0, "kz": 0.0})
        self.assertTrue(any("ellipticity" in p for p in problems))


if __name__ == "__main__":
    unittest.main()
