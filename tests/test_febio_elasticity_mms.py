"""Gen-only tests for the FEBio 3D linear-elasticity MMS family.

These tests verify everything that CAN be verified offline (FEBio
4.12.0 is built from source on this install — see the elasticity_mms
KNOWLEDGE build-recipe pitfall — and the separate LIVE convergence gate
checks that a refinement sweep approaches the theoretical displacement
L2 order of 2, which is what confirms the source-derived
body_force_sign = -1 convention):

  - the emitted deck is well-formed FEBio 4.0 XML with no unresolved
    placeholders,
  - node/element counts and coordinates match the n, L parameters,
  - every NodeSet / map / material referenced anywhere actually exists,
  - the per-node Dirichlet maps equal u* at the mapped nodes,
  - validate_parameters rejects bad input,
  - the body-force math expressions equal -div(sigma(u*)) — checked
    BOTH against an independent sympy re-derivation (skipped when sympy
    is absent) AND against a pure-numpy second-order finite-difference
    divergence of the analytically evaluated stress field, so the MMS
    math is not shipped unverified.
"""
from __future__ import annotations

import math
import re
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from backends.febio.generators import elasticity_mms as mms
from backends.febio.generators import GENERATORS as FEBIO_GENERATORS


# Non-default parameters everywhere: anchoring on defaults would hide a
# parameter that is silently ignored.
_PARAMS = {
    "n": 3,
    "L": 2.5,
    "E": 725.0,
    "nu": 0.27,
    "rho": 3.2,
    "amplitude": 2.0e-3,
    "ax": 0.9, "ay": 0.55, "az": 0.7,
    "kx": 1.0, "ky": 2.0, "kz": 1.5,
    "steps": 2,
}


def _gen(params=None):
    return mms._elasticity_mms_3d_cube_hex8(dict(_PARAMS if params is None
                                                 else params))


def _root(params=None) -> ET.Element:
    return ET.fromstring(_gen(params))


def _eval_expr(expr: str, X, Y, Z):
    """Evaluate a FEBio math expression string with numpy semantics."""
    py = expr.replace("^", "**")
    return eval(py, {"__builtins__": {}},
                {"sin": np.sin, "cos": np.cos, "exp": np.exp,
                 "X": X, "Y": Y, "Z": Z})


def _u_star(params, x, y, z):
    return mms.manufactured_displacement_at(params, x, y, z)


def _sigma(params, x, y, z, h=1e-6):
    """Small-strain stress at a point from central FD of u* (numpy)."""
    p = dict(mms._DEFAULTS)
    p.update(params)
    E, nu = float(p["E"]), float(p["nu"])
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    grad = np.zeros((3, 3))
    for j, dv in enumerate(np.eye(3)):
        up = np.array(_u_star(params, x + h * dv[0], y + h * dv[1],
                              z + h * dv[2]))
        um = np.array(_u_star(params, x - h * dv[0], y - h * dv[1],
                              z - h * dv[2]))
        grad[:, j] = (up - um) / (2 * h)
    eps = 0.5 * (grad + grad.T)
    return lam * np.trace(eps) * np.eye(3) + 2 * mu * eps


class TestDeckStructure(unittest.TestCase):
    def test_xml_parses(self):
        root = _root()
        self.assertEqual(root.tag, "febio_spec")
        self.assertEqual(root.get("version"), "4.0")

    def test_no_unresolved_placeholders(self):
        content = _gen()
        self.assertNotIn("{", content)  # no unexpanded format braces
        self.assertNotIn("}", content)
        self.assertNotIn("None", content)
        self.assertNotIn("nan", content.lower())

    def test_registry_wiring(self):
        self.assertIn("elasticity_mms_3d_cube_hex8", FEBIO_GENERATORS)
        self.assertIs(FEBIO_GENERATORS["elasticity_mms_3d_cube_hex8"],
                      mms._elasticity_mms_3d_cube_hex8)

    def test_backend_surface(self):
        from backends.febio.backend import FebioBackend
        b = FebioBackend()
        caps = {c.name: c for c in b.supported_physics()}
        self.assertIn("elasticity_mms", caps)
        self.assertEqual(caps["elasticity_mms"].template_variants,
                         ["3d_cube_hex8"])
        content = b.generate_input("elasticity_mms", "3d_cube_hex8",
                                   dict(_PARAMS))
        self.assertIn("febio_spec", content)
        self.assertEqual(b.validate_input(content), [])

    def test_knowledge_present(self):
        from backends.febio.backend import FebioBackend
        k = FebioBackend().get_knowledge("elasticity_mms")
        self.assertTrue(k)
        self.assertTrue(k.get("pitfalls"))
        self.assertIn("mms_setup", k)

    def test_generate_with_empty_params_works(self):
        # advertised contract: generate_input(name, variant, {}) must work
        root = _root({})
        self.assertEqual(root.tag, "febio_spec")


class TestMeshHonorsParameters(unittest.TestCase):
    def test_node_and_element_counts(self):
        for n in (2, 3, 4):
            p = dict(_PARAMS, n=n)
            root = _root(p)
            nodes = root.findall("./Mesh/Nodes/node")
            elems = root.findall("./Mesh/Elements/elem")
            self.assertEqual(len(nodes), (n + 1) ** 3, f"n={n}")
            self.assertEqual(len(elems), n ** 3, f"n={n}")

    def test_coordinates_span_L(self):
        root = _root()
        L = _PARAMS["L"]
        coords = [tuple(float(v) for v in nd.text.split(","))
                  for nd in root.findall("./Mesh/Nodes/node")]
        arr = np.array(coords)
        self.assertAlmostEqual(arr.min(), 0.0)
        self.assertAlmostEqual(arr.max(), L)
        # first node at origin, last at (L,L,L)
        self.assertEqual(coords[0], (0.0, 0.0, 0.0))
        np.testing.assert_allclose(coords[-1], (L, L, L))
        # uniform spacing along x for the first edge
        n = _PARAMS["n"]
        xs = arr[: n + 1, 0]
        np.testing.assert_allclose(np.diff(xs), L / n)

    def test_connectivity_in_range_and_hex8(self):
        root = _root()
        n = _PARAMS["n"]
        nmax = (n + 1) ** 3
        elems = root.findall("./Mesh/Elements/elem")
        self.assertEqual(root.find("./Mesh/Elements").get("type"), "hex8")
        for el in elems:
            conn = [int(v) for v in el.text.split(",")]
            self.assertEqual(len(conn), 8)
            self.assertEqual(len(set(conn)), 8)
            self.assertTrue(all(1 <= c <= nmax for c in conn))

    def test_material_and_control_honored(self):
        root = _root()
        matl = root.find("./Material/material")
        self.assertEqual(float(matl.findtext("E")), _PARAMS["E"])
        self.assertEqual(float(matl.findtext("v")), _PARAMS["nu"])
        self.assertEqual(float(matl.findtext("density")), _PARAMS["rho"])
        self.assertEqual(matl.get("type"), "isotropic elastic")
        ctrl = root.find("./Control")
        self.assertEqual(ctrl.findtext("analysis"), "STATIC")
        self.assertEqual(int(ctrl.findtext("time_steps")), _PARAMS["steps"])
        self.assertAlmostEqual(float(ctrl.findtext("step_size")),
                               1.0 / _PARAMS["steps"])


class TestReferencesResolve(unittest.TestCase):
    def test_every_referenced_set_and_map_exists(self):
        root = _root()
        nodesets = {ns.get("name") for ns in root.findall("./Mesh/NodeSet")}
        maps = {nd.get("name"): nd
                for nd in root.findall("./MeshData/NodeData")}
        matnames = {m.get("name")
                    for m in root.findall("./Material/material")}

        # bc node_set + map references
        bcs = root.findall("./Boundary/bc")
        self.assertEqual(len(bcs), 3)
        dofs = set()
        for bc in bcs:
            self.assertEqual(bc.get("type"), "prescribed displacement")
            self.assertIn(bc.get("node_set"), nodesets)
            dofs.add(bc.findtext("dof"))
            val = bc.find("value")
            self.assertEqual(val.get("type"), "map")
            self.assertIn(val.text, maps, "bc references undefined map")
            self.assertIsNotNone(val.get("lc"))
        self.assertEqual(dofs, {"x", "y", "z"})

        # NodeData node_set references
        for name, nd in maps.items():
            self.assertIn(nd.get("node_set"), nodesets, name)
            self.assertEqual(nd.get("data_type"), "scalar", name)

        # domain -> material by NAME (v4 requirement)
        dom = root.find("./MeshDomains/SolidDomain")
        self.assertIn(dom.get("mat"), matnames)

        # every lc referenced has a load controller
        lcs = {lc.get("id")
               for lc in root.findall("./LoadData/load_controller")}
        for val in root.iter():
            lc = val.get("lc") if hasattr(val, "get") else None
            if lc is not None:
                self.assertIn(lc, lcs)

    def test_boundary_set_is_complete_and_on_boundary(self):
        n, L = _PARAMS["n"], _PARAMS["L"]
        root = _root()
        coords = {}
        for i, nd in enumerate(root.findall("./Mesh/Nodes/node"), start=1):
            self.assertEqual(int(nd.get("id")), i)
            coords[i] = tuple(float(v) for v in nd.text.split(","))
        ns = root.find("./Mesh/NodeSet[@name='boundary']")
        ids = [int(v) for v in ns.text.split(",")]
        expected_count = (n + 1) ** 3 - (n - 1) ** 3
        self.assertEqual(len(ids), expected_count)
        self.assertEqual(ids, sorted(ids))
        tol = 1e-12
        for nid in ids:
            x, y, z = coords[nid]
            on_face = any(abs(c) < tol or abs(c - L) < tol
                          for c in (x, y, z))
            self.assertTrue(on_face, f"node {nid} not on boundary")

    def test_map_entries_match_u_star(self):
        root = _root()
        coords = {int(nd.get("id")): tuple(float(v)
                                           for v in nd.text.split(","))
                  for nd in root.findall("./Mesh/Nodes/node")}
        ns = root.find("./Mesh/NodeSet[@name='boundary']")
        ids = [int(v) for v in ns.text.split(",")]
        for comp, mapname in ((0, "ux_map"), (1, "uy_map"), (2, "uz_map")):
            nd = root.find(f"./MeshData/NodeData[@name='{mapname}']")
            entries = nd.findall("node")
            self.assertEqual(len(entries), len(ids), mapname)
            for lid_expected, entry in enumerate(entries, start=1):
                self.assertEqual(int(entry.get("lid")), lid_expected)
                nid = ids[lid_expected - 1]
                expect = _u_star(_PARAMS, *coords[nid])[comp]
                self.assertAlmostEqual(float(entry.text), expect,
                                       delta=1e-15 + 1e-12 * abs(expect))

    def test_logfile_output_requested(self):
        root = _root()
        nd = root.find("./Output/logfile/node_data")
        self.assertIsNotNone(nd)
        self.assertEqual(nd.get("data"), "ux;uy;uz")
        self.assertTrue(nd.get("file"))
        plot = root.find("./Output/plotfile")
        self.assertIsNotNone(plot.find("var[@type='displacement']"))


class TestBodyForceExpressions(unittest.TestCase):
    def test_uses_uppercase_reference_coords_only(self):
        for expr in mms.body_force_expressions(_PARAMS):
            # strip function names, then no lowercase letters may remain
            residue = expr.replace("sin", "").replace("cos", "")
            self.assertFalse(re.search(r"[a-z]", residue), expr)
            self.assertTrue(re.search(r"\b[XYZ]\b", expr))

    def test_expressions_in_deck_match_module(self):
        root = _root()
        bl = root.find("./Loads/body_load")
        self.assertEqual(bl.get("type"), "non-const")
        fx, fy, fz = mms.body_force_expressions(_PARAMS)
        self.assertEqual(bl.findtext("x"), fx)
        self.assertEqual(bl.findtext("y"), fy)
        self.assertEqual(bl.findtext("z"), fz)

    def test_body_force_matches_fd_divergence(self):
        """Numerical MMS verification: rho * sign * expr == -div(sigma).

        div(sigma) is computed by second-order central differences of the
        stress field, which itself comes from central differences of u*
        — a chain fully independent of the closed-form coefficients.
        """
        p = dict(_PARAMS)
        rho = p["rho"]
        sign = mms._DEFAULTS["body_force_sign"]
        exprs = mms.body_force_expressions(p)
        rng = np.random.default_rng(42)
        L = p["L"]
        pts = 0.1 * L + 0.8 * L * rng.random((12, 3))
        h = 1e-4 * L
        scale = p["E"] * p["amplitude"]  # magnitude scale for tolerances
        for (x, y, z) in pts:
            div = np.zeros(3)
            for j, dv in enumerate(np.eye(3)):
                sp = _sigma(p, x + h * dv[0], y + h * dv[1], z + h * dv[2])
                sm = _sigma(p, x - h * dv[0], y - h * dv[1], z - h * dv[2])
                div += (sp[:, j] - sm[:, j]) / (2 * h)
            f_expected = -div  # applied body force per unit volume
            for i in range(3):
                emitted = _eval_expr(exprs[i], x, y, z)
                # emitted = sign * f / rho  =>  f = rho * emitted / sign
                f_emitted = rho * emitted / sign
                self.assertAlmostEqual(
                    f_emitted, f_expected[i],
                    delta=1e-6 * scale + 1e-6 * abs(f_expected[i]),
                    msg=f"component {i} at {(x, y, z)}")

    def test_body_force_matches_sympy_exact(self):
        """Symbolic re-derivation with sympy, evaluated at random points."""
        try:
            import sympy as sp
        except ImportError:
            self.skipTest("sympy not installed")
        p = dict(mms._DEFAULTS)
        p.update(_PARAMS)
        x, y, z = sp.symbols("x y z")
        gx, gy, gz = mms._wave_numbers(p)
        A, B, C = mms._amplitudes(p)
        lam, mu = mms._lame(p)
        u = sp.Matrix([
            A * sp.sin(gx * x) * sp.sin(gy * y) * sp.sin(gz * z),
            B * sp.cos(gx * x) * sp.cos(gy * y) * sp.cos(gz * z),
            C * sp.sin(gx * x) * sp.cos(gy * y) * sp.sin(gz * z),
        ])
        Xv = [x, y, z]
        grad = sp.Matrix(3, 3, lambda i, j: sp.diff(u[i], Xv[j]))
        eps = (grad + grad.T) / 2
        sig = lam * eps.trace() * sp.eye(3) + 2 * mu * eps
        f_sym = -sp.Matrix([sum(sp.diff(sig[i, j], Xv[j]) for j in range(3))
                            for i in range(3)])
        f_fun = sp.lambdify((x, y, z), f_sym, "numpy")
        exprs = mms.body_force_expressions(p)
        rho, sign = p["rho"], p["body_force_sign"]
        rng = np.random.default_rng(7)
        scale = p["E"] * p["amplitude"]
        for (px, py, pz) in p["L"] * rng.random((8, 3)):
            f_ref = np.asarray(f_fun(px, py, pz)).ravel()
            for i in range(3):
                f_emitted = rho * _eval_expr(exprs[i], px, py, pz) / sign
                self.assertAlmostEqual(
                    f_emitted, f_ref[i],
                    delta=1e-10 * scale + 1e-10 * abs(f_ref[i]))


class TestValidateParameters(unittest.TestCase):
    def _assert_rejects(self, message_part, **overrides):
        errors = mms.validate_parameters(dict(_PARAMS, **overrides))
        self.assertTrue(errors, f"expected rejection for {overrides}")
        self.assertTrue(any(message_part in e for e in errors),
                        f"{overrides} -> {errors}")

    def test_accepts_defaults_and_reference_params(self):
        self.assertEqual(mms.validate_parameters({}), [])
        self.assertEqual(mms.validate_parameters(dict(_PARAMS)), [])

    def test_rejects_bad_n(self):
        self._assert_rejects("n must be", n=1)
        self._assert_rejects("n must be", n=0)
        self._assert_rejects("n must be", n=2.5)
        self._assert_rejects("n must be", n="four")

    def test_rejects_bad_L(self):
        self._assert_rejects("L must be", L=0.0)
        self._assert_rejects("L must be", L=-1.0)

    def test_rejects_bad_E(self):
        self._assert_rejects("E must be", E=0.0)
        self._assert_rejects("E must be", E=-5.0)

    def test_rejects_bad_nu(self):
        self._assert_rejects("nu must", nu=0.5)
        self._assert_rejects("nu must", nu=0.7)
        self._assert_rejects("nu must", nu=-1.0)
        self._assert_rejects("nu must", nu=-2.0)

    def test_rejects_bad_rho(self):
        self._assert_rejects("rho must be", rho=0.0)

    def test_rejects_non_numeric_coefficients(self):
        for key in ("amplitude", "ax", "ay", "az", "kx", "ky", "kz",
                    "E", "nu", "L"):
            self._assert_rejects("must be a finite number",
                                 **{key: "abc"})
        self._assert_rejects("must be a finite number",
                             amplitude=float("nan"))

    def test_rejects_bad_steps_and_sign(self):
        self._assert_rejects("steps must be", steps=0)
        self._assert_rejects("body_force_sign", body_force_sign=2.0)

    def test_generator_raises_on_invalid(self):
        with self.assertRaises(ValueError):
            _gen(dict(_PARAMS, nu=0.5))
        with self.assertRaises(ValueError):
            _gen(dict(_PARAMS, n=1))


if __name__ == "__main__":
    unittest.main()
