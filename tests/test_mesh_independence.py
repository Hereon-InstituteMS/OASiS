"""Tests for the automated mesh-independence study (verify_mesh_independence).

MMS convergence tests require a manufactured exact solution; application
problems have none, so the established recourse is the heuristic
mesh-refinement study: halve the discretisation length and accept the
solution only once global norms AND values at selected points stop
changing materially. src/core/mesh_independence.py implements the
comparison/verdict arithmetic; the verify_mesh_independence MCP tool
drives the re-runs through the normal backend layer.

Covered here:
  * template instantiation (placeholder substitution, refinement ladder)
  * the volume-weighted global L2 norm (constant-field exactness,
    refinement invariance, cell-family coverage, RMS fallback)
  * probe selection/interpolation and the near-zero probe floor
  * compare_levels verdict logic — including the NOT-converged path
  * end-to-end: the actual MCP tool runs a real scikit-fem problem at
    two resolutions and returns a verdict (converged AND not-converged),
    with the verification-gate stamp wired to the study outcome.
"""
from __future__ import annotations

import asyncio
import json
import sys
import unittest
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from core import mesh_independence as mi  # noqa: E402


def _quad_mesh(n: int):
    """Uniform n x n quad mesh of the unit square: (points, connectivity)."""
    xs = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])
    conn = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            conn.append([a, a + n + 1, a + n + 2, a + 1])
    return pts, np.array(conn)


def _tet_mesh_unit_cube(n: int):
    """Unit cube split into n^3 hex cells, each into 5 tets."""
    xs = np.linspace(0.0, 1.0, n + 1)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    def nid(i, j, k):
        return (i * (n + 1) + j) * (n + 1) + k

    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # VTK hexahedron corner ordering
                v = [nid(i, j, k), nid(i + 1, j, k),
                     nid(i + 1, j + 1, k), nid(i, j + 1, k),
                     nid(i, j, k + 1), nid(i + 1, j, k + 1),
                     nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
                for a, b, c, d in [(0, 1, 3, 4), (1, 2, 3, 6), (1, 3, 4, 6),
                                   (1, 4, 5, 6), (3, 4, 6, 7)]:
                    tets.append([v[a], v[b], v[c], v[d]])
    return pts, np.array(tets)


# ── template instantiation ───────────────────────────────────────────────


class TestSubstitution(unittest.TestCase):

    def test_replaces_every_occurrence(self):
        out = mi.substitute_resolution(
            "nx = __RESOLUTION__\nny = __RESOLUTION__\n", 16)
        self.assertEqual(out, "nx = 16\nny = 16\n")

    def test_integer_valued_floats_render_as_int(self):
        # MCP delivers `resolution` as float; `nx = 16.0` must not appear
        # where an element count is expected.
        self.assertEqual(mi.format_resolution(16.0), "16")
        self.assertEqual(mi.format_resolution(0.05), "0.05")

    def test_missing_token_raises(self):
        # Without the placeholder the study would re-run the identical
        # script and fake a converged verdict — must refuse loudly.
        with self.assertRaises(ValueError):
            mi.substitute_resolution("nx = 16", 32)


class TestRefinementLadder(unittest.TestCase):

    def test_divisions_multiply(self):
        self.assertEqual(mi.refinement_resolutions(16, 2.0, 2), [16, 32, 64])

    def test_size_divides(self):
        self.assertEqual(
            mi.refinement_resolutions(0.1, 2.0, 1, "size"), [0.1, 0.05])

    def test_at_least_one_refinement_required(self):
        with self.assertRaises(ValueError):
            mi.refinement_resolutions(16, 2.0, 0)

    def test_factor_must_refine(self):
        with self.assertRaises(ValueError):
            mi.refinement_resolutions(16, 1.0, 1)


# ── global norm ──────────────────────────────────────────────────────────


class TestGlobalL2(unittest.TestCase):

    def test_constant_field_on_unit_square_is_exact(self):
        pts, conn = _quad_mesh(8)
        u = np.full(len(pts), 3.0)
        norm, kind = mi.compute_global_l2(pts, [("quad", conn)], u)
        self.assertEqual(kind, "volume_weighted_l2")
        self.assertAlmostEqual(norm, 3.0, places=12)

    def test_refinement_invariance_smooth_field(self):
        # The SAME smooth field sampled on two refinements must give nearly
        # the same norm — that is the property the comparison relies on.
        norms = []
        for n in (8, 16):
            pts, conn = _quad_mesh(n)
            u = np.sin(np.pi * pts[:, 0]) * np.sin(np.pi * pts[:, 1])
            norm, _ = mi.compute_global_l2(pts, [("quad", conn)], u)
            norms.append(norm)
        self.assertLess(abs(norms[1] - norms[0]) / norms[1], 5e-3)

    def test_constant_field_on_unit_cube_tets(self):
        pts, tets = _tet_mesh_unit_cube(2)
        u = np.full(len(pts), 2.0)
        norm, kind = mi.compute_global_l2(pts, [("tetra", tets)], u)
        self.assertEqual(kind, "volume_weighted_l2")
        self.assertAlmostEqual(norm, 2.0, places=10)

    def test_vector_field_uses_magnitude(self):
        pts, conn = _quad_mesh(4)
        u = np.tile([3.0, 4.0], (len(pts), 1))  # |u| = 5 everywhere
        norm, _ = mi.compute_global_l2(pts, [("quad", conn)], u)
        self.assertAlmostEqual(norm, 5.0, places=12)

    def test_rms_fallback_without_supported_cells(self):
        pts = np.random.default_rng(0).random((10, 3))
        u = np.full(10, 4.0)
        norm, kind = mi.compute_global_l2(pts, [], u)
        self.assertEqual(kind, "rms_point")
        self.assertAlmostEqual(norm, 4.0, places=12)

    def test_surface_blocks_in_volume_mesh_are_ignored(self):
        # A 3-D result that also carries boundary triangles must integrate
        # over the volume only.
        pts, tets = _tet_mesh_unit_cube(2)
        bogus_tris = tets[:4, :3]
        u = np.full(len(pts), 2.0)
        norm, _ = mi.compute_global_l2(
            pts, [("triangle", bogus_tris), ("tetra", tets)], u)
        self.assertAlmostEqual(norm, 2.0, places=10)


# ── probes ───────────────────────────────────────────────────────────────


class TestProbes(unittest.TestCase):

    def test_default_probes_lie_in_bbox_and_include_hotspot(self):
        pts, _ = _quad_mesh(8)
        u = np.exp(-((pts[:, 0] - 0.75) ** 2 + (pts[:, 1] - 0.25) ** 2) / 0.01)
        probes = mi.default_probe_points(pts, u)
        self.assertGreaterEqual(len(probes), 3)
        hot = pts[int(np.argmax(u))]
        self.assertTrue(np.allclose(probes[0][:2], hot, atol=1e-12))
        for p in probes:
            self.assertTrue(0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0)

    def test_probe_interpolation_reproduces_linear_field(self):
        pts, _ = _quad_mesh(8)
        u = 2.0 * pts[:, 0] + 3.0 * pts[:, 1]
        vals = mi.probe_field(pts, u, [[0.5, 0.5], [0.25, 0.75]])
        self.assertAlmostEqual(vals[0], 2.5, places=10)
        self.assertAlmostEqual(vals[1], 2.75, places=10)


# ── comparison + verdict ─────────────────────────────────────────────────


def _level(res, l2, gmax, probes, qoi=None):
    d = {"resolution": res, "global_l2": l2, "global_max": gmax,
         "probe_values": probes}
    if qoi is not None:
        d["qoi"] = qoi
    return d


class TestCompareLevels(unittest.TestCase):

    def test_converged_when_all_changes_below_threshold(self):
        c = mi.compare_levels(
            [_level(8, 1.000, 2.000, [0.500, 1.900]),
             _level(16, 1.005, 2.004, [0.502, 1.905])], rel_tol=0.01)
        self.assertTrue(c["converged"])
        self.assertIn("CONVERGED", c["verdict"])
        self.assertEqual(c["failures"], [])
        self.assertEqual(len(c["steps"]), 1)

    def test_not_converged_names_the_failing_probe(self):
        c = mi.compare_levels(
            [_level(8, 1.000, 2.000, [0.50, 1.90]),
             _level(16, 1.002, 2.002, [0.60, 1.905])], rel_tol=0.01)
        self.assertFalse(c["converged"])
        self.assertIn("NOT CONVERGED", c["verdict"])
        self.assertTrue(any("probe 0" in f for f in c["failures"]))

    def test_not_converged_on_global_norm_alone(self):
        # Probes agreeing while the global norm drifts must still fail —
        # BOTH criteria are required.
        c = mi.compare_levels(
            [_level(8, 1.00, 2.000, [0.500]),
             _level(16, 1.10, 2.001, [0.5001])], rel_tol=0.01)
        self.assertFalse(c["converged"])
        self.assertTrue(any("L2" in f for f in c["failures"]))

    def test_verdict_uses_finest_step_only(self):
        # Large early change, small final change → the customary rule
        # accepts on the LAST halving.
        c = mi.compare_levels(
            [_level(8, 1.00, 2.00, [0.5]),
             _level(16, 1.20, 2.30, [0.6]),
             _level(32, 1.201, 2.301, [0.6005])], rel_tol=0.01)
        self.assertTrue(c["converged"])
        self.assertEqual(len(c["steps"]), 2)

    def test_near_zero_probe_does_not_produce_spurious_failure(self):
        # A probe in a dead region of the field: 1e-9 → 3e-9 is a 200%
        # naive relative change, but physically nothing moved (field scale
        # is 2.0). The floor = 1% of global max must absorb it.
        c = mi.compare_levels(
            [_level(8, 1.000, 2.000, [1e-9]),
             _level(16, 1.001, 2.001, [3e-9])], rel_tol=0.01)
        self.assertTrue(c["converged"])

    def test_qoi_from_summary_is_monitored(self):
        c = mi.compare_levels(
            [_level(8, 1.000, 2.000, [0.5], qoi={"tip_deflection": 1.00}),
             _level(16, 1.001, 2.001, [0.5], qoi={"tip_deflection": 1.10})],
            rel_tol=0.01)
        self.assertFalse(c["converged"])
        self.assertTrue(any("tip_deflection" in f for f in c["failures"]))

    def test_two_levels_required(self):
        with self.assertRaises(ValueError):
            mi.compare_levels([_level(8, 1.0, 2.0, [0.5])])


class TestCollectQoi(unittest.TestCase):

    def test_flattens_scalars_and_skips_nonfinite(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "results_summary.json").write_text(json.dumps({
                "max_temperature": 400.0,
                "nested": {"flux": 1.5},
                "label": "steel",           # non-numeric: skipped
                "bad": float("nan"),        # non-finite: skipped
                "per_node": [1, 2, 3],      # array: skipped
            }))
            qoi = mi.collect_qoi_scalars(td)
        self.assertEqual(qoi, {"max_temperature": 400.0, "nested.flux": 1.5})

    def test_missing_summary_gives_empty(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(mi.collect_qoi_scalars(td), {})

    def test_discretisation_descriptors_are_not_qois(self):
        # Live agent-validation finding (qwen campaign, scenario S1): a
        # summary carrying resolution/ndofs made the verdict fail on
        # "QoI 'ndofs' changed 74.61%" although every physical quantity
        # had settled. Descriptors that change under refinement BY
        # CONSTRUCTION must never be monitored.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "results_summary.json").write_text(json.dumps({
                "max_T": 2.69, "l2_norm": 2.15,          # physical QoIs
                "resolution": 32, "ndofs": 4225,          # descriptors
                "n_elements": 2048, "wall_time": 0.56,
                "mesh": {"n_cells": 2048, "nx": 32},
            }))
            qoi = mi.collect_qoi_scalars(td)
        self.assertEqual(set(qoi), {"max_T", "l2_norm"})

    def test_descriptor_qois_do_not_flip_the_verdict(self):
        # End-to-end on the comparison: physical quantities settled while
        # the descriptor entries differ wildly between the levels' summary
        # files -> still CONVERGED, because collect_qoi_scalars filtered
        # the descriptors out.
        import tempfile
        levels = []
        for res, ndofs in ((32, 4225), (64, 16641)):
            with tempfile.TemporaryDirectory() as td:
                (Path(td) / "results_summary.json").write_text(json.dumps({
                    "max_T": 2.6944, "resolution": res, "ndofs": ndofs}))
                qoi = mi.collect_qoi_scalars(td)
            levels.append(_level(res, 2.1518, 2.6944, [2.4135], qoi=qoi))
        c = mi.compare_levels(levels, rel_tol=0.01)
        self.assertTrue(c["converged"], c)


class TestExtractLevelMetrics(unittest.TestCase):

    def _write_vtu(self, td: Path, field="u", values=None):
        import meshio
        pts, conn = _quad_mesh(4)
        u = values if values is not None else pts[:, 0] * pts[:, 1]
        path = td / "result.vtu"
        meshio.Mesh(np.column_stack([pts, np.zeros(len(pts))]),
                    [("quad", conn)], point_data={field: u}).write(str(path))
        return path

    def test_reads_field_and_probes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = self._write_vtu(Path(td))
            m = mi.extract_level_metrics(path, field="u")
        self.assertEqual(m["field"], "u")
        self.assertEqual(m["norm_type"], "volume_weighted_l2")
        self.assertEqual(len(m["probe_points"]), len(m["probe_values"]))
        self.assertTrue(np.isfinite(m["global_l2"]))

    def test_missing_field_raises_with_available_names(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = self._write_vtu(Path(td))
            with self.assertRaises(ValueError) as cm:
                mi.extract_level_metrics(path, field="temperature")
        self.assertIn("u", str(cm.exception))

    def test_auto_selection_skips_vtk_metadata_arrays(self):
        # dolfinx VTKFile writes vtkOriginalPointIds/vtkGhostType alongside
        # the field; auto-selection must never monitor those.
        import meshio
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            pts, conn = _quad_mesh(4)
            path = Path(td) / "result.vtu"
            meshio.Mesh(
                np.column_stack([pts, np.zeros(len(pts))]), [("quad", conn)],
                point_data={
                    "vtkGhostType": np.zeros(len(pts)),
                    "z_temperature": pts[:, 0],
                }).write(str(path))
            m = mi.extract_level_metrics(path)
        self.assertEqual(m["field"], "z_temperature")

    def test_pyvista_fallback_reader_matches_meshio(self):
        # dolfinx's VTKFile emits VTU 2.2 with Lagrange cells, which meshio
        # rejects; the pyvista fallback must deliver the same geometry and
        # data for a file BOTH can read.
        try:
            import pyvista  # noqa: F401
        except ImportError:
            self.skipTest("pyvista not installed")
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            path = self._write_vtu(Path(td))
            pts_a, cells_a, pdata_a = mi.read_nodal_mesh(path)
            pts_b, cells_b, pdata_b = mi._read_with_pyvista(path)
        self.assertEqual(pts_a.shape[0], pts_b.shape[0])
        self.assertIn("u", pdata_b)
        np.testing.assert_allclose(np.sort(pdata_a["u"]),
                                   np.sort(pdata_b["u"]), rtol=1e-12)
        fam_b, conn_b = cells_b[0]
        self.assertEqual(fam_b, "quad")
        self.assertEqual(conn_b.shape[1], 4)
        norm_a, _ = mi.compute_global_l2(pts_a, cells_a, pdata_a["u"])
        norm_b, _ = mi.compute_global_l2(pts_b, cells_b, pdata_b["u"])
        self.assertAlmostEqual(norm_a, norm_b, places=12)

    def test_nonfinite_field_is_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            pts, _ = _quad_mesh(4)
            bad = np.full(len(pts), np.nan)
            path = self._write_vtu(Path(td), values=bad)
            with self.assertRaises(ValueError):
                mi.extract_level_metrics(path, field="u")


# ── end-to-end through the real MCP tool on a real backend ───────────────


class _StubMCP:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco

    def resource(self, *args, **kwargs):
        def deco(fn):
            return fn
        return deco

    def prompt(self, *args, **kwargs):
        def deco(fn):
            return fn
        return deco


# Test-problem template: Poisson with a peaked interior source on the unit
# square — no closed-form solution, the mesh-independence use case. The
# discretisation parameter is the __RESOLUTION__ placeholder (repo rule:
# templates carry placeholders, never hard-coded discretisations).
_SKFEM_TEMPLATE = '''"""Poisson, peaked interior source, homogeneous Dirichlet."""
import numpy as np
import meshio
from skfem import MeshTri, ElementTriP1, Basis, asm, solve, condense, LinearForm
from skfem.models.poisson import laplace

n = __RESOLUTION__
m = MeshTri.init_tensor(np.linspace(0.0, 1.0, n + 1),
                        np.linspace(0.0, 1.0, n + 1))
basis = Basis(m, ElementTriP1())


@LinearForm
def load(v, w):
    x, y = w.x
    return np.exp(-((x - 0.6) ** 2 + (y - 0.4) ** 2) / 0.02) * v


K = asm(laplace, basis)
f = asm(load, basis)
D = basis.get_dofs().flatten()
u = solve(*condense(K, f, D=D))
meshio.Mesh(np.column_stack([m.p.T, np.zeros(m.p.shape[1])]),
            [("triangle", m.t.T)], point_data={"u": u}).write("result.vtu")
'''


class TestVerifyMeshIndependenceE2E(unittest.TestCase):
    """Drive the ACTUAL tool against the scikit-fem backend."""

    @classmethod
    def setUpClass(cls):
        from core.registry import load_all_backends, get_backend
        from tools.consolidated import register_consolidated_tools
        load_all_backends()
        backend = get_backend("skfem")
        status, msg = backend.check_availability()
        if status.value != "available":
            raise unittest.SkipTest(f"skfem not available: {msg}")
        stub = _StubMCP()
        register_consolidated_tools(stub)
        cls.tool = staticmethod(stub.tools["verify_mesh_independence"])

    def _run(self, **kw):
        out = asyncio.run(self.tool(**kw))
        return json.loads(out)

    def test_registered_with_critic_approved(self):
        import inspect
        params = inspect.signature(self.tool).parameters
        self.assertIn("critic_approved", params)
        self.assertIn("rel_tol", params)

    def test_resolved_problem_converges(self):
        d = self._run(solver="skfem", input_template=_SKFEM_TEMPLATE,
                      resolution=32, job_name="test_meshcheck_converged",
                      critic_approved=True)
        self.assertEqual(d["status"], "completed")
        self.assertTrue(d["converged"], d)
        self.assertIn("CONVERGED", d["verdict"])
        self.assertEqual(len(d["levels"]), 2)
        self.assertEqual([lv["resolution"] for lv in d["levels"]], [32, 64])
        step = d["refinement_steps"][0]
        self.assertLess(step["global_l2_rel_change"], 0.01)
        self.assertLess(max(step["probe_rel_changes"]), 0.01)
        # the study passed AND the critic approved → verified result
        self.assertTrue(d["trustworthy_result"])

    def test_underresolved_problem_is_not_converged(self):
        d = self._run(solver="skfem", input_template=_SKFEM_TEMPLATE,
                      resolution=3, job_name="test_meshcheck_notconverged",
                      critic_approved=True)
        self.assertEqual(d["status"], "completed")
        self.assertFalse(d["converged"], d)
        self.assertIn("NOT CONVERGED", d["verdict"])
        self.assertTrue(d["failures"])
        # gate: a mesh-dependent solution is never a trustworthy result
        self.assertFalse(d["trustworthy_result"])
        self.assertIn("NOT VERIFIED", d["verification"])

    def test_directory_named_like_result_file_is_ignored(self):
        # Live agent-validation finding (qwen campaign, scenario S1):
        # dolfinx VTXWriter emits a DIRECTORY named *.vtu; the level's
        # result-file pick must skip directories or the study dies on
        # "unreadable by meshio: Is a directory". The decoy's stem ends in
        # a digit so sorted_by_step would rank it LAST (i.e. pick it)
        # if directories were not filtered out.
        template = _SKFEM_TEMPLATE.replace(
            'point_data={"u": u}).write("result.vtu")',
            'point_data={"u": u}).write("result.vtu")\n'
            'import os\nos.makedirs("zzz9.vtu", exist_ok=True)')
        d = self._run(solver="skfem", input_template=template,
                      resolution=8, job_name="test_meshcheck_decoy_dir",
                      critic_approved=True)
        self.assertEqual(d["status"], "completed", d)
        for lv in d["levels"]:
            self.assertEqual(lv["result_file"], "result.vtu")

    def test_template_without_placeholder_is_refused(self):
        d = self._run(solver="skfem",
                      input_template=_SKFEM_TEMPLATE.replace(
                          "__RESOLUTION__", "16"),
                      resolution=16, job_name="test_meshcheck_noplaceholder")
        self.assertEqual(d["status"], "failed")
        self.assertIn("__RESOLUTION__", d["error"])
        self.assertFalse(d["trustworthy_result"])


if __name__ == "__main__":
    unittest.main()
