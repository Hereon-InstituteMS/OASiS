"""Regression tests for the anti-fabrication checks.

Every test corresponds to a forgery that was demonstrated to pass an earlier
version of these checks, or to an honest case that an earlier version wrongly
refused. They exist so those routes cannot silently reopen.
"""
from __future__ import annotations

import numpy as np
import pytest

skfem = pytest.importorskip("skfem")

from core.fabrication_gate import (check_field_sanity, check_mesh_sanity,
                                   check_probe_coverage, select_artefact,
                                   select_field)
from core.residual_check import check_residual


# ── fixtures ──────────────────────────────────────────────────────────────
def unit_square(refine: int = 4):
    m = skfem.MeshTri().refined(refine)
    return m.p.T.copy(), [("triangle", m.t.T.copy())], m


def poisson_source(x):
    return 2 * np.pi ** 2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def exact(px):
    return np.sin(np.pi * px[:, 0]) * np.sin(np.pi * px[:, 1])


def genuine_solution(m):
    from skfem import Basis, ElementTriP1, asm, condense, solve, BilinearForm, LinearForm
    from skfem.helpers import dot, grad
    basis = Basis(m, ElementTriP1())

    @BilinearForm
    def a(u, v, w):
        return dot(grad(u), grad(v))

    @LinearForm
    def L(v, w):
        return poisson_source(w.x) * v

    A, b = asm(a, basis), asm(L, basis)
    return solve(*condense(A, b, D=basis.get_dofs()))


# ── the discriminator: does the field satisfy the equations? ──────────────
def test_genuine_solve_satisfies_the_discrete_equations():
    p, c, m = unit_square(5)
    v = check_residual(p, c, genuine_solution(m), dim=2,
                       source_fn=poisson_source, domain_measure_expected=1.0)
    assert v.supported and v.solver_like
    assert v.relative_residual < 1e-8


def test_exact_solution_sampled_at_nodes_is_not_a_solve():
    """The most seductive forgery: perfectly accurate, and still not a solve.

    An analytic field misses the discrete system by the truncation error, which
    is orders of magnitude above solver tolerance.
    """
    p, c, m = unit_square(5)
    v = check_residual(p, c, exact(p), dim=2, source_fn=poisson_source,
                       domain_measure_expected=1.0)
    assert v.supported and not v.solver_like
    assert v.relative_residual > 1e-6


def test_perturbed_analytic_field_is_not_a_solve():
    """The eight-line fabricator: analytic field plus an h-dependent wobble.
    It is MORE accurate than the genuine solve and passes a mesh-independence
    study, so accuracy cannot be the discriminator."""
    p, c, m = unit_square(5)
    vals = exact(p) * (1 - 0.9 * (1 / 32) ** 2)
    v = check_residual(p, c, vals, dim=2, source_fn=poisson_source,
                       domain_measure_expected=1.0)
    assert v.supported and not v.solver_like


def test_constant_field_is_not_a_solve():
    p, c, m = unit_square(4)
    v = check_residual(p, c, np.full(len(p), 0.5), dim=2,
                       source_fn=poisson_source, domain_measure_expected=1.0)
    assert not v.solver_like


def test_wrong_domain_is_refused_by_residual_check():
    p, c, m = unit_square(4)
    v = check_residual(p * 0.5, c, genuine_solution(m), dim=2,
                       source_fn=poisson_source, domain_measure_expected=1.0)
    assert v.relative_residual is None and not v.solver_like
    assert "domain" in v.detail


# ── mesh sanity ───────────────────────────────────────────────────────────
def test_four_node_mesh_has_no_interior_and_is_refused():
    """Four nodes and one constant cover the unit square exactly and can hit
    any target value. Geometrically valid, so only the interior-node
    requirement catches it."""
    p = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], float)
    c = [("triangle", np.array([[0, 1, 2], [1, 3, 2]]))]
    v = check_mesh_sanity(p, c, dim=2, domain_measure_expected=1.0)
    assert not v.accepted and "interior" in v.reasons[0]


def test_scaled_coordinates_are_refused():
    p, c, _ = unit_square(4)
    v = check_mesh_sanity(p * 1e-6, c, dim=2, domain_measure_expected=1.0)
    assert not v.accepted and "domain" in v.reasons[0]


def test_degenerate_cell_is_refused():
    p, c, m = unit_square(4)
    cells = [("triangle", np.vstack([m.t.T, [[0, 0, 0]]]))]
    v = check_mesh_sanity(p, cells, dim=2)
    assert not v.accepted and "degenerate" in v.reasons[0]


def test_unreferenced_node_is_refused():
    p, c, _ = unit_square(4)
    v = check_mesh_sanity(np.vstack([p, [[9, 9]]]), c, dim=2)
    assert not v.accepted and "not referenced" in v.reasons[0]


def test_honest_mesh_is_accepted():
    p, c, _ = unit_square(4)
    v = check_mesh_sanity(p, c, dim=2, domain_measure_expected=1.0)
    assert v.accepted and v.checks["n_interior_nodes"] > 0


# ── field sanity ──────────────────────────────────────────────────────────
def test_partially_nan_field_is_refused():
    """A norm can come out finite while most nodes are NaN."""
    p, _, _ = unit_square(4)
    vals = np.ones(len(p))
    vals[3:8] = np.nan
    v = check_field_sanity(vals, n_points=len(p))
    assert not v.accepted and "not finite" in v.reasons[0]


def test_finite_field_is_accepted():
    p, _, _ = unit_square(4)
    assert check_field_sanity(np.ones(len(p)), n_points=len(p)).accepted


# ── selection: never guess, never substitute ──────────────────────────────
def test_missing_requested_field_is_not_substituted():
    name, why = select_field({"tiny_helper": 1, "u": 2}, "error")
    assert name is None and "does not substitute" in why


def test_vtk_metadata_arrays_are_never_graded():
    name, _ = select_field({"vtkGhostType": 1, "u": 2}, "")
    assert name == "u"


def test_ambiguous_field_is_refused():
    name, why = select_field({"u": 1, "p": 2}, "")
    assert name is None and "does not guess" in why


def test_multiple_artefacts_require_an_explicit_choice(tmp_path):
    """A decoy named to sort first once pre-empted the genuine result, and an
    honest time series was graded at its initial condition."""
    for n in ("a_checkpoint.vtu", "solution.vtu"):
        (tmp_path / n).write_text("x")
    got, why = select_artefact(sorted(tmp_path.glob("*.vtu")), None)
    assert got is None and "explicitly" in why


def test_single_artefact_is_unambiguous(tmp_path):
    (tmp_path / "solution.vtu").write_text("x")
    got, _ = select_artefact(sorted(tmp_path.glob("*.vtu")), None)
    assert got is not None


# ── probe coverage: a bounding box is not coverage ────────────────────────
def test_slivers_spanning_the_bounding_box_are_refused():
    p = np.array([[0, 0], [.05, 0], [0, .05], [.95, .95], [1, .95], [.95, 1]], float)
    c = [("triangle", np.array([[0, 1, 2], [3, 4, 5]]))]
    v = check_probe_coverage(p, c, [[0.5, 0.5]], dim=2)
    assert not v.accepted and "no cell" in v.reasons[0]


def test_one_dangling_node_does_not_reopen_the_hole():
    p, c, _ = unit_square(4)
    v = check_probe_coverage(np.vstack([p, [[10, 10]]]), c, [[5, 5]], dim=2)
    assert not v.accepted


def test_probe_inside_the_mesh_is_accepted():
    p, c, _ = unit_square(4)
    assert check_probe_coverage(p, c, [[0.5, 0.5]], dim=2).accepted
