"""Vector-valued interface exchange: the pieces of the generic coupling path
that assumed one scalar per interface point.

Each test here corresponds to something that was broken and is now not, or to a
discrimination that has to survive a rewrite. Nothing here runs a solver; the
end-to-end evidence is in scripts/tier2_fixtures/coupling/vector_pair_*.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.field_transfer import (InterfaceData, format_for_fenics,      # noqa: E402
                                 interpolate_to_points)
from core.quality_checks import (check_interface_balance,               # noqa: E402
                                 check_interface_flux_profile,
                                 interface_nodal_weights)


# ── the quadrature ──────────────────────────────────────────────────────────

def _tilted_patch(nu_: int, nv_: int) -> np.ndarray:
    u, v = np.meshgrid(np.linspace(0.0, 2.0, nu_), np.linspace(0.0, 3.0, nv_))
    u, v = u.ravel(), v.ravel()
    return np.column_stack([u, v, 0.5 * u + 0.25 * v])


PATCH_AREA = 6.0 * math.sqrt(1.0 + 0.25 + 0.0625)


def test_weights_reproduce_the_measure_of_a_line_in_2d():
    co = np.column_stack([np.full(11, 0.5), np.linspace(0.0, 1.0, 11)])
    w, dim, _ = interface_nodal_weights(co)
    assert dim == 1
    assert w.sum() == pytest.approx(1.0, rel=1e-12)


def test_weights_order_along_the_curve_not_along_a_coordinate_axis():
    """A lexicographic sort is correct only for an axis-aligned interface. On a
    diagonal one it still happens to work; on a curve it does not, and on a
    surface it is meaningless. The ordering is taken from the point cloud's own
    principal direction, so a 3-D diagonal line integrates exactly."""
    t = np.linspace(0.0, 1.0, 7)
    w, dim, _ = interface_nodal_weights(np.column_stack([t, 2 * t, 3 * t]))
    assert dim == 1
    assert w.sum() == pytest.approx(math.sqrt(14.0), rel=1e-12)


def test_weights_on_a_surface_give_its_area_not_a_snake_through_it():
    """THE DEFECT. The old rule sorted lexicographically and trapezoided the
    distances between consecutive points, which on a surface accumulates the
    length of a row-by-row snake — a quantity that GROWS with resolution and is
    unrelated to the area."""
    pts = _tilted_patch(9, 7)
    w, dim, _ = interface_nodal_weights(pts)
    assert dim == 2
    assert w.sum() == pytest.approx(PATCH_AREA, rel=1e-12)

    order = np.lexsort(tuple(pts[:, k] for k in range(2, -1, -1)))
    snake = float(np.linalg.norm(np.diff(pts[order], axis=0), axis=1).sum())
    assert snake > 5 * PATCH_AREA
    # and it gets worse with refinement, which no quadrature does
    fine = _tilted_patch(17, 13)
    order2 = np.lexsort(tuple(fine[:, k] for k in range(2, -1, -1)))
    snake2 = float(np.linalg.norm(np.diff(fine[order2], axis=0), axis=1).sum())
    assert snake2 > snake
    w2, _, _ = interface_nodal_weights(fine)
    assert w2.sum() == pytest.approx(PATCH_AREA, rel=1e-12)


def test_a_point_cloud_that_fills_a_volume_has_no_surface_integral():
    rng = np.random.default_rng(0)
    w, dim, why = interface_nodal_weights(rng.normal(size=(60, 3)))
    assert w is None and dim == 3 and "VOLUME" in why


def test_every_point_at_one_location_is_reported_not_integrated():
    w, dim, why = interface_nodal_weights([[1.0, 2.0]] * 5)
    assert w is None and dim == 0 and "same location" in why


# ── the balance check on a vector interface ─────────────────────────────────

def _traction(pts):
    u, v = pts[:, 0], pts[:, 1]
    return np.column_stack([1.0 + 0.30 * u - 0.20 * v,
                            0.4 - 0.15 * u + 0.35 * v])


def test_a_conservative_vector_exchange_on_a_surface_balances():
    a, b = _tilted_patch(9, 7), _tilted_patch(13, 11)
    ta, tb = _traction(a), -_traction(b)
    assert not check_interface_balance(
        {"coordinates": a.tolist(), "normal_fluxes": ta.tolist()},
        {"coordinates": b.tolist(), "normal_fluxes": tb.tolist()})
    # and the plain sums the old rule effectively compared do NOT agree
    assert abs(abs(tb.sum(axis=0)[0] / ta.sum(axis=0)[0]) - 1.0) > 0.5


def test_two_sides_with_the_same_point_count_but_different_spacing_integrate():
    """Equal point counts used to be taken as 'directly comparable', which is
    true only when the two sides sample the SAME points."""
    ya = np.linspace(0.0, 1.0, 9)
    yb = np.linspace(0.0, 1.0, 9) ** 3          # same count, different spacing
    ca = np.column_stack([np.full(9, 0.5), ya])
    cb = np.column_stack([np.full(9, 0.5), yb])
    fa = np.column_stack([1.0 + 5.0 * ya, 0.5 * ya])
    fb = -np.column_stack([1.0 + 5.0 * yb, 0.5 * yb])
    assert not check_interface_balance(
        {"coordinates": ca.tolist(), "normal_fluxes": fa.tolist()},
        {"coordinates": cb.tolist(), "normal_fluxes": fb.tolist()})
    assert abs(abs(fb.sum(axis=0)[0] / fa.sum(axis=0)[0]) - 1.0) > 0.1


def test_a_non_conservative_component_is_still_caught_on_a_surface():
    a, b = _tilted_patch(9, 7), _tilted_patch(13, 11)
    swapped = -_traction(b)[:, ::-1]      # totals alike, each component wrong
    w = check_interface_balance(
        {"coordinates": a.tolist(), "normal_fluxes": _traction(a).tolist()},
        {"coordinates": b.tolist(), "normal_fluxes": swapped.tolist()})
    assert w


def test_an_unintegrable_interface_is_reported_not_summed():
    """"Conservation was not evaluated" must never be the same output as
    "conservation was evaluated and is fine"."""
    rng = np.random.default_rng(1)
    vol = rng.normal(size=(50, 3))
    w = check_interface_balance(
        {"coordinates": vol.tolist(), "normal_fluxes": np.ones((50, 2)).tolist()},
        {"coordinates": vol[:30].tolist(),
         "normal_fluxes": (-np.ones((30, 2))).tolist()})
    assert w and "could NOT be evaluated" in w[0]


def test_the_scalar_balance_path_is_unchanged_without_coordinates():
    assert not check_interface_balance({"normal_fluxes": [1.0, 1.0]},
                                       {"normal_fluxes": [-1.0, -1.0]})
    bad = check_interface_balance({"normal_fluxes": [-100.0]},
                                  {"normal_fluxes": [0.1]})
    assert bad and "UNIT MISMATCH" in bad[0]


# ── the pointwise profile check ─────────────────────────────────────────────

def test_flux_profile_judges_each_component_on_its_own_scale():
    """One scale over the whole array is set by the largest component, so a
    tangential traction two orders below the normal one can be 100% wrong and
    read as 1% of "the interface scale"."""
    n = 9
    co = [[0.5, y] for y in np.linspace(0, 1, n)]
    fa = np.column_stack([np.full(n, 1.0e3), np.full(n, 5.0)])
    fb = np.column_stack([np.full(n, -1.0e3), np.full(n, +5.0)])   # y wrong
    findings, _ = check_interface_flux_profile(
        {"coordinates": co, "normal_fluxes": fa.tolist()},
        {"coordinates": co, "normal_fluxes": fb.tolist()})
    assert findings and "component [1]" in findings[0]
    # a component that is zero on both sides is not condemned on roundoff
    rng = np.random.default_rng(2)
    za = np.column_stack([np.full(n, 1.0e3), rng.normal(0, 3e-17, n)])
    zb = np.column_stack([np.full(n, -1.0e3), rng.normal(0, 3e-17, n)])
    ok, _ = check_interface_flux_profile(
        {"coordinates": co, "normal_fluxes": za.tolist()},
        {"coordinates": co, "normal_fluxes": zb.tolist()})
    assert ok == []


# ── interpolation and code generation ───────────────────────────────────────

def test_vector_interpolation_on_a_1d_interface():
    """np.interp takes only a 1-D `fp`, so this branch RAISED
    'object too deep for desired array' on every vector field — and a straight
    interface in 2-D is the commonest interface there is."""
    y = np.linspace(0.0, 1.0, 11)
    co = np.column_stack([np.full(11, 0.5), y])
    v = np.column_stack([y, 2.0 + y])
    tgt = np.column_stack([np.full(5, 0.5), np.linspace(0.05, 0.95, 5)])
    out = np.asarray(interpolate_to_points(InterfaceData(co, v, "u"), tgt), float)
    assert out.shape == (5, 2)
    assert out == pytest.approx(np.column_stack([tgt[:, 1], 2.0 + tgt[:, 1]]),
                                abs=1e-12)


def test_scalar_interpolation_still_returns_a_flat_array():
    y = np.linspace(0.0, 1.0, 11)
    co = np.column_stack([np.full(11, 0.5), y])
    tgt = np.column_stack([np.full(5, 0.5), np.linspace(0.05, 0.95, 5)])
    out = np.asarray(interpolate_to_points(InterfaceData(co, y.copy(), "t"), tgt),
                     float)
    assert out.shape == (5,)
    assert out == pytest.approx(tgt[:, 1], abs=1e-12)


def test_vector_interpolation_on_a_2d_interface_backfills_per_component():
    """The NaN backfill indexed a coordinate array with a 2-D boolean mask."""
    pts = _tilted_patch(7, 5)
    v = np.column_stack([pts[:, 0], pts[:, 1], pts[:, 0] - pts[:, 1]])
    tgt = pts[:3] + np.array([5.0, 5.0, 0.0])        # outside: forces the fill
    out = np.asarray(interpolate_to_points(InterfaceData(pts, v, "u"), tgt),
                     float)
    assert out.shape == (3, 3)
    assert np.all(np.isfinite(out))


def test_generated_fenics_snippet_is_blocked_space_aware_and_compiles():
    """The scalar snippet writes x.array[i] with i a NODE index, which on a
    blocked space fills component 0 and leaves the rest at zero."""
    y = np.linspace(0.0, 1.0, 5)
    co = np.column_stack([np.full(5, 0.5), y])
    vec = InterfaceData(co, np.column_stack([y, 2.0 + y]), "u")
    for bc in ("dirichlet", "neumann"):
        code = format_for_fenics(vec, bc, 0, 0.5)
        compile(code, "snippet", "exec")
        assert "_n * _bs + _c" in code
        assert "_ncomp = 2" in code
    scal = InterfaceData(co, y.copy(), "T")
    code = format_for_fenics(scal, "dirichlet", 0, 0.5)
    compile(code, "snippet", "exec")
    assert "_ncomp = 1" in code


# ── the monolithic comparison inside `couple` ───────────────────────────────

def test_monolithic_check_compares_a_vector_reference_per_component(tmp_path):
    """`.ravel()` made every vector reference unusable: with N points and 2
    components the flat size is 2N against N coordinates, so the shape guard
    concluded "not enough coordinates" and the strongest check in the tool
    reported NOT CHECKED on every vector coupling."""
    import json
    from tools.consolidated import _run_monolithic_check

    y = np.linspace(0.0, 1.0, 9)
    ref = {"field_name": "displacement", "n_points": 9,
           "coordinates": np.column_stack([np.full(9, 0.5), y]).tolist(),
           "values": np.column_stack([y, 2.0 * y]).tolist()}
    wd = tmp_path / "mono"
    wd.mkdir()
    script = tmp_path / "m.py"
    script.write_text("import json, pathlib\n"
                      f"pathlib.Path('monolithic.json').write_text({json.dumps(json.dumps(ref))})\n")
    spec = json.dumps({"command": [sys.executable, str(script)],
                       "work_dir": str(wd), "timeout": 120})

    yy = np.linspace(0.0, 1.0, 6)                      # a different sampling
    good = {"field_name": "displacement",
            "coordinates": np.column_stack([np.full(6, 0.5), yy]).tolist(),
            "values": np.column_stack([yy, 2.0 * yy]).tolist()}
    rep, findings, not_run = _run_monolithic_check(spec, {"A": good})
    assert rep["status"] == "checked"
    assert rep["reference_components"] == 2
    assert not findings
    assert not any("not enough coordinates" in n for n in not_run)

    # A y component that is 100% wrong while x is fine: the TOTAL relative L2
    # is dominated by x, so only a per-component comparison sees it.
    bad = {"field_name": "displacement",
           "coordinates": good["coordinates"],
           "values": np.column_stack([yy, 1.0e-4 * yy]).tolist()}
    _, findings, _ = _run_monolithic_check(spec, {"A": bad})
    assert findings and any("[1]" in f for f in findings)


def test_monolithic_check_refuses_to_compare_a_scalar_with_a_vector(tmp_path):
    import json
    from tools.consolidated import _run_monolithic_check

    y = np.linspace(0.0, 1.0, 9)
    ref = {"field_name": "displacement", "n_points": 9,
           "coordinates": np.column_stack([np.full(9, 0.5), y]).tolist(),
           "values": y.tolist()}                       # SCALAR reference
    wd = tmp_path / "mono"
    wd.mkdir()
    script = tmp_path / "m.py"
    script.write_text("import json, pathlib\n"
                      f"pathlib.Path('monolithic.json').write_text({json.dumps(json.dumps(ref))})\n")
    spec = json.dumps({"command": [sys.executable, str(script)],
                       "work_dir": str(wd), "timeout": 120})
    vec = {"field_name": "displacement",
           "coordinates": np.column_stack([np.full(9, 0.5), y]).tolist(),
           "values": np.column_stack([y, 2.0 * y]).tolist()}
    _, findings, not_run = _run_monolithic_check(spec, {"A": vec})
    assert not findings
    assert any("component" in n and "nothing to compare" in n for n in not_run)
