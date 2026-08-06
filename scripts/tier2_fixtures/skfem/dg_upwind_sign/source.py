"""Tier-2: getting the upwind side wrong destroys the DG flux's dissipation.

Claim: skfem dg_methods#3 -- the upwind flux is bn * u_upwind * [v] and the
upwind side must be identified from the sign of b.n.  A wrong upwind/downwind
choice gives a centered flux that is unconditionally unstable for pure
advection.

Wrong variants, both a one-token edit of the correct form
``np.minimum(sv*bn, 0.0)``:
  A) ``0.5*sv*bn``            -- centered flux, sign of b.n never consulted
  B) ``np.maximum(sv*bn, 0.0)`` -- the sign flipped, i.e. the DOWNWIND side

The exact, mesh-independent diagnostic is the sign of the quadratic form
z^T F z of the interior-facet flux matrix F: upwinding must dissipate.

Observed on skfem 12.0.1 (MeshTri().refined(3)):
  * upwind   -> z^T F z > 0 for every random probe (dissipative)
  * centered -> z^T F z takes BOTH signs and is ~30x smaller in magnitude:
    the flux adds no net dissipation at all
  * downwind -> z^T F z < 0 for every probe (anti-dissipative, energy
    production); the pure-advection matrix is then EXACTLY SINGULAR and
    spsolve returns NaN, and with a small mass shift added to make it solvable
    the solution blows up to O(1e12) instead of staying in [0, 1].
  * on the front-advection test the centered flux overshoots ~2x more than
    upwind and its wiggles persist ~1e3 further downstream.

Mutation control: ``T2_MUTATE=1 python source.py`` applies the documented
one-token fix at the pathology site -- inside ``flux_coefficient`` the
"centered" (``0.5*sv*bn``) and "downwind" (``np.maximum(sv*bn, 0.0)``) branches
are replaced by the correct upwind coefficient ``np.minimum(sv*bn, 0.0)``.  All
three variants are then the same dissipative operator, so the anti-dissipation,
the singular matrix and the overshoot contrast all disappear and the fixture
goes red.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementDG,
    ElementTriP1,
    FacetBasis,
    InteriorFacetBasis,
    LinearForm,
    MeshTri,
    asm,
)
from scipy.sparse.linalg import spsolve

MUTATE = os.environ.get("T2_MUTATE") == "1"

B = np.array([1.0, 0.5])


def flux_coefficient(kind, sv, bn):
    if MUTATE and kind in ("centered", "downwind"):
        # The documented fix: consult the sign of b.n and take the upwind side.
        kind = "upwind"
    if kind == "upwind":
        return np.minimum(sv * bn, 0.0)
    if kind == "centered":
        return 0.5 * sv * bn
    if kind == "downwind":
        return np.maximum(sv * bn, 0.0)
    raise AssertionError(kind)


def make_flux_form(kind):
    @BilinearForm
    def form(u, v, w):
        bn = B[0] * w.n[0] + B[1] * w.n[1]
        sv = (-1.0) ** w.idx[1]
        ju = (-1.0) ** w.idx[0] * u
        return flux_coefficient(kind, sv, bn) * (-sv) * ju * v
    return form


@BilinearForm
def advection_volume(u, v, w):
    return (B[0] * u.grad[0] + B[1] * u.grad[1]) * v


@BilinearForm
def mass(u, v, w):
    return u * v


@BilinearForm
def inflow_bilinear(u, v, w):
    bn = B[0] * w.n[0] + B[1] * w.n[1]
    return -np.minimum(bn, 0.0) * u * v


def inflow_load(gfun):
    @LinearForm
    def form(v, w):
        bn = B[0] * w.n[0] + B[1] * w.n[1]
        return -np.minimum(bn, 0.0) * gfun(w.x) * v
    return form


def g_front(x):
    return np.where(x[1] < 0.5, 1.0, 0.0)


def flux_matrix(m, kind):
    e = ElementDG(ElementTriP1())
    i0 = InteriorFacetBasis(m, e, side=0)
    i1 = InteriorFacetBasis(m, e, side=1)
    return asm(make_flux_form(kind), [i0, i1], [i0, i1])


def solve_front(m, kind, shift=0.0):
    e = ElementDG(ElementTriP1())
    ib = Basis(m, e)
    fb = FacetBasis(m, e)
    A = (asm(advection_volume, ib)
         + flux_matrix(m, kind)
         + asm(inflow_bilinear, fb))
    if shift:
        A = A + shift * asm(mass, ib)
    f = asm(inflow_load(g_front), fb)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        u = spsolve(A.tocsr(), f)
        msgs = [str(c.message) for c in caught]
    return u, ib, msgs


def main() -> int:
    ok = True
    m = MeshTri().refined(3)
    rng = np.random.default_rng(0)

    # --- dissipation sign of the interior-facet flux operator --------------
    signs = {}
    for kind in ("upwind", "centered", "downwind"):
        F = flux_matrix(m, kind)
        vals = np.array([float(z @ (F @ z))
                         for z in rng.standard_normal((8, F.shape[0]))])
        signs[kind] = vals
        print(f"{kind}_quadform_all_positive={bool((vals > 0).all())}")
        print(f"{kind}_quadform_all_negative={bool((vals < 0).all())}")
        print(f"{kind}_quadform_max_abs={np.abs(vals).max():.4e}")

    if not (signs["upwind"] > 0).all():
        print("FAIL: the upwind flux operator was not dissipative", file=sys.stderr)
        ok = False
    if not (signs["downwind"] < 0).all():
        print("FAIL: the sign-flipped (downwind) flux was not anti-dissipative",
              file=sys.stderr)
        ok = False
    mixed = bool((signs["centered"] > 0).any() and (signs["centered"] < 0).any())
    print(f"centered_quadform_indefinite={mixed}")
    ratio = float(np.abs(signs["upwind"]).min() / np.abs(signs["centered"]).max())
    print(f"upwind_dissipation_exceeds_centered_by_gt_10x={ratio > 10.0}")
    if not mixed or ratio <= 10.0:
        print("FAIL: the centered flux still supplied net dissipation",
              file=sys.stderr)
        ok = False

    # --- WRONG variant B: sign flipped -> singular / geometric blow-up -----
    u_dn, ib_dn, msgs_dn = solve_front(m, "downwind")
    print(f"downwind_all_nan={bool(np.isnan(u_dn).all())}")
    print(f"downwind_warning={msgs_dn!r}")
    if not np.isnan(u_dn).all():
        print("FAIL: the downwind flux gave a finite solution", file=sys.stderr)
        ok = False

    u_dns, _, _ = solve_front(m, "downwind", shift=1e-2)
    print(f"downwind_with_mass_shift_max_abs_gt_1e6={bool(np.abs(u_dns).max() > 1e6)}")
    print(f"downwind_with_mass_shift_max_abs={np.abs(u_dns).max():.4e}")
    if np.abs(u_dns).max() <= 1e6:
        print("FAIL: the downwind flux did not blow up geometrically",
              file=sys.stderr)
        ok = False

    # --- WRONG variant A vs RIGHT variant on the advected front -----------
    over = {}
    for kind in ("upwind", "centered"):
        u, ib, _ = solve_front(m, kind)
        over[kind] = max(float(u.max() - 1.0), float(-u.min()))
        print(f"{kind}_front_overshoot={over[kind]:.4f}")
    print(f"centered_overshoot_exceeds_upwind={over['centered'] > over['upwind']}")
    if over["centered"] <= over["upwind"]:
        print("FAIL: the centered flux did not overshoot more than upwind",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
