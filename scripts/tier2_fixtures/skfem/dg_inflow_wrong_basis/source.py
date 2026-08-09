"""Tier-2: an inflow BC assembled over InteriorFacetBasis instead of FacetBasis.

Claim: skfem dg_methods#2 -- FacetBasis assembles over BOUNDARY facets (not
interior); it is what inflow/outflow flux BCs belong on.  Applying an inflow BC
by adding the term over InteriorFacetBasis instead silently adds the inflow to
every interior edge, so the solution picks up a spurious source distributed
across the mesh interior; a pure-Dirichlet steady advection with the wrong
basis shows a non-monotone solution.

Wrong variant: assemble the inflow load ``-min(b.n, 0) * g * v`` over
``InteriorFacetBasis`` while the (well-posed) inflow bilinear term stays on
``FacetBasis`` -- i.e. exactly the one-character basis mix-up.  The matrix is
still non-singular, so nothing warns.

Test problem: pure advection b = (1, 0.5), f = 0, inflow datum g = 1 on the
whole inflow boundary.  The exact solution is u == 1 and upwind DG reproduces
it to machine precision, so ANY departure is the spurious source.

Observed on skfem 12.0.1 (MeshTri().refined(3), 176 interior facets):
  * FacetBasis (right)          -> max|u - 1| = 2.2e-15, u in [1, 1]
  * InteriorFacetBasis (wrong)  -> u in [-0.67, +16.9], max|u - 1| = 15.9,
    and the streamwise band means are non-monotone.

Mutation control: T2_MUTATE=1 moves the wrong-variant inflow load back onto
FacetBasis (load_on = fb) -- the documented fix, at the one line where the basis
is chosen. The "wrong" leg then reproduces u == 1, so n_interior_facets=176,
wrong_max_abs_dev_from_one_gt_1=True, wrong_violates_max_principle=True,
wrong_streamwise_decreases_gt_0=True, wrong_band_growth_gt_5=True and
wrong_over_right_dev_ratio_gt_1e10=True all vanish and the fixture goes red.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

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
G_INFLOW = 1.0


@BilinearForm
def advection_volume(u, v, w):
    return (B[0] * u.grad[0] + B[1] * u.grad[1]) * v


@BilinearForm
def upwind_flux(u, v, w):
    bn = B[0] * w.n[0] + B[1] * w.n[1]
    sv = (-1.0) ** w.idx[1]
    ju = (-1.0) ** w.idx[0] * u
    return np.minimum(sv * bn, 0.0) * (-sv) * ju * v


@BilinearForm
def inflow_bilinear(u, v, w):
    bn = B[0] * w.n[0] + B[1] * w.n[1]
    return -np.minimum(bn, 0.0) * u * v


@LinearForm
def inflow_load(v, w):
    bn = B[0] * w.n[0] + B[1] * w.n[1]
    return -np.minimum(bn, 0.0) * G_INFLOW * v


def n_streamwise_decreases(u, ib):
    """Sample u along two streamlines x(t) = x0 + b t and count decreases.

    The exact solution is the constant 1, so it must not decrease anywhere.
    """
    interp = ib.interpolator(u)
    t = np.linspace(0.0, 0.55, 40)
    n = 0
    for y0 in (0.02, 0.3):
        vals = np.asarray(interp(np.vstack([0.02 + B[0] * t, y0 + B[1] * t])))
        n += int(np.sum(np.diff(vals) < -1e-9))
    return n


def streamwise_band_growth(u, doflocs, nbin=6):
    """Ratio of last to first streamwise band mean; 1.0 for the exact answer."""
    s = B[0] * doflocs[0] + B[1] * doflocs[1]
    edges = np.linspace(s.min(), s.max(), nbin + 1)
    means = np.array([u[(s >= edges[k]) & (s <= edges[k + 1])].mean()
                      for k in range(nbin)])
    return float(means[-1] / means[0]), means


def run(load_basis: str):
    m = MeshTri().refined(3)
    e = ElementDG(ElementTriP1())
    ib = Basis(m, e)
    i0 = InteriorFacetBasis(m, e, side=0)
    i1 = InteriorFacetBasis(m, e, side=1)
    fb = FacetBasis(m, e)

    A = (asm(advection_volume, ib)
         + asm(upwind_flux, [i0, i1], [i0, i1])
         + asm(inflow_bilinear, fb))
    # Pathology: the inflow load is assembled on an InteriorFacetBasis.
    # Mutation: the documented fix -- put the inflow load back on FacetBasis.
    load_on = fb if (load_basis == "facet" or MUTATE) else i0
    f = asm(inflow_load, load_on)
    u = spsolve(A.tocsr(), f)
    return u, ib, m, int(load_on.nelems)


def main() -> int:
    ok = True

    # --- RIGHT variant: inflow load on FacetBasis (boundary facets) --------
    u_ok, ib_ok, m, n_bnd = run("facet")
    dev_ok = float(np.abs(u_ok - 1.0).max())
    dec_ok = n_streamwise_decreases(u_ok, ib_ok)
    growth_ok, _ = streamwise_band_growth(u_ok, ib_ok.doflocs)
    print(f"n_boundary_facets={n_bnd}")
    print(f"right_max_abs_dev_from_one_lt_1e-10={dev_ok < 1e-10}")
    print(f"right_respects_max_principle={bool(u_ok.min() > 1 - 1e-10 and u_ok.max() < 1 + 1e-10)}")
    print(f"right_streamwise_decreases={dec_ok}")
    print(f"right_band_growth_is_one={bool(abs(growth_ok - 1.0) < 1e-10)}")
    print(f"right_dev={dev_ok:.3e}")
    if dev_ok >= 1e-10 or dec_ok != 0:
        print("FAIL: the correct FacetBasis inflow did not reproduce u == 1",
              file=sys.stderr)
        ok = False

    # --- WRONG variant: same load assembled on InteriorFacetBasis ----------
    u_bad, ib_bad, _, n_int = run("interior")
    dev_bad = float(np.abs(u_bad - 1.0).max())
    dec_bad = n_streamwise_decreases(u_bad, ib_bad)
    growth_bad, means_bad = streamwise_band_growth(u_bad, ib_bad.doflocs)
    print(f"n_interior_facets={n_int}")
    print(f"wrong_solution_is_finite={bool(np.isfinite(u_bad).all())}")
    print(f"wrong_max_abs_dev_from_one_gt_1={dev_bad > 1.0}")
    print(f"wrong_violates_max_principle={bool(u_bad.min() < -0.1 or u_bad.max() > 2.0)}")
    print(f"wrong_streamwise_decreases_gt_0={dec_bad > 0}")
    print(f"wrong_band_growth_gt_5={growth_bad > 5.0}")
    print(f"wrong_band_means={np.round(means_bad, 3).tolist()}")
    print(f"wrong_dev={dev_bad:.3e} wrong_min={u_bad.min():.4f} wrong_max={u_bad.max():.4f}")
    print(f"wrong_over_right_dev_ratio_gt_1e10={dev_bad / max(dev_ok, 1e-300) > 1e10}")
    if not np.isfinite(u_bad).all():
        print("FAIL: the wrong-basis system was not solvable, so nothing was silent",
              file=sys.stderr)
        ok = False
    if dev_bad <= 1.0:
        print("FAIL: the interior-facet inflow load did not corrupt the solution",
              file=sys.stderr)
        ok = False
    if dec_bad <= 0:
        print("FAIL: the wrong-basis solution stayed monotone streamwise",
              file=sys.stderr)
        ok = False
    if n_int <= n_bnd:
        print("FAIL: InteriorFacetBasis did not cover more facets than FacetBasis",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
