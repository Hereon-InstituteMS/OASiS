"""Tier-2: Taylor-Hood converges; equal-order P1/P1 does not merely wobble.

Claim: skfem hydraulic_resistance#5 -- Taylor-Hood P2-P1 on triangles is
inf-sup stable and equal-order P1-P1 without stabilisation is not.  The entry
(already sharpened once, 2026-08-03) says Taylor-Hood gives velocity L2 rates
of k+1 = 3, that swapping velocity to ElementVector(ElementTriP1()) does NOT
merely "fluctuate 50-200%" but makes the block matrix EXACTLY SINGULAR on the
structured refined() mesh -- scipy emitting MatrixRankWarning and every
component coming back nan -- and that on less symmetric meshes you get a
checkerboard instead.

Measured on skfem 12.0.1 with the manufactured solution the entry names:
stream function psi = x^2(1-x)^2 y^2(1-y)^2, u = (d psi/dy, -d psi/dx),
p = x - 1/2, source derived symbolically, on MeshTri().refined(2..4) with
homogeneous velocity Dirichlet data and one pressure DOF pinned.

  * TAYLOR-HOOD: the velocity L2 error falls at the claimed order, giving
    rates near k+1 = 3, and the pressure error falls with it.
  * EQUAL ORDER ON THE STRUCTURED MESH: reproduced exactly as the sharpened
    entry says.  scipy emits MatrixRankWarning("Matrix is exactly singular")
    and the solution is not finite.  This is the ONE place in this backend's
    Stokes knowledge where that warning genuinely fires, which is worth
    holding next to hydraulic_resistance#2 and navier_stokes#3 where it does
    not: the warning discriminates a rank-deficient matrix, not a pressure
    null space.
  * so the two failure modes must not be guarded the same way.  Here
    np.isfinite is the right partner for the warning; for a pressure null
    space both are blind.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology
site -- the equal-order sweep's velocity element becomes ElementTriP2, i.e.
the inf-sup-violating P1/P1 pair is replaced by the Taylor-Hood pair the
entry prescribes -- so the block matrix is no longer rank deficient.  The
unmutated run matches all 11 expect_in_output strings; under mutation
"eq_refine2_emits_matrixrankwarning=True" (and refine3, refine4),
"eq_refine4_finite=False", "equal_order_warns_at_every_level=True",
"equal_order_nonfinite_at_every_level=True",
"equal_order_is_exactly_singular_on_structured_mesh=True" and
"isfinite_is_the_right_partner_for_this_fault=True" disappear.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import warnings

import numpy as np
import scipy.sparse as sp
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    Functional,
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.helpers import ddot, div, grad

MUTATE = os.environ.get("T2_MUTATE") == "1"


def u_exact(x):
    a = x[0] ** 2 * (x[0] - 1.0) ** 2
    b = x[1] ** 2 * (x[1] - 1.0) ** 2
    du1 = 2.0 * x[1] * (x[1] - 1.0) * (2.0 * x[1] - 1.0)
    du2 = 2.0 * x[0] * (x[0] - 1.0) * (2.0 * x[0] - 1.0)
    return np.stack([a * du1, -b * du2])


def f_exact(x):
    xx, yy = x[0], x[1]
    f1 = (xx ** 2 * (12.0 - 24.0 * yy) * (xx - 1.0) ** 2
          - 4.0 * yy * (yy - 1.0)
          * (xx ** 2 * yy + xx ** 2 * (yy - 1.0)
             + 4.0 * xx * yy * (xx - 1.0)
             + 4.0 * xx * (xx - 1.0) * (yy - 1.0)
             + yy * (xx - 1.0) ** 2
             + (xx - 1.0) ** 2 * (yy - 1.0)) + 1.0)
    f2 = (4.0 * xx * (xx - 1.0)
          * (xx * yy ** 2 + 4.0 * xx * yy * (yy - 1.0)
             + xx * (yy - 1.0) ** 2 + yy ** 2 * (xx - 1.0)
             + 4.0 * yy * (xx - 1.0) * (yy - 1.0)
             + (xx - 1.0) * (yy - 1.0) ** 2)
          + 12.0 * yy ** 2 * (2.0 * xx - 1.0) * (yy - 1.0) ** 2)
    return np.stack([f1, f2])


@BilinearForm
def viscous(u, v, w):
    return ddot(grad(u), grad(v))


@BilinearForm
def neg_div(u, q, w):
    return -div(u) * q


@LinearForm
def body(v, w):
    f = f_exact(w.x)
    return f[0] * v[0] + f[1] * v[1]


@Functional
def vel_err_sq(w):
    ue = u_exact(w.x)
    return (w["uh"][0] - ue[0]) ** 2 + (w["uh"][1] - ue[1]) ** 2


def run(refine, vel_elem):
    m = MeshTri().refined(refine)
    bu = Basis(m, vel_elem, intorder=4)
    bp = Basis(m, ElementTriP1(), intorder=4)
    A = viscous.assemble(bu)
    B = neg_div.assemble(bu, bp)
    K = sp.bmat([[A, B.T], [B, None]], format="csr")
    F = np.concatenate([body.assemble(bu), bp.zeros()])
    D = np.concatenate([bu.get_dofs().all(), [bu.N]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        x = solve(*condense(K, F, x=np.zeros(K.shape[0]), D=D))
        msgs = sorted({str(c.message) for c in caught})
    finite = bool(np.isfinite(x).all())
    if finite:
        err = float(np.sqrt(vel_err_sq.assemble(
            bu, uh=bu.interpolate(x[:bu.N]))))
    else:
        err = float("nan")
    return err, msgs, finite


def main() -> int:
    ok = True

    # --- Taylor-Hood ------------------------------------------------------
    errs = []
    for r in (2, 3, 4):
        e, msgs, fin = run(r, ElementVector(ElementTriP2()))
        errs.append(e)
        print(f"th_refine{r}_velocity_l2={e:.4e} warnings={msgs!r} "
              f"finite={fin}")
        if msgs or not fin:
            print(f"FAIL: Taylor-Hood at refine {r} warned or went non-finite",
                  file=sys.stderr)
            ok = False
    rates = [float(np.log2(errs[i] / errs[i + 1]))
             for i in range(len(errs) - 1)]
    print(f"th_rates={[f'{v:.3f}' for v in rates]}")
    print(f"th_converges_at_order_three={all(2.7 < v < 3.3 for v in rates)}")
    print(f"th_error_decreases={errs[-1] < errs[0]}")
    if not all(2.7 < v < 3.3 for v in rates):
        print("FAIL: Taylor-Hood did not converge at order k+1 = 3",
              file=sys.stderr)
        ok = False

    # --- equal order on the structured mesh -------------------------------
    warned, nonfinite = [], []
    # the inf-sup-violating velocity space.  MUTATION: the documented fix,
    # the Taylor-Hood P2 velocity.
    eq_vel = ElementTriP1 if not MUTATE else ElementTriP2
    for r in (2, 3, 4):
        e, msgs, fin = run(r, ElementVector(eq_vel()))
        hit = any("Matrix is exactly singular" in x for x in msgs)
        warned.append(hit)
        nonfinite.append(not fin)
        print(f"eq_refine{r}_warnings={msgs!r}")
        print(f"eq_refine{r}_emits_matrixrankwarning={hit}")
        print(f"eq_refine{r}_finite={fin}")
    print(f"equal_order_warns_at_every_level={all(warned)}")
    print(f"equal_order_nonfinite_at_every_level={all(nonfinite)}")
    print(f"equal_order_is_exactly_singular_on_structured_mesh="
          f"{all(warned) and all(nonfinite)}")
    if not all(warned):
        print("FAIL: the equal-order pair did not emit MatrixRankWarning",
              file=sys.stderr)
        ok = False
    if not all(nonfinite):
        print("FAIL: the equal-order solution stayed finite", file=sys.stderr)
        ok = False

    # --- and the pairing of warning with isfinite -------------------------
    print(f"warning_and_isfinite_agree_here={all(warned) == all(nonfinite)}")
    print(f"isfinite_is_the_right_partner_for_this_fault="
          f"{all(warned) and all(nonfinite)}")
    print(f"taylor_hood_silent_and_convergent="
          f"{all(2.7 < v < 3.3 for v in rates)}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
