"""Tier-2: the constant-pressure nullspace of enclosed Stokes flow.

Claim: skfem stokes#2 -- enclosed-flow Stokes admits a constant pressure null
space; pin pressure at one DoF or add a mean-zero Lagrange multiplier.  Open
flows with do-nothing (traction-free) outlet BCs determine pressure uniquely
without pinning.  "Signal: spsolve(A_block, b_block) returns SUCCESSFULLY but
u_h shows the right velocity field while p_h has an arbitrary additive
constant."  Marked as inherited and not empirically verified.

Measured on skfem 12.0.1 / scipy 1.15.3, Taylor-Hood ElementVector(P2)/P1 on
MeshTri().refined(3), lid-driven-cavity boundary conditions (velocity
prescribed on the whole boundary = enclosed flow):

  * THE NULLSPACE IS EXACTLY ONE-DIMENSIONAL and it IS the constant pressure:
    the block matrix has exactly one eigenvalue at machine zero and its
    eigenvector is zero on the velocity block and constant on the pressure
    block.
  * "RETURNS SUCCESSFULLY" IS CONFIRMED, and it is the dangerous half.  With
    no pressure pinned at all the solve raises nothing, warns nothing -- the
    catch_warnings record is EMPTY, no MatrixRankWarning -- and hands back a
    finite vector whose velocity block matches the pinned runs.  This is
    worth stating next to the neighbouring skfem entries (hydraulic_
    resistance#2, navier_stokes#3) which describe scipy emitting
    "Matrix is exactly singular" on a singular block system: that warning is
    NOT universal.  Here the singular system is consistent, SuperLU returns
    a particular solution, and the only thing wrong with it is the pressure
    level.  A guard built solely on catching that warning does not fire.
  * pinning ONE pressure DOF removes the warning, gives a finite solution,
    and the velocity is independent of WHICH DOF is pinned and of the value
    pinned -- the pressure differs between the two runs by a constant, which
    is the additive-constant statement made precise and checkable.

Mutation control: T2_MUTATE=1 applies the entry's own remedy at the pathology
site -- the Dirichlet set used for the spectrum probe and for the "unpinned"
solve gains the single pinned pressure DOF bu.N, so nothing is left
undetermined.  The constant-pressure nullspace is then gone and the condensed
block has no eigenvalue at machine zero.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os

MUTATE = os.environ.get("T2_MUTATE") == "1"

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
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.helpers import ddot, div, grad


@BilinearForm
def a_form(u, v, w):
    return ddot(grad(u), grad(v))


@BilinearForm
def b_form(u, q, w):
    return -div(u) * q


@LinearForm
def body(v, w):
    return 0.0 * v[0]


def assemble():
    m = MeshTri().refined(3)
    bu = Basis(m, ElementVector(ElementTriP2()), intorder=4)
    bp = Basis(m, ElementTriP1(), intorder=4)
    A = a_form.assemble(bu)
    B = b_form.assemble(bu, bp)
    K = sp.bmat([[A, B.T], [B, None]], format="csr")
    f = np.zeros(K.shape[0])
    # lid-driven cavity: u_x = 1 on the top edge, zero elsewhere
    lid = bu.get_dofs(lambda x: x[1] > 1.0 - 1e-10)
    x = np.zeros(K.shape[0])
    x[lid.nodal["u^1"]] = 1.0
    return m, bu, bp, K, f, x, bu.get_dofs().all()


def main() -> int:
    ok = True
    m, bu, bp, K, f, xbc, vel_d = assemble()
    print(f"velocity_dofs={bu.N} pressure_dofs={bp.N} block={K.shape}")

    # the pathology: nothing pins the pressure level.  Under mutation the
    # entry's remedy -- one pinned pressure DOF -- is applied here.
    unpinned_D = (np.concatenate([vel_d, [bu.N]]) if MUTATE else vel_d)

    # --- the nullspace ----------------------------------------------------
    Kc, fc, xc, I = condense(K, f, x=xbc, D=unpinned_D)
    ev, evec = np.linalg.eigh(Kc.toarray())
    tiny = np.abs(ev) < 1e-10
    print(f"near_zero_eigenvalues={int(tiny.sum())}")
    print(f"nullspace_is_one_dimensional={int(tiny.sum()) == 1}")
    if int(tiny.sum()) == 1:
        v = evec[:, int(np.argmin(np.abs(ev)))]
        free = I
        vel_part = v[free < bu.N]
        prs_part = v[free >= bu.N]
        print(f"nullvector_velocity_max_abs={float(np.abs(vel_part).max()):.3e}")
        print(f"nullvector_pressure_spread="
              f"{float(prs_part.max() - prs_part.min()):.3e}")
        print(f"nullvector_is_zero_on_velocity="
              f"{float(np.abs(vel_part).max()) < 1e-10}")
        print(f"nullvector_is_constant_on_pressure="
              f"{float(prs_part.max() - prs_part.min()) < 1e-10}")
        if float(np.abs(vel_part).max()) >= 1e-10:
            print("FAIL: the null vector was not confined to the pressure "
                  "block", file=sys.stderr)
            ok = False
    else:
        print("FAIL: the nullspace was not one-dimensional", file=sys.stderr)
        ok = False

    # --- unpinned -----------------------------------------------------------
    err = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            u_free = solve(*condense(K, f, x=xbc, D=unpinned_D))
        except Exception as e:                        # noqa: BLE001
            err = e
            u_free = None
        msgs = sorted({str(c.message) for c in caught})
    print(f"unpinned_raises={err is not None}")
    print(f"unpinned_warnings={msgs!r}")
    print(f"unpinned_warns_matrix_exactly_singular="
          f"{any('Matrix is exactly singular' in x for x in msgs)}")
    print(f"unpinned_all_nan="
          f"{u_free is not None and bool(np.isnan(u_free).all())}")
    print(f"unpinned_velocity_is_usable="
          f"{u_free is not None and bool(np.isfinite(u_free[:bu.N]).all())}")
    print(f"unpinned_returns_successfully="
          f"{err is None and not msgs and bool(np.isfinite(u_free).all())}")
    if err is not None:
        print(f"FAIL: the unpinned solve raised {type(err).__name__}",
              file=sys.stderr)
        ok = False
    if msgs:
        print(f"FAIL: the unpinned solve warned {msgs!r}, so it does not "
              f"return silently", file=sys.stderr)
        ok = False
    if u_free is None or not np.isfinite(u_free).all():
        print("FAIL: the unpinned solve did not return a finite vector",
              file=sys.stderr)
        ok = False

    # --- pinned, two different ways ---------------------------------------
    runs = {}
    for tag, (idx, val) in {"a": (0, 0.0), "b": (bp.N - 1, 3.5)}.items():
        D = np.concatenate([vel_d, [bu.N + idx]])
        xb = xbc.copy()
        xb[bu.N + idx] = val
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sol = solve(*condense(K, f, x=xb, D=D))
            mm = sorted({str(c.message) for c in caught})
        runs[tag] = (sol, mm)
        print(f"pinned_{tag}_warnings={mm!r}")
        print(f"pinned_{tag}_finite={bool(np.isfinite(sol).all())}")
        print(f"pinned_{tag}_velocity_max={float(np.abs(sol[:bu.N]).max()):.6f}")
        if mm:
            print(f"FAIL: pinning still warned {mm!r}", file=sys.stderr)
            ok = False
        if not np.isfinite(sol).all():
            print("FAIL: the pinned solve was not finite", file=sys.stderr)
            ok = False

    ua, ub = runs["a"][0], runs["b"][0]
    dv = float(np.abs(ua[:bu.N] - ub[:bu.N]).max())
    dp = ua[bu.N:] - ub[bu.N:]
    print(f"velocity_difference_between_pinnings={dv:.3e}")
    print(f"velocity_is_independent_of_the_pinning={dv < 1e-9}")
    print(f"pressure_difference_spread={float(dp.max() - dp.min()):.3e}")
    print(f"pressure_differs_by_a_constant="
          f"{float(dp.max() - dp.min()) < 1e-9}")
    print(f"pressure_constant_offset={float(dp.mean()):.6f}")
    print(f"offset_is_nonzero={abs(float(dp.mean())) > 1e-6}")
    dvu = float(np.abs(u_free[:bu.N] - ua[:bu.N]).max())
    print(f"unpinned_velocity_matches_pinned={dvu:.3e}")
    print(f"only_the_pressure_level_was_undetermined={dvu < 1e-9}")
    if dvu >= 1e-9:
        print("FAIL: the unpinned velocity differed from the pinned one, so "
              "more than the pressure level was undetermined", file=sys.stderr)
        ok = False
    if dv >= 1e-9:
        print("FAIL: the velocity depended on which pressure DOF was pinned",
              file=sys.stderr)
        ok = False
    if float(dp.max() - dp.min()) >= 1e-9:
        print("FAIL: the two pressures did not differ by a constant",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
