"""Tier-2: the enclosed-flow pressure nullspace emits no warning at all.

Claim: skfem navier_stokes#3 -- corrected 2026-08-06.  The entry previously
said spsolve "does NOT raise -- it emits `MatrixRankWarning: Matrix is exactly
singular` ... and RETURNS, giving an all-NaN vector, or else returns a
pressure field with an arbitrary constant offset.  Catch it with
warnings.catch_warnings(record=True) plus an np.isfinite(x).all() assertion."

Measured on skfem 12.0.1 / scipy 1.15.3, lid-driven cavity (enclosed flow:
velocity prescribed on the whole boundary), Taylor-Hood ElementVector(P2)/P1,
at three refinement levels:

  * the null space is real and one-dimensional at every level, and its
    vector is zero on velocity and constant on pressure.
  * BOTH HALVES OF THE PRESCRIBED GUARD ARE BLIND HERE.  The
    catch_warnings record comes back EMPTY -- no MatrixRankWarning, no
    warning of any kind -- and np.isfinite(x).all() is True.  An agent
    following the entry writes two checks, both pass, and reports success on
    a solution whose pressure level is arbitrary.  There is no all-NaN
    branch either.
  * the velocity is not merely finite, it is CORRECT: it matches the pinned
    solution to round-off.  Only the pressure level is undetermined, which
    is why neither guard can see anything wrong.
  * the observable that IS present: the nullity of the condensed block, or
    two solves with different pins -- identical velocity, pressure differing
    by a constant.  Both are checked here.
  * for contrast the same guard is exercised on a genuinely rank-deficient
    matrix, where the warning does fire AND the result is non-finite --
    which is the case isfinite was the right partner for.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology site
-- the enclosed-flow solve at each refinement is given one pinned pressure DOF
in addition to the velocity Dirichlet set, which is the remedy this whole entry
exists to motivate.  The null space is then empty and
'r2_nullspace_is_one_dimensional=True', 'r3_nullspace_is_one_dimensional=True'
and 'r4_nullspace_is_one_dimensional=True' disappear from the output.  The two
blind-guard lines are supposed to survive: their point is that they do NOT move
when the pathology does.  Re-run: T2_MUTATE=1 python source.py
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
import scipy.sparse.linalg as spl
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    MeshTri,
    condense,
    solve,
)
from skfem.helpers import ddot, div, grad

MUTATE = os.environ.get("T2_MUTATE") == "1"


@BilinearForm
def a_form(u, v, w):
    return ddot(grad(u), grad(v))


@BilinearForm
def b_form(u, q, w):
    return -div(u) * q


def cavity(refine):
    m = MeshTri().refined(refine)
    bu = Basis(m, ElementVector(ElementTriP2()), intorder=4)
    bp = Basis(m, ElementTriP1(), intorder=4)
    A = a_form.assemble(bu)
    B = b_form.assemble(bu, bp)
    K = sp.bmat([[A, B.T], [B, None]], format="csr")
    lid = bu.get_dofs(lambda x: x[1] > 1.0 - 1e-10)
    xb = np.zeros(K.shape[0])
    xb[lid.nodal["u^1"]] = 1.0
    return bu, bp, K, xb, bu.get_dofs().all()


def guarded_solve(K, xb, D):
    """Exactly the guard the entry prescribes."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Kc, fc, xc, I = condense(K, np.zeros(K.shape[0]), x=xb, D=D)
        x = solve(Kc, fc)
        msgs = sorted({str(c.message) for c in caught})
    full = xc.copy()
    full[I] = x
    return full, msgs, Kc


def main() -> int:
    ok = True
    empties, finites = [], []
    for r in (2, 3, 4):
        bu, bp, K, xb, vd = cavity(r)
        # THE PATHOLOGY: an enclosed flow constrained on velocity only, with
        # no pressure pin, so the block carries a constant-pressure null
        # space.  T2_MUTATE applies the documented fix at this site by adding
        # one pinned pressure DOF to the Dirichlet set.
        dset = np.concatenate([vd, [bu.N]]) if MUTATE else vd
        full, msgs, Kc = guarded_solve(K, xb, dset)
        nul = int((np.abs(np.linalg.eigvalsh(Kc.toarray())) < 1e-10).sum())
        empties.append(not msgs)
        finites.append(bool(np.isfinite(full).all()))
        print(f"r{r}_block={K.shape} nullity={nul}")
        print(f"r{r}_warnings={msgs!r}")
        print(f"r{r}_nullspace_is_one_dimensional={nul == 1}")
        print(f"r{r}_catch_warnings_record_is_empty={not msgs}")
        print(f"r{r}_isfinite_passes={bool(np.isfinite(full).all())}")
        print(f"r{r}_all_nan={bool(np.isnan(full).all())}")
        if nul != 1:
            print(f"FAIL: refine {r} did not have a 1-D null space",
                  file=sys.stderr)
            ok = False
    print(f"warning_absent_at_every_refinement={all(empties)}")
    print(f"isfinite_passes_at_every_refinement={all(finites)}")
    print(f"both_prescribed_guards_are_blind={all(empties) and all(finites)}")
    if not all(empties):
        print("FAIL: a warning WAS emitted, so the prescribed guard fires",
              file=sys.stderr)
        ok = False
    if not all(finites):
        print("FAIL: the solution was non-finite, so the isfinite guard "
              "fires", file=sys.stderr)
        ok = False

    # --- the observable that IS present ------------------------------------
    bu, bp, K, xb, vd = cavity(3)
    sols = {}
    for tag, (idx, val) in {"a": (0, 0.0), "b": (bp.N - 1, 2.5)}.items():
        D = np.concatenate([vd, [bu.N + idx]])
        xx = xb.copy()
        xx[bu.N + idx] = val
        full, msgs, _ = guarded_solve(K, xx, D)
        sols[tag] = full
        print(f"pinned_{tag}_warnings={msgs!r}")
    unpinned, _, _ = guarded_solve(K, xb, vd)
    dv_ab = float(np.abs(sols["a"][:bu.N] - sols["b"][:bu.N]).max())
    dv_un = float(np.abs(sols["a"][:bu.N] - unpinned[:bu.N]).max())
    dp = sols["a"][bu.N:] - sols["b"][bu.N:]
    print(f"velocity_difference_between_pins={dv_ab:.3e}")
    print(f"velocity_difference_pinned_vs_unpinned={dv_un:.3e}")
    print(f"velocity_is_identical_across_pins={dv_ab < 1e-9}")
    print(f"unpinned_velocity_is_also_correct={dv_un < 1e-9}")
    print(f"pressure_difference_spread={float(dp.max() - dp.min()):.3e}")
    print(f"pressure_differs_by_a_constant="
          f"{float(dp.max() - dp.min()) < 1e-9}")
    print(f"pressure_offset_is_nonzero={abs(float(dp.mean())) > 1e-6}")
    print(f"nullspace_observable_via_two_pins="
          f"{dv_ab < 1e-9 and float(dp.max() - dp.min()) < 1e-9
             and abs(float(dp.mean())) > 1e-6}")
    if not (dv_ab < 1e-9 and float(dp.max() - dp.min()) < 1e-9):
        print("FAIL: the two-pin probe did not isolate a constant pressure "
              "shift", file=sys.stderr)
        ok = False
    if dv_un >= 1e-9:
        print("FAIL: the unpinned velocity was wrong, so the failure is not "
              "confined to the pressure level", file=sys.stderr)
        ok = False

    # --- the case the guard WAS right for ----------------------------------
    A = sp.eye(6, format="csr").tolil()
    A[3, 3] = 0.0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        z = spl.spsolve(A.tocsr(), np.ones(6))
        zmsgs = sorted({str(c.message) for c in caught})
    print(f"rank_deficient_warnings={zmsgs!r}")
    print(f"rank_deficient_warns={bool(zmsgs)}")
    print(f"rank_deficient_isfinite_fails={not bool(np.isfinite(z).all())}")
    print(f"guard_works_on_a_genuinely_rank_deficient_matrix="
          f"{bool(zmsgs) and not bool(np.isfinite(z).all())}")
    if not (zmsgs and not np.isfinite(z).all()):
        print("FAIL: the guard did not fire even on a rank-deficient matrix",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
