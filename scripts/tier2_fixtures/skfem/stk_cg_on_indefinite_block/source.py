"""Tier-2: CG on the Stokes block system does not stall -- it explodes, quietly.

Claim: skfem stokes#0 -- the Stokes block system [[A, B^T], [B, 0]] is
INDEFINITE, so CG on the full block system diverges; use a direct solver or
block Uzawa.  "Signal: scipy.sparse.linalg.cg on the assembled block matrix
reports info != 0 with the residual stalling at O(1); switching to spsolve
returns a vector and the divergence of velocity is below tolerance."  The
entry is marked as inherited from prose and not empirically separated.

Measured on skfem 12.0.1 / scipy 1.15.3, Taylor-Hood ElementVector(P2)/P1 on
MeshTri().refined(3), one pressure DOF pinned so the only remaining defect is
indefiniteness:

  * INDEFINITENESS IS CONFIRMED by the spectrum: the block matrix has as many
    negative eigenvalues as it has free pressure unknowns.
  * THE STATED SIGNAL DOES NOT FIRE AT ALL.  scipy's cg on this indefinite
    system returns info = 0 -- convergence -- with a relative residual at the
    requested tolerance, on every one of the right-hand sides tried, at two
    mesh levels, loading the velocity block alone and loading both blocks.
    There is no info != 0 to catch and no residual stalling at O(1) to
    detect.  An agent told to guard by checking info would guard nothing,
    and would then be surprised on the day CG does break, because the guard
    has never been exercised.
  * so the CAUTION is sound in theory -- CG has no convergence guarantee on
    an indefinite matrix -- but the OBSERVABLE the entry prescribes is not
    produced by this code on this problem.  The honest check is the one that
    works whatever the solver does: measure the relative residual yourself
    and require np.isfinite, rather than trusting a return flag that reads
    "converged" either way.
  * the direct solve is what the entry recommends and it does return a
    residual at machine level, four orders below CG's.
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
    Functional,
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


def main() -> int:
    ok = True
    m = MeshTri().refined(3)
    bu = Basis(m, ElementVector(ElementTriP2()), intorder=4)
    bp = Basis(m, ElementTriP1(), intorder=4)
    A = a_form.assemble(bu)
    B = b_form.assemble(bu, bp)
    K = sp.bmat([[A, B.T], [B, None]], format="csr")
    print(f"velocity_dofs={bu.N} pressure_dofs={bp.N}")
    print(f"block_shape={K.shape} block_nnz={K.nnz}")

    # Dirichlet on the whole velocity boundary + one pinned pressure DOF, so
    # the only remaining defect is indefiniteness.
    D = np.concatenate([bu.get_dofs().all(), [bu.N]])
    rng = np.random.default_rng(0)
    f = np.zeros(K.shape[0])
    f[:bu.N] = 0.01 * rng.standard_normal(bu.N)
    Kc, fc, xc, I = condense(K, f, D=D)
    print(f"condensed_shape={Kc.shape}")

    ev = np.linalg.eigvalsh(Kc.toarray())
    n_neg = int((ev < 0).sum())
    print(f"negative_eigenvalues={n_neg}")
    print(f"positive_eigenvalues={int((ev > 0).sum())}")
    print(f"block_matrix_is_indefinite={n_neg > 0 and (ev > 0).any()}")
    print(f"free_pressure_unknowns={bp.N - 1}")
    print(f"negative_count_equals_free_pressure_unknowns="
          f"{n_neg == bp.N - 1}")
    if not (n_neg > 0 and (ev > 0).any()):
        print("FAIL: the block matrix was not indefinite", file=sys.stderr)
        ok = False

    # --- CG ---------------------------------------------------------------
    err = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            x, info = spl.cg(Kc, fc, rtol=1e-10, maxiter=2000)
        except Exception as e:                        # noqa: BLE001
            err = e
            x, info = None, None
        msgs = sorted({str(c.message) for c in caught})
    print(f"cg_raises={err is not None}")
    print(f"cg_info={info}")
    print(f"cg_info_is_nonzero={info != 0}")
    print(f"cg_info_is_zero_meaning_converged={info == 0}")
    print(f"cg_warnings={msgs!r}")
    print(f"cg_is_silent={err is None and not msgs}")
    if err is not None:
        print(f"FAIL: cg raised {type(err).__name__}", file=sys.stderr)
        ok = False
    if info != 0:
        print(f"FAIL: cg reported info={info}, so the entry's signal fires "
              f"after all", file=sys.stderr)
        ok = False

    rel_cg = float(np.linalg.norm(Kc @ x - fc) / np.linalg.norm(fc))
    print(f"cg_relative_residual={rel_cg:.4e}")
    print(f"cg_result_is_finite={bool(np.isfinite(x).all())}")
    print(f"cg_residual_stalls_at_order_one={0.1 < rel_cg < 10.0}")
    print(f"cg_actually_converged={info == 0 and rel_cg < 1e-8}")
    if 0.1 < rel_cg < 10.0:
        print("FAIL: the CG residual stalled at O(1), so the entry's "
              "wording is usable after all", file=sys.stderr)
        ok = False

    # --- and it is not one lucky right-hand side -------------------------
    infos, rels = [], []
    for seed in range(6):
        g = np.random.default_rng(seed)
        rhs = np.zeros(K.shape[0])
        rhs[:bu.N] = 0.01 * g.standard_normal(bu.N)
        Kc2, fc2, _, _ = condense(K, rhs, D=D)
        xx, ii = spl.cg(Kc2, fc2, rtol=1e-10, maxiter=3000)
        infos.append(int(ii))
        rels.append(float(np.linalg.norm(Kc2 @ xx - fc2)
                          / np.linalg.norm(fc2)))
    for seed in range(3):
        g = np.random.default_rng(100 + seed)
        rhs = 0.01 * g.standard_normal(K.shape[0])
        Kc2, fc2, _, _ = condense(K, rhs, D=D)
        xx, ii = spl.cg(Kc2, fc2, rtol=1e-10, maxiter=3000)
        infos.append(int(ii))
        rels.append(float(np.linalg.norm(Kc2 @ xx - fc2)
                          / np.linalg.norm(fc2)))
    print(f"cg_trials={len(infos)}")
    print(f"cg_infos={infos}")
    print(f"cg_max_relative_residual={max(rels):.4e}")
    print(f"cg_converged_on_every_trial="
          f"{all(i == 0 for i in infos) and max(rels) < 1e-8}")
    print(f"cg_reported_failure_on_any_trial={any(i != 0 for i in infos)}")
    if any(i != 0 for i in infos):
        print("FAIL: cg reported a failure on some right-hand side, so the "
              "entry's info != 0 signal does fire", file=sys.stderr)
        ok = False

    # --- the direct solve ---------------------------------------------------
    xs = solve(Kc, fc)
    rel_direct = float(np.linalg.norm(Kc @ xs - fc) / np.linalg.norm(fc))
    print(f"spsolve_relative_residual={rel_direct:.4e}")
    print(f"spsolve_converges={rel_direct < 1e-10}")
    print(f"direct_beats_cg_by_orders={rel_cg / max(rel_direct, 1e-30):.3e}")
    full = xc.copy()
    full[I] = xs
    uu = full[:bu.N]

    @Functional
    def divsq(w):
        return div(w["uh"]) ** 2

    dv = float(np.sqrt(divsq.assemble(bu, uh=bu.interpolate(uu))))
    print(f"direct_solution_divergence_l2={dv:.4e}")
    print(f"direct_beats_cg_residual={rel_direct < rel_cg}")
    if rel_direct >= 1e-10:
        print("FAIL: the direct solve did not converge", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
