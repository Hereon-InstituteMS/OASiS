"""Tier-2: CG on the non-symmetric upwind-DG matrix -- and what scipy actually does.

Claim: skfem dg_methods#9 -- the DG system is non-symmetric even for symmetric
problems (upwind asymmetry).  "Signal: scipy.sparse.linalg.cg fails with
`RuntimeError: matrix not positive definite` or stalls; switch to gmres /
direct LU."

PARTIAL FALSIFICATION (scipy 1.15.3): ``cg`` raises NOTHING.  It returns an
``info`` code -- here info = 1000, i.e. maxiter reached -- and the returned
vector has a relative residual of 1.3e+03, larger than the right-hand side
itself.  No RuntimeError, no exception of any kind, and the string
"matrix not positive definite" is never produced.  A gate that only catches
exceptions therefore accepts a completely unconverged answer.

Wrong variant: hand the upwind-DG advection matrix to
``scipy.sparse.linalg.cg`` and inspect the return value instead of expecting a
raise.

Test problem: pure advection b = (1, 0.5), f = 0, inflow g = 1, exact u == 1.

Observed on skfem 12.0.1 / scipy 1.15.3 (MeshTri().refined(3), 384 DOFs):
  * max|A - A.T| = 6.25e-02 -- the operator is genuinely non-symmetric
  * the symmetric part is only positive SEMI-definite (lambda_min = -3.4e-17)
  * cg   -> info != 0, relative residual > 1
  * gmres-> info = 0, relative residual 9.3e-11
  * splu -> relative residual 8.0e-16 and max|u - 1| = 1.8e-15
"""
from __future__ import annotations

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
from scipy.sparse.linalg import cg, gmres, splu

B = np.array([1.0, 0.5])
CLAIMED_MSG = "matrix not positive definite"


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
    return -np.minimum(bn, 0.0) * 1.0 * v


def build():
    m = MeshTri().refined(3)
    e = ElementDG(ElementTriP1())
    ib = Basis(m, e)
    i0 = InteriorFacetBasis(m, e, side=0)
    i1 = InteriorFacetBasis(m, e, side=1)
    fb = FacetBasis(m, e)
    A = (asm(advection_volume, ib)
         + asm(upwind_flux, [i0, i1], [i0, i1])
         + asm(inflow_bilinear, fb))
    return A.tocsr(), asm(inflow_load, fb), ib


def main() -> int:
    ok = True
    A, f, ib = build()
    print(f"n_dofs={A.shape[0]}")

    asym = float(abs(A - A.T).max())
    print(f"matrix_is_nonsymmetric={asym > 1e-12}")
    print(f"max_abs_A_minus_AT={asym:.4e}")
    dense = A.toarray()
    lam_min = float(np.linalg.eigvalsh(0.5 * (dense + dense.T))[0])
    print(f"symmetric_part_is_positive_definite={lam_min > 1e-12}")
    print(f"symmetric_part_lambda_min={lam_min:.4e}")
    if asym <= 1e-12:
        print("FAIL: the upwind DG operator came out symmetric", file=sys.stderr)
        ok = False
    if lam_min > 1e-12:
        print("FAIL: the symmetric part was positive definite after all",
              file=sys.stderr)
        ok = False

    # --- WRONG variant: use CG on it --------------------------------------
    nrm_f = float(np.linalg.norm(f))
    iters = [0]
    raised = ""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            x_cg, info = cg(A, f, rtol=1e-10, maxiter=1000,
                            callback=lambda xk: iters.__setitem__(0, iters[0] + 1))
        except Exception as exc:           # noqa: BLE001 -- we are probing
            raised = f"{type(exc).__name__}: {exc}"
            x_cg, info = np.zeros_like(f), -1
        cg_warnings = [str(c.message) for c in caught]
    res_cg = float(np.linalg.norm(A @ x_cg - f) / nrm_f)
    print(f"cg_raised_an_exception={bool(raised)}")
    print(f"cg_exception_text={raised!r}")
    print(f"cg_claimed_message_emitted={CLAIMED_MSG in raised}")
    print(f"cg_warnings={cg_warnings!r}")
    print(f"cg_info_nonzero={info != 0}")
    print(f"cg_iterations={iters[0]}")
    print(f"cg_relative_residual_gt_1={res_cg > 1.0}")
    print(f"cg_relative_residual={res_cg:.4e}")
    if raised:
        print("FAIL: scipy cg raised -- the falsification no longer holds",
              file=sys.stderr)
        ok = False
    if info == 0 or res_cg <= 1.0:
        print("FAIL: cg converged on the non-symmetric DG matrix", file=sys.stderr)
        ok = False

    # --- RIGHT variants: gmres and a direct LU ----------------------------
    x_g, info_g = gmres(A, f, rtol=1e-10, maxiter=2000)
    res_g = float(np.linalg.norm(A @ x_g - f) / nrm_f)
    print(f"gmres_info_zero={info_g == 0}")
    print(f"gmres_relative_residual_lt_1e-8={res_g < 1e-8}")

    x_lu = splu(A.tocsc()).solve(f)
    res_lu = float(np.linalg.norm(A @ x_lu - f) / nrm_f)
    dev_lu = float(np.abs(x_lu - 1.0).max())
    print(f"lu_relative_residual_lt_1e-12={res_lu < 1e-12}")
    print(f"lu_reproduces_exact_constant={dev_lu < 1e-10}")
    print(f"gmres_res={res_g:.3e} lu_res={res_lu:.3e} lu_dev={dev_lu:.3e}")
    if not (info_g == 0 and res_g < 1e-8):
        print("FAIL: gmres did not converge on the DG matrix", file=sys.stderr)
        ok = False
    if not (res_lu < 1e-12 and dev_lu < 1e-10):
        print("FAIL: the direct LU solve did not recover the exact solution",
              file=sys.stderr)
        ok = False

    print(f"gmres_beats_cg_by_gt_1e8={res_cg / max(res_g, 1e-300) > 1e8}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
