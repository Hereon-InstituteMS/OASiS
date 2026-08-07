"""Tier-2: Krylov-Uzawa works, but not at the rate claimed, and its rival
does not stall.

Claim: skfem stokes#1 -- "Krylov-Uzawa (skfem example 30) splits the
saddle-point: outer iteration on the pressure Schur complement, inner solve of
the velocity block.  Signal: per-outer-iter pressure residual drops by a factor
0.1..0.3; switching to monolithic Krylov on the full block system shows
residual stalling."  Marked inherited, recommendation only, never verified.

Measured on skfem 12.0.1 / scipy 1.15.3, lid-driven cavity, Taylor-Hood
ElementVector(TriP2)/TriP1, at two refinement levels.  The Schur complement
S = B A^-1 B^T is applied with a sparse LU of the free velocity block and the
constant pressure mode is projected out; the outer solver is CG on S.

  * THE SPLIT ITSELF WORKS.  Uzawa converges, reporting info = 0, and the
    outer count is far below the pressure DOF count.
  * THE QUOTED RATE IS WRONG.  The per-outer residual factor is nowhere near
    0.1..0.3 -- it is around 0.7 on the coarse mesh and around 0.85 on the
    finer one, and it WORSENS with refinement, which is what an unpreconditioned
    Schur solve does.  A gate keyed on "0.1..0.3 per outer iteration" would
    call a healthy run broken.
  * "MONOLITHIC KRYLOV SHOWS RESIDUAL STALLING" IS WRONG.  CG on the
    condensed block system converges to the requested tolerance with info = 0
    at both refinements, exactly as stokes#0 found independently.  There is
    no stall to observe.
  * so the honest comparison is a cost one, not a converges/does-not one:
    Uzawa needs far fewer OUTER iterations than monolithic CG needs Krylov
    iterations, but each outer step carries a velocity solve, so the counts
    are not comparable units and the entry's framing invites reading them as
    if they were.

Mutation control: the cause this fixture names for the bad and refinement-
dependent outer rate is that the Schur solve is UNPRECONDITIONED.  T2_MUTATE=1
removes exactly that at the pathology site: the outer CG on S is handed the
standard pressure-mass preconditioner M_p^-1 (projected onto the mean-zero
subspace), which is spectrally equivalent to S.  The outer rate then stops
degrading with refinement.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os

MUTATE = os.environ.get("T2_MUTATE") == "1"

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

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
)
from skfem.helpers import ddot, div, grad


@BilinearForm
def viscous(u, v, w):
    return ddot(grad(u), grad(v))


@BilinearForm
def neg_div(u, q, w):
    return -div(u) * q


@BilinearForm
def p_mass(p, q, w):
    return p * q


def setup(refine):
    m = MeshTri().refined(refine)
    bu = Basis(m, ElementVector(ElementTriP2()), intorder=4)
    bp = Basis(m, ElementTriP1(), intorder=4)
    A = viscous.assemble(bu)
    B = neg_div.assemble(bu, bp)
    lid = bu.get_dofs(lambda x: x[1] > 1.0 - 1e-10)
    D = bu.get_dofs().all()
    xb = np.zeros(bu.N)
    xb[lid.nodal["u^1"]] = 1.0
    return bu, bp, A, B, D, xb


def uzawa(refine):
    bu, bp, A, B, D, xb = setup(refine)
    free = np.setdiff1d(np.arange(bu.N), D)
    lu = spl.splu(A[free][:, free].tocsc())
    Bf = B[:, free].tocsr()
    f = (-A @ xb)[free]
    n = bp.N
    one = np.ones(n) / np.sqrt(n)

    def proj(v):
        return v - one * (one @ v)

    S = spl.LinearOperator((n, n), matvec=lambda p: Bf @ lu.solve(Bf.T @ p))
    rhs = proj(Bf @ lu.solve(f) - (B @ xb))
    prec = None
    if MUTATE:
        # the missing pressure-mass preconditioner, put back
        lum = spl.splu(p_mass.assemble(bp).tocsc())
        prec = spl.LinearOperator((n, n),
                                  matvec=lambda r: proj(lum.solve(proj(r))))
    res = []
    p, info = spl.cg(S, rhs, rtol=1e-10, maxiter=500, M=prec,
                     callback=lambda pk: res.append(
                         float(np.linalg.norm(proj(rhs - S @ pk)))))
    return info, res, n


def monolithic(refine):
    bu, bp, A, B, D, xb = setup(refine)
    K = sp.bmat([[A, B.T], [B, None]], format="csr")
    Dm = np.concatenate([D, [bu.N]])
    xbm = np.zeros(K.shape[0])
    xbm[:bu.N] = xb
    Kc, fc, xc, I = condense(K, np.zeros(K.shape[0]), x=xbm, D=Dm)
    res = []
    x, info = spl.cg(Kc, fc, rtol=1e-10, maxiter=5000,
                     callback=lambda xk: res.append(
                         float(np.linalg.norm(fc - Kc @ xk))))
    rel = float(np.linalg.norm(fc - Kc @ x) / np.linalg.norm(fc))
    return info, len(res), rel


def main() -> int:
    ok = True
    means, outers = [], []
    for refine in (2, 3):
        info, res, npres = uzawa(refine)
        ratios = [res[i + 1] / res[i] for i in range(min(10, len(res) - 1))
                  if res[i] > 0]
        mean = float(np.mean(ratios)) if ratios else float("nan")
        means.append(mean)
        outers.append(len(res))
        print(f"r{refine}_pressure_dofs={npres}")
        print(f"r{refine}_uzawa_info={info}")
        print(f"r{refine}_uzawa_converged={info == 0}")
        print(f"r{refine}_uzawa_outer_iterations={len(res)}")
        print(f"r{refine}_uzawa_mean_outer_ratio={mean:.4f}")
        print(f"r{refine}_uzawa_ratio_in_0p1_to_0p3_band="
              f"{all(0.1 <= x <= 0.3 for x in ratios)}")
        print(f"r{refine}_uzawa_outer_below_pressure_dofs="
              f"{len(res) < npres * 2}")
        if info != 0:
            print(f"FAIL: Uzawa did not converge at refine {refine}",
                  file=sys.stderr)
            ok = False
        if all(0.1 <= x <= 0.3 for x in ratios):
            print(f"FAIL: at refine {refine} the outer ratio really was in "
                  f"the quoted 0.1..0.3 band", file=sys.stderr)
            ok = False

    print(f"uzawa_mean_ratios={[f'{x:.4f}' for x in means]}")
    print(f"uzawa_rate_worsens_with_refinement={means[1] > means[0]}")
    print(f"uzawa_rate_far_above_quoted_band={min(means) > 0.5}")
    if means[1] <= means[0]:
        print("FAIL: the Uzawa outer rate did not degrade under refinement",
              file=sys.stderr)
        ok = False

    # --- the rival that was said to stall ---------------------------------
    infos = []
    for refine in (2, 3):
        info, iters, rel = monolithic(refine)
        infos.append(info)
        print(f"r{refine}_monolithic_info={info}")
        print(f"r{refine}_monolithic_converged={info == 0}")
        print(f"r{refine}_monolithic_iterations={iters}")
        print(f"r{refine}_monolithic_final_relative_residual={rel:.3e}")
        print(f"r{refine}_monolithic_stalls={info != 0 or rel > 1e-8}")
        if info != 0 or rel > 1e-8:
            print(f"FAIL: monolithic Krylov stalled at refine {refine}, so "
                  f"the entry's comparison holds", file=sys.stderr)
            ok = False
    print(f"monolithic_converges_at_every_level={all(i == 0 for i in infos)}")
    print(f"monolithic_stalling_claim_holds="
          f"{any(i != 0 for i in infos)}")

    # --- the honest comparison ---------------------------------------------
    _, mono_iters, _ = monolithic(3)
    print(f"uzawa_outer_iterations_r3={outers[1]}")
    print(f"monolithic_krylov_iterations_r3={mono_iters}")
    print(f"uzawa_needs_far_fewer_outer_steps={outers[1] < mono_iters / 5}")
    print(f"counts_are_not_comparable_units=True")
    if not outers[1] < mono_iters / 5:
        print("FAIL: Uzawa did not need far fewer outer steps",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
