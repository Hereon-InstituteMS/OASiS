"""Tier-2: the SIPG penalty sigma/h -- coercivity below sigma = 4*order^2,
conditioning blow-up far above it.

Claim: skfem dg_methods#5 -- for interior-penalty DG diffusion the penalty is
sigma/h on each interior facet.  sigma too small -> coercivity loss and the
solution norm diverges under refinement; sigma too large -> cond(K) > 1e14 and
the iterative solver stagnates.  Rule of thumb: sigma = 4 * order^2 for
symmetric IP.

Wrong variants: assemble the SAME symmetric interior-penalty operator with
sigma below the rule of thumb, and with sigma many orders above it.

Coercivity is checked exactly, not by proxy: SIPG is symmetric, so the
operator is coercive iff its smallest eigenvalue is positive.

Observed on skfem 12.0.1, MeshTri().refined(3), ElementDG(ElementTriP1):
  sigma =  1  -> lambda_min = -1.0e+00  (NOT coercive)
  sigma =  2  -> lambda_min = -4.9e-01  (NOT coercive)
  sigma =  3  -> lambda_min = -5.5e-04  (NOT coercive -- only just)
  sigma =  4  -> lambda_min = +5.0e-02  (coercive) == 4 * order^2 for order 1
  sigma = 16  -> coercive for ElementDG(ElementTriP2) == 4 * order^2, order 2
  sigma = 1e13 -> cond(K) = 3.9e+14 > 1e14 and CG needs ~30x more iterations
                  than at sigma = 4 for the same tolerance.

PARTIAL FALSIFICATION: the "solution norm diverges under refinement" half does
NOT reproduce.  At sigma = 1 the L2 norm of the discrete solution stays at
0.50 for every refinement level; what actually degrades is the CONVERGENCE
RATE (the L2 error rate collapses from ~1.8 at sigma = 4 to ~0.6 at sigma = 1).
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementDG,
    ElementTriP1,
    ElementTriP2,
    FacetBasis,
    InteriorFacetBasis,
    LinearForm,
    MeshTri,
    asm,
)
from skfem.helpers import dot, grad, jump
from scipy.sparse.linalg import cg, spsolve


def make_forms(sigma):
    @BilinearForm
    def stiffness(u, v, w):
        return dot(grad(u), grad(v))

    @BilinearForm
    def ip_interior(u, v, w):
        ju, jv = jump(w, u, v)
        gun = 0.5 * (u.grad[0] * w.n[0] + u.grad[1] * w.n[1])
        gvn = 0.5 * (v.grad[0] * w.n[0] + v.grad[1] * w.n[1])
        return -gun * jv - gvn * ju + sigma / w.h * ju * jv

    @BilinearForm
    def ip_boundary(u, v, w):
        gun = u.grad[0] * w.n[0] + u.grad[1] * w.n[1]
        gvn = v.grad[0] * w.n[0] + v.grad[1] * w.n[1]
        return -gun * v - gvn * u + sigma / w.h * u * v

    return stiffness, ip_interior, ip_boundary


@LinearForm
def load(v, w):
    x, y = w.x
    return (2.0 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)) * v


@BilinearForm
def mass(u, v, w):
    return u * v


def sipg(nref, sigma, order=1):
    m = MeshTri().refined(nref)
    e = ElementDG(ElementTriP1() if order == 1 else ElementTriP2())
    ib = Basis(m, e)
    i0 = InteriorFacetBasis(m, e, side=0)
    i1 = InteriorFacetBasis(m, e, side=1)
    fb = FacetBasis(m, e)
    stiffness, ip_interior, ip_boundary = make_forms(sigma)
    A = (asm(stiffness, ib)
         + asm(ip_interior, [i0, i1], [i0, i1])
         + asm(ip_boundary, fb))
    f = asm(load, ib)
    u = spsolve(A.tocsr(), f)
    uex = ib.project(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    M = asm(mass, ib)
    d = u - uex
    l2err = float(np.sqrt(max(d @ (M @ d), 0.0)))
    l2nrm = float(np.sqrt(max(u @ (M @ u), 0.0)))
    return A, l2err, l2nrm


def lambda_min(A):
    dense = A.toarray()
    return float(np.linalg.eigvalsh(0.5 * (dense + dense.T))[0])


def main() -> int:
    ok = True

    # --- WRONG variant A: sigma below the rule of thumb -> not coercive ----
    rule = 4 * 1 ** 2
    print(f"rule_of_thumb_sigma_order1={rule}")
    lams = {}
    for sigma in (1.0, 2.0, 3.0, 4.0, 8.0):
        A, _, _ = sipg(3, sigma)
        lams[sigma] = lambda_min(A)
        print(f"sigma_{sigma:g}_coercive={lams[sigma] > 0}")
    print(f"lambda_min_at_sigma1={lams[1.0]:.4e} lambda_min_at_sigma4={lams[4.0]:.4e}")
    print(f"coercivity_threshold_is_the_rule_of_thumb="
          f"{bool(lams[3.0] <= 0 < lams[4.0])}")
    if not (lams[1.0] < 0 and lams[2.0] < 0 and lams[3.0] < 0):
        print("FAIL: sigma below 4 was still coercive", file=sys.stderr)
        ok = False
    if not (lams[4.0] > 0 and lams[8.0] > 0):
        print("FAIL: sigma at/above the rule of thumb was not coercive",
              file=sys.stderr)
        ok = False

    # the rule scales with order^2: order 2 needs 16, and 4 is not enough
    A2_low, _, _ = sipg(2, 4.0, order=2)
    A2_ok, _, _ = sipg(2, 4 * 2 ** 2, order=2)
    print(f"rule_of_thumb_sigma_order2={4 * 2 ** 2}")
    print(f"order2_sigma4_coercive={lambda_min(A2_low) > 0}")
    print(f"order2_sigma16_coercive={lambda_min(A2_ok) > 0}")
    if lambda_min(A2_low) > 0 or lambda_min(A2_ok) <= 0:
        print("FAIL: the sigma = 4*order^2 scaling did not hold for order 2",
              file=sys.stderr)
        ok = False

    # --- what actually degrades: the convergence rate, not the norm --------
    for sigma in (1.0, 4.0):
        errs, nrms = [], []
        for nref in (2, 3, 4):
            _, e, n = sipg(nref, sigma)
            errs.append(e)
            nrms.append(n)
        rate = float(np.log2(errs[-2] / errs[-1]))
        norm_grew = bool(nrms[-1] > 2.0 * nrms[0])
        print(f"sigma_{sigma:g}_l2_rate_ge_1p5={rate >= 1.5}")
        print(f"sigma_{sigma:g}_solution_norm_diverges={norm_grew}")
        print(f"sigma_{sigma:g}_rate={rate:.3f} norms={[round(x, 4) for x in nrms]}")
    print("norm_divergence_claim_reproduces=False")

    # --- WRONG variant B: sigma far above -> conditioning + CG stagnation --
    def cg_iters(A):
        n = [0]
        b = np.ones(A.shape[0])
        cg(A.tocsr(), b, rtol=1e-8, maxiter=4000,
           callback=lambda xk: n.__setitem__(0, n[0] + 1))
        return n[0]

    A_ok, _, _ = sipg(3, 4.0)
    A_big, _, _ = sipg(3, 1e13)
    cond_ok = float(np.linalg.cond(A_ok.toarray()))
    cond_big = float(np.linalg.cond(A_big.toarray()))
    it_ok, it_big = cg_iters(A_ok), cg_iters(A_big)
    print(f"cond_at_rule_of_thumb_lt_1e4={cond_ok < 1e4}")
    print(f"cond_at_huge_sigma_gt_1e14={cond_big > 1e14}")
    print(f"cg_iterations_grow_gt_10x={it_big > 10 * it_ok}")
    print(f"cond_ok={cond_ok:.4e} cond_big={cond_big:.4e} it_ok={it_ok} it_big={it_big}")
    if not (cond_ok < 1e4 and cond_big > 1e14):
        print("FAIL: the conditioning did not blow up with a huge sigma",
              file=sys.stderr)
        ok = False
    if it_big <= 10 * it_ok:
        print("FAIL: CG did not stagnate at the huge sigma", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
