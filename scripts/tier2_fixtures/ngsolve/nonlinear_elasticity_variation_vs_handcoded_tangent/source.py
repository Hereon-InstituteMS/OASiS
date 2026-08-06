"""Tier-2: a hand-coded tangent that does not match the residual converges
linearly; NGSolve's own linearisation of the same residual converges
quadratically.

Claim: ngsolve nonlinear_elasticity#3 -- Variation() auto-differentiates the
energy for both residual and tangent, which is much safer than hand-coding.  The
catalog's stated signal is that "a hand-coded P(F) and dP/dF with a sign error or
factor-of-2 produces linear (not quadratic) Newton convergence".

Wrong variants exercised here (2-D unit_square maxh 0.4, VectorH1 order 2,
Neo-Hookean, 8 % prescribed stretch, hand-rolled Newton so the tangent can be
chosen independently of the residual):
  (a) exact residual  +  tangent linearised from P with the mu term doubled;
  (b) P with the mu term doubled used for BOTH residual and tangent.

Observed on NGSolve 6.2.2604:
  * the hand-coded P = mu*F + (lam*ln J - mu)*F^-T reproduces the residual of
    Variation(W*dx) to ~6e-15, so P itself is right;
  * (a) needs 39 residual evaluations instead of 5, and the ratio
    |r_{k+1}|/|r_k| settles on a constant ~0.55 -- textbook LINEAR convergence --
    while reaching the SAME solution;
  * matched tangent gives |r_{k+1}| / |r_k|^2 bounded ~1e-3 -- QUADRATIC.
  * (b) is the correction to the claim: because NGSolve differentiates whatever
    residual form you hand it, a factor-of-2 error inside P does NOT slow
    convergence -- it converges quadratically in 6 steps to a WRONG displacement
    (max|u| 0.159 vs the correct 0.080).  The linear signature appears only when
    the tangent is inconsistent with the residual.

Mutation control: T2_MUTATE=1 sets P_TANGENT_FAC from 2.0 to 1.0, i.e. it removes
the factor-of-2 error from the hand-coded P -- the documented fix, at the
pathology site.  a_wrong then equals a_exact, so both wrong variants collapse onto
the matched-tangent run and 'mismatched_tangent_iters_ge_4x_exact=True',
'mismatched_tangent_ratio_constant_linear=True' and 'wrongP_solution_is_wrong=True'
go missing; the fixture goes red.
Re-run: T2_MUTATE=1 python source.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    Det,
    GridFunction,
    Grad,
    Id,
    InnerProduct,
    Inv,
    Mesh,
    Norm,
    Projector,
    Trace,
    Variation,
    VectorH1,
    dx,
    log,
    unit_square,
    x,
    y,
)

E, NU = 200.0, 0.3
MU = E / (2 * (1 + NU))
LAM = E * NU / ((1 + NU) * (1 - 2 * NU))

DISP = 0.08
TOL = 1e-10
MAXIT = 60

# Factor multiplying the mu term of the hand-coded P used for the TANGENT only.
# Mutation control: T2_MUTATE=1 sets it to 1.0, removing the factor-of-2 error.
MUTATE = os.environ.get("T2_MUTATE") == "1"
P_TANGENT_FAC = 1.0 if MUTATE else 2.0


def main() -> int:
    ok = True

    mesh = Mesh(unit_square.GenerateMesh(maxh=0.4))
    fes = VectorH1(mesh, order=2, dirichlet="left|right")
    u, v = fes.TnT()
    ident = Id(2)

    def pk1(fac: float):
        F = ident + Grad(u)
        J = Det(F)
        return fac * MU * F + (LAM * log(J) - MU) * Inv(F).trans

    F = ident + Grad(u)
    C = F.trans * F
    J = Det(F)
    W = 0.5 * MU * (Trace(C) - 2) - MU * log(J) + 0.5 * LAM * log(J) ** 2

    a_var = BilinearForm(fes, symmetric=True)
    a_var += Variation(W * dx)
    a_exact = BilinearForm(fes)
    a_exact += InnerProduct(pk1(1.0), Grad(v)) * dx
    a_wrong = BilinearForm(fes)
    a_wrong += InnerProduct(pk1(P_TANGENT_FAC), Grad(v)) * dx

    # --- the hand-coded P is itself correct --------------------------------
    probe = GridFunction(fes)
    probe.Set(CoefficientFunction((0.03 * x, 0.01 * y)))
    r_var = probe.vec.CreateVector()
    r_hand = probe.vec.CreateVector()
    diff = probe.vec.CreateVector()
    a_var.Apply(probe.vec, r_var)
    a_exact.Apply(probe.vec, r_hand)
    diff.data = r_var - r_hand
    rel = Norm(diff) / Norm(r_var)
    print(f"hand_P_matches_variation_residual={rel < 1e-12}")
    if rel >= 1e-12:
        print(f"FAIL: hand-coded PK1 disagrees with Variation() "
              f"(rel={rel:.3e})", file=sys.stderr)
        ok = False

    proj_free = Projector(fes.FreeDofs(), True)

    def newton(a_res, a_tan):
        gfu = GridFunction(fes)
        gfu.Set(CoefficientFunction((DISP, 0.0)),
                definedon=mesh.Boundaries("right"))
        r = gfu.vec.CreateVector()
        rf = gfu.vec.CreateVector()
        w = gfu.vec.CreateVector()
        hist = []
        mat_type = ""
        for _it in range(MAXIT):
            a_res.Apply(gfu.vec, r)
            rf.data = proj_free * r
            hist.append(Norm(rf))
            if hist[-1] < TOL:
                break
            a_tan.AssembleLinearization(gfu.vec)
            mat_type = type(a_tan.mat).__name__
            w.data = a_tan.mat.Inverse(fes.FreeDofs()) * r
            gfu.vec.data -= w
        return hist, float(np.abs(gfu.vec.FV().NumPy()).max()), mat_type

    h_exact, u_exact, mat_type = newton(a_exact, a_exact)
    h_mis, u_mis, _ = newton(a_exact, a_wrong)
    h_bad, u_bad, _ = newton(a_wrong, a_wrong)
    print(f"tangent_matrix_type={mat_type}")

    # --- RIGHT variant: matched (Variation-grade) tangent -> quadratic ------
    q = [h_exact[i + 1] / h_exact[i] ** 2 for i in range(len(h_exact) - 2)]
    print(f"exact_tangent_n_residuals={len(h_exact)}")
    print(f"exact_tangent_converged={h_exact[-1] < TOL}")
    print(f"exact_tangent_quadratic={all(c < 1.0 for c in q) and len(q) >= 2}")
    if not (h_exact[-1] < TOL and len(h_exact) <= 8):
        print(f"FAIL: matched tangent did not converge quadratically "
              f"(n={len(h_exact)}, last={h_exact[-1]:.3e})", file=sys.stderr)
        ok = False

    # --- WRONG variant (a): mismatched tangent -> linear -------------------
    ratios = [h_mis[i + 1] / h_mis[i] for i in range(len(h_mis) - 1)]
    tail = ratios[-5:]
    linear = (len(h_mis) >= 4 * len(h_exact)
              and all(0.05 < c < 0.95 for c in tail)
              and (max(tail) - min(tail)) < 0.02)
    print(f"mismatched_tangent_n_residuals={len(h_mis)}")
    print(f"mismatched_tangent_iters_ge_4x_exact={len(h_mis) >= 4 * len(h_exact)}")
    print(f"mismatched_tangent_ratio_constant_linear={linear}")
    print(f"mismatched_tangent_reaches_same_solution="
          f"{abs(u_mis - u_exact) < 1e-6}")
    if not linear:
        print(f"FAIL: mismatched tangent did not show linear convergence "
              f"(n={len(h_mis)}, tail={tail})", file=sys.stderr)
        ok = False

    # --- WRONG variant (b): factor-2 P everywhere -> fast but wrong --------
    qb = [h_bad[i + 1] / h_bad[i] ** 2 for i in range(len(h_bad) - 2)]
    print(f"wrongP_n_residuals={len(h_bad)}")
    print(f"wrongP_still_quadratic={h_bad[-1] < TOL and all(c < 1.0 for c in qb)}")
    print(f"wrongP_solution_is_wrong={abs(u_bad - u_exact) > 0.05}")
    if not (h_bad[-1] < TOL and abs(u_bad - u_exact) > 0.05):
        print(f"FAIL: a factor-2 error in P no longer yields a fast-but-wrong "
              f"solve (n={len(h_bad)}, u={u_bad})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
