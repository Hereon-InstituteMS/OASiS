"""Tier-2: feeding the 2nd Piola-Kirchhoff stress into the PK1 weak form.

Claim: skfem hyperelasticity#2 -- P = mu*(F - F^-T) + lam*lnJ*F^-T; using the
2nd-PK form S = mu*(I - C^-1) + lam*lnJ*C^-1 inside the @BilinearForm and
feeding it directly to int P:grad(v) dx "adds a missing F-pre-multiplication
-- the asm residual is wrong by a factor of F and Newton diverges."

Measured on skfem 12.0.1 (MeshTri().refined(3), ElementVector(ElementTriP1()),
mu = lam = 1, left edge clamped, right edge pulled to a fixed stretch, Newton
driven by the exact PK1 tangent):

  * THE ALGEBRA IS CONFIRMED.  F @ S reproduces mu*(F - F^-T) + lam*lnJ*F^-T
    to machine precision at every quadrature point, because F C^-1 = F^-T
    exactly.  So the missing factor really is a left multiplication by F, and
    the two residual vectors genuinely differ.
  * "NEWTON DIVERGES" IS FALSE.  This is the silent case, not the loud one.
    Newton on the S-in-PK1-slot residual converges -- the residual falls to
    machine zero, monotonically, with no NaN and no warning -- but only at a
    LINEAR rate (a constant residual ratio per iteration) instead of the
    quadratic rate the correct residual gives.  It converges to a DIFFERENT
    displacement field, and the discrepancy grows with the applied stretch.
    Every internal convergence test passes; nothing is printed; the answer is
    simply wrong.

The honest signal is therefore: not an exception and not a divergence, but a
degraded convergence ORDER plus a solution that moves when you fix the
pre-multiplication.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    ElementVector,
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.helpers import ddot, det, grad, inv, transpose

MU = 1.0
LAM = 1.0


def deformation_gradient(w):
    du = w["disp"].grad
    ident = np.zeros_like(du)
    ident[0, 0] = 1.0
    ident[1, 1] = 1.0
    return ident + du


def pk1(F):
    Fit = transpose(inv(F))
    return MU * (F - Fit) + LAM * np.log(det(F)) * Fit


def pk2(F):
    """S = mu*(I - C^-1) + lam*lnJ*C^-1 with C = F^T F."""
    C = np.einsum("ki...,kj...->ij...", F, F)
    Cinv = inv(C)
    ident = np.zeros_like(C)
    ident[0, 0] = 1.0
    ident[1, 1] = 1.0
    return MU * (ident - Cinv) + LAM * np.log(det(F)) * Cinv


def residual_form(use_pk2):
    @LinearForm
    def form(v, w):
        F = deformation_gradient(w)
        T = pk2(F) if use_pk2 else pk1(F)
        return ddot(T, grad(v))
    return form


@BilinearForm
def tangent_form(u, v, w):
    F = deformation_gradient(w)
    Fit = transpose(inv(F))
    lnJ = np.log(det(F))
    dF = grad(u)
    FitdFtFit = np.einsum("ij...,kj...,kl...->il...", Fit, dF, Fit)
    term = MU * dF + (MU - LAM * lnJ) * FitdFtFit + LAM * ddot(Fit, dF) * Fit
    return ddot(term, grad(v))


def setup(refine=3):
    m = MeshTri().refined(refine)
    ib = Basis(m, ElementVector(ElementTriP1()))
    left = ib.get_dofs(lambda x: x[0] < 1e-10)
    right = ib.get_dofs(lambda x: x[0] > 1.0 - 1e-10)
    return m, ib, np.concatenate([left.all(), right.all()]), right


def newton(use_pk2, stretch, nit=60):
    m, ib, D, right = setup()
    xbc = ib.zeros()
    xbc[right.nodal["u^1"]] = stretch
    free = np.setdiff1d(np.arange(ib.N), D)
    u = ib.zeros()
    R_form = residual_form(use_pk2)
    hist = []
    for _ in range(nit):
        w = ib.interpolate(u)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            R = R_form.assemble(ib, disp=w)
            K = tangent_form.assemble(ib, disp=w)
            du = solve(*condense(K, -R, x=xbc - u, D=D))
        u = u + du
        hist.append(float(np.linalg.norm(R[free])))
        if hist[-1] < 1e-15 and len(hist) > 2:
            break
    return u, hist, [str(c.message) for c in caught]


def main() -> int:
    ok = True
    m, ib, D, _ = setup()
    print(f"dofs_N={ib.N} elements={m.t.shape[1]}")

    # --- the algebra: F @ S == P exactly ------------------------------
    rng = np.random.default_rng(0)
    F = np.zeros((2, 2, m.t.shape[1], 3))
    F[0, 0] = 1.0 + 0.2 * rng.random(F[0, 0].shape)
    F[1, 1] = 1.0 + 0.2 * rng.random(F[1, 1].shape)
    F[0, 1] = 0.1 * rng.standard_normal(F[0, 1].shape)
    F[1, 0] = 0.1 * rng.standard_normal(F[1, 0].shape)
    P = pk1(F)
    S = pk2(F)
    FS = np.einsum("ik...,kj...->ij...", F, S)
    algebra_err = float(np.abs(FS - P).max())
    print(f"max_abs_F_times_S_minus_P={algebra_err:.3e}")
    print(f"missing_factor_is_exactly_F={algebra_err < 1e-13}")
    print(f"max_abs_S_minus_P={float(np.abs(S - P).max()):.3e}")
    print(f"S_and_P_are_different={float(np.abs(S - P).max()) > 1e-3}")
    if algebra_err >= 1e-13:
        print("FAIL: F @ S did not reproduce P", file=sys.stderr)
        ok = False

    # --- what Newton actually does -----------------------------------
    for stretch in (0.05, 0.20):
        tag = f"s{int(stretch * 100):02d}"
        u_ok, h_ok, _ = newton(False, stretch)
        u_bad, h_bad, warns = newton(True, stretch)
        conv_ok = h_ok[-1] < 1e-13
        conv_bad = h_bad[-1] < 1e-13
        print(f"{tag}_correct_converged={conv_ok} iters={len(h_ok)}")
        print(f"{tag}_pk2_in_pk1_slot_converged={conv_bad} "
              f"iters={len(h_bad)}")
        print(f"{tag}_pk2_in_pk1_slot_diverged={not conv_bad}")
        print(f"{tag}_pk2_finite={bool(np.isfinite(u_bad).all())}")
        mono = all(h_bad[i + 1] <= h_bad[i] for i in range(1, len(h_bad) - 1))
        print(f"{tag}_pk2_residual_monotone_decreasing={mono}")
        print(f"{tag}_pk2_residual_first_to_last="
              f"{h_bad[1]:.2e}->{h_bad[-1]:.2e}")
        print(f"{tag}_pk2_warnings={warns!r}")
        # convergence ORDER: quadratic vs linear
        ratios = [h_ok[i + 1] / h_ok[i] for i in range(1, min(4, len(h_ok) - 1))
                  if h_ok[i] > 0]
        ratios_bad = [h_bad[i + 1] / h_bad[i]
                      for i in range(1, min(6, len(h_bad) - 1))
                      if h_bad[i] > 0]
        print(f"{tag}_correct_ratios={[f'{r:.2e}' for r in ratios]}")
        print(f"{tag}_pk2_ratios={[f'{r:.3f}' for r in ratios_bad]}")
        tail = ratios_bad[1:] or ratios_bad
        spread = (max(tail) - min(tail)) if tail else 1.0
        print(f"{tag}_pk2_ratio_is_constant_linear_rate={spread < 0.05}")
        if spread >= 0.05:
            print(f"FAIL: at stretch {stretch} the S-in-PK1-slot residual "
                  f"ratio was not a constant linear rate", file=sys.stderr)
            ok = False
        rel = float(np.linalg.norm(u_bad - u_ok) / np.linalg.norm(u_ok))
        print(f"{tag}_relative_solution_difference={rel:.4e}")
        print(f"{tag}_answers_differ={rel > 1e-3}")
        if not conv_bad:
            print(f"FAIL: at stretch {stretch} the S-in-PK1-slot Newton did "
                  f"NOT converge, so the claim's 'diverges' would hold",
                  file=sys.stderr)
            ok = False
        if not (rel > 1e-3):
            print(f"FAIL: at stretch {stretch} the wrong residual gave the "
                  f"same answer", file=sys.stderr)
            ok = False
        if warns:
            print("FAIL: the wrong residual emitted a warning, so it is not "
                  "silent after all", file=sys.stderr)
            ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
