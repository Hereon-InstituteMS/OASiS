"""Tier-2: dropping the I4_sym_C^-1 term from the Neo-Hookean tangent.

Claim: skfem hyperelasticity#4 -- the material tangent is
C4 = lam * C^-1 (x) C^-1 + 2*(mu - lam*lnJ) * I4_sym_C^-1; dropping the
I4_sym_C^-1 term and using a scalar-multiplied identity I4 instead gives
"Newton convergence rate ~0.5 (linear instead of quadratic)"; the exact
C^-1-built tangent restores quadratic convergence in the
asm + condense + spsolve pipeline.

Measured on skfem 12.0.1 with a total-Lagrangian PK2 formulation
(MeshTri().refined(3), ElementVector(ElementTriP1()), mu = lam = 1, left edge
clamped, right edge pulled, residual int S : dE(v) dx, tangent
int dE(v) : C4 : dE(du) dx + int S : grad(du)^T grad(v) dx):

  * THE ORDER CLAIM IS CONFIRMED.  With the exact I4_sym_C^-1 term Newton
    reaches machine zero in three or four iterations at every stretch tested;
    replacing it with 2*(mu - lam*lnJ)*dE (a scalar times the identity
    fourth-order tensor) turns the residual history into a straight line on a
    log plot -- a constant ratio per iteration, i.e. exactly linear.
  * THE RATE IS NOT A FIXED ~0.5.  The measured linear factor depends on how
    far the state is from the reference configuration: it is small at a few
    percent stretch and rises substantially at large stretch, so the
    iteration count blows up from single digits to tens.  The entry's single
    number should not be treated as a diagnostic threshold.
  * IT ALWAYS CONVERGES.  The inexact tangent never diverges and never
    produces NaN; it only costs iterations.  Nothing is printed.  The failure
    mode is a slow solve, not a wrong answer -- both tangents land on the
    same displacement field to machine precision.
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


def ident_like(A):
    out = np.zeros_like(A)
    out[0, 0] = 1.0
    out[1, 1] = 1.0
    return out


def kinematics(w):
    du = w["disp"].grad
    F = ident_like(du) + du
    C = np.einsum("ki...,kj...->ij...", F, F)
    Cinv = inv(C)
    with np.errstate(all="ignore"):
        lnJ = np.log(det(F))
    S = MU * (ident_like(C) - Cinv) + LAM * lnJ * Cinv
    return F, Cinv, lnJ, S


def green_lagrange_variation(F, G):
    A = np.einsum("ki...,kj...->ij...", F, G)
    return 0.5 * (A + transpose(A))


@LinearForm
def residual(v, w):
    F, _, _, S = kinematics(w)
    return ddot(S, green_lagrange_variation(F, grad(v)))


def tangent(exact_i4sym):
    @BilinearForm
    def form(u, v, w):
        F, Cinv, lnJ, S = kinematics(w)
        dEu = green_lagrange_variation(F, grad(u))
        dEv = green_lagrange_variation(F, grad(v))
        if exact_i4sym:
            sandwich = np.einsum("ik...,kl...,lj...->ij...", Cinv, dEu, Cinv)
            sandwich = 0.5 * (sandwich + transpose(sandwich))
        else:
            # scalar-multiplied identity I4 -- the C^-1 sandwich dropped
            sandwich = dEu
        c4de = LAM * ddot(Cinv, dEu) * Cinv + 2.0 * (MU - LAM * lnJ) * sandwich
        geom = np.einsum("ki...,kj...->ij...", grad(u), grad(v))
        return ddot(c4de, dEv) + ddot(S, geom)
    return form


def setup(refine=3):
    m = MeshTri().refined(refine)
    ib = Basis(m, ElementVector(ElementTriP1()))
    left = ib.get_dofs(lambda x: x[0] < 1e-10)
    right = ib.get_dofs(lambda x: x[0] > 1.0 - 1e-10)
    return m, ib, np.concatenate([left.all(), right.all()]), right


def newton(exact_i4sym, stretch, nit=60, tol=1e-13):
    m, ib, D, right = setup()
    xbc = ib.zeros()
    xbc[right.nodal["u^1"]] = stretch
    free = np.setdiff1d(np.arange(ib.N), D)
    K_form = tangent(exact_i4sym)
    u = ib.zeros()
    hist, msgs = [], []
    for _ in range(nit):
        w = ib.interpolate(u)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            R = residual.assemble(ib, disp=w)
            K = K_form.assemble(ib, disp=w)
            u = u + solve(*condense(K, -R, x=xbc - u, D=D))
            msgs += [str(c.message) for c in caught]
        hist.append(float(np.linalg.norm(
            residual.assemble(ib, disp=ib.interpolate(u))[free])))
        if not np.isfinite(hist[-1]) or hist[-1] < tol:
            break
    return u, hist, sorted(set(msgs))


def main() -> int:
    ok = True
    m, ib, _, _ = setup()
    print(f"dofs_N={ib.N} elements={m.t.shape[1]}")

    factors = {}
    for stretch in (0.02, 0.30):
        tag = f"s{int(stretch * 100):02d}"
        u_ok, h_ok, w_ok = newton(True, stretch)
        u_bad, h_bad, w_bad = newton(False, stretch)
        print(f"{tag}_exact_i4sym_iters={len(h_ok)}")
        print(f"{tag}_dropped_i4sym_iters={len(h_bad)}")
        print(f"{tag}_exact_history={[f'{x:.2e}' for x in h_ok[:4]]}")
        print(f"{tag}_dropped_history={[f'{x:.2e}' for x in h_bad[:5]]}")
        print(f"{tag}_dropped_needs_more_iterations={len(h_bad) > len(h_ok)}")

        # quadratic: e_{k+1} ~ e_k^2  ->  ratio collapses
        # linear:    e_{k+1} ~ q e_k  ->  ratio constant
        r_ok = [h_ok[i + 1] / h_ok[i] for i in range(len(h_ok) - 1)
                if h_ok[i] > 0]
        # the linear factor is asymptotic: skip the first iterations, which
        # are a transient, and read the rate off a late window
        r_bad = [h_bad[i + 1] / h_bad[i] for i in range(4, 10)
                 if i + 1 < len(h_bad) and h_bad[i] > 0]
        print(f"{tag}_exact_ratios={[f'{x:.2e}' for x in r_ok[:3]]}")
        print(f"{tag}_dropped_ratios={[f'{x:.3f}' for x in r_bad]}")
        spread = (max(r_bad) - min(r_bad)) if r_bad else 1.0
        print(f"{tag}_dropped_rate_is_constant_linear={spread < 0.05}")
        print(f"{tag}_dropped_mean_linear_factor="
              f"{float(np.mean(r_bad)):.3f}")
        factors[tag] = float(np.mean(r_bad))
        print(f"{tag}_exact_ratio_collapses="
              f"{bool(len(r_ok) >= 2 and r_ok[1] < r_ok[0] ** 1.5)}")
        print(f"{tag}_dropped_converged={bool(h_bad[-1] < 1e-13)}")
        print(f"{tag}_dropped_finite={bool(np.isfinite(u_bad).all())}")
        print(f"{tag}_dropped_warnings={w_bad!r}")
        rel = float(np.linalg.norm(u_bad - u_ok) / np.linalg.norm(u_ok))
        print(f"{tag}_same_answer={rel < 1e-9} rel_diff={rel:.2e}")

        if spread >= 0.05:
            print(f"FAIL: at stretch {stretch} the dropped-I4sym rate was not "
                  f"a constant linear factor", file=sys.stderr)
            ok = False
        if len(h_bad) <= len(h_ok):
            print(f"FAIL: at stretch {stretch} dropping I4_sym_Cinv cost no "
                  f"iterations", file=sys.stderr)
            ok = False
        if not h_bad[-1] < 1e-13:
            print(f"FAIL: at stretch {stretch} the inexact tangent did not "
                  f"converge at all", file=sys.stderr)
            ok = False
        if rel >= 1e-9:
            print(f"FAIL: at stretch {stretch} the inexact tangent changed "
                  f"the answer", file=sys.stderr)
            ok = False

    print(f"linear_factor_grows_with_stretch="
          f"{factors['s30'] > factors['s02']}")
    if factors["s30"] <= factors["s02"]:
        print("FAIL: the linear factor did not depend on the stretch, so a "
              "single quoted rate would have been safe", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
