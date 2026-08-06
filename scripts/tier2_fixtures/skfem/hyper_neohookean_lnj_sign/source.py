"""Tier-2: the sign of the -mu*lnJ term in the Neo-Hookean energy.

Claim: skfem hyperelasticity#1 -- W = mu/2*(I1-2) - mu*lnJ + lam/2*(lnJ)^2;
flipping the sign on the lnJ term gives "a Newton tangent that is symmetric
but with the WRONG sign -- Newton diverges with the residual growing by a
factor ~10 per iteration".  The claim also prescribes the sanity check:
"at F = I, W must be 0 and P must be 0; verify before stepping."

What was actually measured on skfem 12.0.1 (MeshTri().refined(3),
ElementVector(ElementTriP1()), mu = lam = 1, left edge clamped, right edge
pulled to 5 % stretch, hand-coded PK1 residual + exact tangent):

  * the PRESCRIBED SANITY CHECK IS THE REAL SIGNAL.  With the correct
    P = mu*(F - F^-T) + lam*lnJ*F^-T the assembled residual at the
    undeformed state is EXACTLY zero; with the flipped sign
    P = mu*(F + F^-T) + lam*lnJ*F^-T it is O(1) at F = I.  That is
    detectable before a single Newton step is taken.
  * "tangent ... with the WRONG sign" is NOT what happens.  The flipped
    tangent is symmetric AND still positive definite on the free DOFs --
    zero negative eigenvalues, smallest eigenvalue positive.  So a
    definiteness check does not catch this bug; only the F = I residual
    does.
  * "residual growing by a factor ~10 per iteration" is NOT what happens
    either.  The very first Newton step overshoots the 5 % boundary
    stretch by more than an order of magnitude, which drives det(F) <= 0,
    and log(J) then returns NaN: the run is NaN-everywhere from
    iteration 1 onward.  There is no ~10x-per-iteration ramp to watch.

This fixture asserts the mechanism, not the magnitudes.

Mutation control: T2_MUTATE=1 applies the documented fix at the two pathology
sites -- the sign of the lnJ term.  pk1()'s `+Fit` branch becomes `-Fit` and
tangent_form()'s flipped `coef` becomes the correct `(MU - LAM*lnJ)`, so the
"flipped" law IS the correct law under mutation.  The flipped run then becomes
stress-free at F = I and converges, which removes
flipped_stress_free_at_reference=False, F_eq_I_check_discriminates=True,
flipped_first_step_overshoots_bc_by_gt_10x=True and flipped_reaches_nan=True.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
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

MUTATE = os.environ.get("T2_MUTATE") == "1"


def deformation_gradient(w):
    du = w["disp"].grad
    ident = np.zeros_like(du)
    ident[0, 0] = 1.0
    ident[1, 1] = 1.0
    return ident + du


def pk1(F, flipped):
    """P for W = mu/2*(I1-2) -/+ mu*lnJ + lam/2*lnJ^2."""
    J = det(F)
    Fit = transpose(inv(F))
    with np.errstate(invalid="ignore", divide="ignore"):
        lnJ = np.log(J)
    # correct: mu*(F - F^-T);  flipped +mu*lnJ in W gives mu*(F + F^-T)
    # PATHOLOGY: the +Fit branch.  T2_MUTATE=1 applies the documented fix
    # (W uses -mu*lnJ, hence P = mu*(F - F^-T)) so the flipped law is correct.
    bad = flipped and not MUTATE
    return MU * (F + (Fit if bad else -Fit)) + LAM * lnJ * Fit


def residual_form(flipped):
    @LinearForm
    def form(v, w):
        return ddot(pk1(deformation_gradient(w), flipped), grad(v))
    return form


def tangent_form(flipped):
    @BilinearForm
    def form(u, v, w):
        F = deformation_gradient(w)
        J = det(F)
        Fit = transpose(inv(F))
        with np.errstate(invalid="ignore", divide="ignore"):
            lnJ = np.log(J)
        dF = grad(u)
        FitdFtFit = np.einsum("ij...,kj...,kl...->il...", Fit, dF, Fit)
        # PATHOLOGY (tangent, consistent with pk1): the flipped coefficient.
        # T2_MUTATE=1 restores the documented (MU - LAM*lnJ).
        coef = (-(MU + LAM * lnJ) if (flipped and not MUTATE)
                else (MU - LAM * lnJ))
        term = MU * dF + coef * FitdFtFit + LAM * ddot(Fit, dF) * Fit
        return ddot(term, grad(v))
    return form


def setup(refine=3):
    m = MeshTri().refined(refine)
    ib = Basis(m, ElementVector(ElementTriP1()))
    left = ib.get_dofs(lambda x: x[0] < 1e-10)
    right = ib.get_dofs(lambda x: x[0] > 1.0 - 1e-10)
    D = np.concatenate([left.all(), right.all()])
    return m, ib, D, right


def newton(flipped, stretch=0.05, nit=4):
    m, ib, D, right = setup()
    xbc = ib.zeros()
    xbc[right.nodal["u^1"]] = stretch
    free = np.setdiff1d(np.arange(ib.N), D)
    u = ib.zeros()
    R_form, K_form = residual_form(flipped), tangent_form(flipped)
    hist, peak = [], []
    for _ in range(nit):
        w = ib.interpolate(u)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            R = R_form.assemble(ib, disp=w)
            K = K_form.assemble(ib, disp=w)
            du = solve(*condense(K, -R, x=xbc - u, D=D))
        u = u + du
        hist.append(float(np.linalg.norm(R[free])))
        peak.append(float(np.abs(u).max()))
    return u, hist, peak


def main() -> int:
    ok = True
    m, ib, D, _ = setup()
    print(f"dofs_N={ib.N} elements={m.t.shape[1]}")

    # --- the sanity check the claim itself prescribes -----------------
    w0 = ib.interpolate(ib.zeros())
    r_ok = residual_form(False).assemble(ib, disp=w0)
    r_bad = residual_form(True).assemble(ib, disp=w0)
    n_ok = float(np.linalg.norm(r_ok))
    n_bad = float(np.linalg.norm(r_bad))
    print(f"correct_residual_at_F_eq_I={n_ok:.3e}")
    print(f"flipped_residual_at_F_eq_I={n_bad:.3e}")
    print(f"correct_stress_free_at_reference={n_ok < 1e-14}")
    print(f"flipped_stress_free_at_reference={n_bad < 1e-14}")
    print(f"F_eq_I_check_discriminates={n_ok < 1e-14 and n_bad > 1e-2}")
    if not (n_ok < 1e-14 and n_bad > 1e-2):
        print("FAIL: the F=I residual did not separate the two signs",
              file=sys.stderr)
        ok = False

    # --- is the flipped tangent really "of the wrong sign"? -----------
    free = np.setdiff1d(np.arange(ib.N), D)
    for label, flipped in (("correct", False), ("flipped", True)):
        K = tangent_form(flipped).assemble(ib, disp=w0).toarray()
        sym = bool(np.abs(K - K.T).max() < 1e-12)
        ev = np.linalg.eigvalsh(K[np.ix_(free, free)])
        print(f"{label}_tangent_symmetric={sym}")
        print(f"{label}_tangent_negative_eigenvalues={int((ev < 0).sum())}")
        print(f"{label}_tangent_positive_definite={bool(ev.min() > 0)}")
        if label == "flipped":
            if not sym:
                print("FAIL: the flipped tangent was not symmetric",
                      file=sys.stderr)
                ok = False
            if ev.min() <= 0:
                print("FAIL: the flipped tangent was indefinite, so a "
                      "definiteness check WOULD have caught this bug",
                      file=sys.stderr)
                ok = False

    # --- what Newton actually does ------------------------------------
    u_ok, h_ok, p_ok = newton(False)
    u_bad, h_bad, p_bad = newton(True)
    print(f"correct_residual_history={[f'{x:.2e}' for x in h_ok]}")
    print(f"flipped_residual_history={[f'{x:.2e}' for x in h_bad]}")
    print(f"flipped_first_step_peak_disp={p_bad[0]:.4e}")
    print(f"applied_boundary_stretch=0.05")
    print(f"flipped_first_step_overshoots_bc_by_gt_10x="
          f"{p_bad[0] > 10 * 0.05}")
    print(f"correct_converges_quadratically="
          f"{h_ok[1] > 0 and h_ok[3] < 1e-12 and h_ok[2] < h_ok[1] ** 1.5}")
    print(f"flipped_reaches_nan={bool(np.isnan(u_bad).any())}")
    print(f"correct_stays_finite={bool(np.isfinite(u_ok).all())}")
    # the magnitude the claim predicted, measured rather than assumed
    ramp = [h_bad[i + 1] / h_bad[i] if h_bad[i] > 0 else float("nan")
            for i in range(len(h_bad) - 1)]
    print(f"flipped_residual_ratios={[f'{x:.2e}' for x in ramp]}")
    print(f"flipped_shows_steady_10x_growth="
          f"{all(np.isfinite(r) and 5 < r < 20 for r in ramp)}")

    if not np.isnan(u_bad).any():
        print("FAIL: the flipped sign produced a finite solution",
              file=sys.stderr)
        ok = False
    if not np.isfinite(u_ok).all():
        print("FAIL: the correct sign did not stay finite", file=sys.stderr)
        ok = False
    if not (p_bad[0] > 10 * 0.05):
        print("FAIL: the flipped first step did not overshoot the BC",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
