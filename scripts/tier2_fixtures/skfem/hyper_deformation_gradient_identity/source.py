"""Tier-2: forgetting the identity in F = I + grad(u).

Claim: skfem hyperelasticity#3 -- F = I + grad(u); forgetting the identity in
the @BilinearForm body gives a degenerate F at the reference configuration
(F = 0 instead of I), J = det(F) = 0 and lnJ = -inf, and the first Newton
residual coming out of asm is either NaN or several orders of magnitude too
large.

Measured on skfem 12.0.1 (MeshTri().refined(3), ElementVector(ElementTriP1()),
mu = lam = 1, left edge clamped, right edge pulled to 5 % stretch):

  * at u = 0 the correct kernel gives det(F) = 1 at every quadrature point and
    an EXACTLY ZERO residual; the identity-less kernel gives det(F) = 0
    everywhere, log(0) = -inf, and a residual that is NaN in every entry --
    so it is the NaN branch of the claim, not the "too large" branch.
  * the resulting displacement is NaN on every FREE degree of freedom but
    stays finite on the constrained ones, because condense writes the
    prescribed boundary values in verbatim.  A gate that inspects only the
    boundary, or that tests np.isnan(u).all() rather than isfinite(u).all(),
    therefore misses it.
  * the ONLY thing emitted is one numpy RuntimeWarning, "invalid value
    encountered in divide", from skfem.helpers.inv.  Nothing raises, and the
    Newton loop runs its full iteration budget producing NaN displacements at
    exit code 0.
  * the damage is present at the very first assembly, before any solve, so
    the cheapest guard is asserting det(F) > 0 at u = 0.
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


def deformation_gradient(w, add_identity):
    du = w["disp"].grad
    if not add_identity:
        return du
    ident = np.zeros_like(du)
    ident[0, 0] = 1.0
    ident[1, 1] = 1.0
    return ident + du


def residual_form(add_identity):
    @LinearForm
    def form(v, w):
        F = deformation_gradient(w, add_identity)
        Fit = transpose(inv(F))
        with np.errstate(all="ignore"):
            P = MU * (F - Fit) + LAM * np.log(det(F)) * Fit
        return ddot(P, grad(v))
    return form


def tangent_form(add_identity):
    @BilinearForm
    def form(u, v, w):
        F = deformation_gradient(w, add_identity)
        Fit = transpose(inv(F))
        with np.errstate(all="ignore"):
            lnJ = np.log(det(F))
        dF = grad(u)
        T = np.einsum("ij...,kj...,kl...->il...", Fit, dF, Fit)
        return ddot(MU * dF + (MU - LAM * lnJ) * T
                    + LAM * ddot(Fit, dF) * Fit, grad(v))
    return form


def setup(refine=3):
    m = MeshTri().refined(refine)
    ib = Basis(m, ElementVector(ElementTriP1()))
    left = ib.get_dofs(lambda x: x[0] < 1e-10)
    right = ib.get_dofs(lambda x: x[0] > 1.0 - 1e-10)
    return m, ib, np.concatenate([left.all(), right.all()]), right


def main() -> int:
    ok = True
    m, ib, D, right = setup()
    print(f"dofs_N={ib.N} elements={m.t.shape[1]}")

    # --- what F looks like at the reference configuration -------------
    w0 = ib.interpolate(ib.zeros())
    du = w0.grad
    for label, add_I in (("with_identity", True), ("without_identity", False)):
        F = deformation_gradient({"disp": w0}, add_I)
        J = np.asarray(det(F))
        print(f"{label}_F00_first={float(np.asarray(F[0, 0]).ravel()[0])}")
        print(f"{label}_detF_min={float(J.min())} detF_max={float(J.max())}")
        print(f"{label}_detF_all_one={bool(np.allclose(J, 1.0))}")
        print(f"{label}_detF_all_zero={bool(np.allclose(J, 0.0))}")
    print(f"grad_u_is_zero_at_reference="
          f"{bool(np.abs(np.asarray(du)).max() < 1e-14)}")

    # --- the very first residual out of asm ----------------------------
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r_ok = residual_form(True).assemble(ib, disp=w0)
        r_bad = residual_form(False).assemble(ib, disp=w0)
        msgs = sorted({str(c.message) for c in caught})
    print(f"correct_first_residual_norm={float(np.linalg.norm(r_ok)):.3e}")
    print(f"correct_first_residual_is_zero="
          f"{bool(np.linalg.norm(r_ok) < 1e-14)}")
    print(f"no_identity_first_residual_all_nan={bool(np.isnan(r_bad).all())}")
    print(f"no_identity_first_residual_finite="
          f"{bool(np.isfinite(r_bad).all())}")
    print(f"numpy_warnings={msgs!r}")
    print(f"nothing_raised_during_assembly=True")
    if not np.isnan(r_bad).all():
        print("FAIL: the identity-less residual was not NaN-everywhere",
              file=sys.stderr)
        ok = False
    if np.linalg.norm(r_ok) >= 1e-14:
        print("FAIL: the correct residual was not zero at the reference "
              "configuration", file=sys.stderr)
        ok = False

    # --- and the Newton loop runs to completion anyway -----------------
    xbc = ib.zeros()
    xbc[right.nodal["u^1"]] = 0.05
    for label, add_I in (("with_identity", True), ("without_identity", False)):
        u = ib.zeros()
        completed = True
        try:
            for _ in range(3):
                w = ib.interpolate(u)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    R = residual_form(add_I).assemble(ib, disp=w)
                    K = tangent_form(add_I).assemble(ib, disp=w)
                    u = u + solve(*condense(K, -R, x=xbc - u, D=D))
        except Exception as e:                       # noqa: BLE001
            completed = False
            print(f"{label}_exception={type(e).__name__}")
        print(f"{label}_newton_completed={completed}")
        print(f"{label}_solution_free_dofs_all_nan="
              f"{bool(np.isnan(u[np.setdiff1d(np.arange(ib.N), D)]).all())}")
        print(f"{label}_solution_finite={bool(np.isfinite(u).all())}")
        if add_I and not np.isfinite(u).all():
            print("FAIL: the correct kernel did not stay finite",
                  file=sys.stderr)
            ok = False
        if not add_I:
            if not completed:
                print("FAIL: the identity-less kernel raised, so it is not "
                      "the silent shape the fixture documents",
                      file=sys.stderr)
                ok = False
            free = np.setdiff1d(np.arange(ib.N), D)
            if not np.isnan(u[free]).all():
                print("FAIL: the identity-less kernel produced a finite "
                      "solution on the free DOFs", file=sys.stderr)
                ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
