"""Tier-2: getting the current Newton iterate into a skfem form kernel.

Claim: skfem hyperelasticity#6 -- use ib.interpolate(u) to obtain the
displacement and gradient at quadrature points inside the bilinear form.
Trying to access u directly inside the @BilinearForm body raises
"NameError: name 'u' is not defined" (or, with an outer closure, computes
against the LAST u rather than the current Newton iterate).  The correct
pattern is u_qp = w['u'] / w['displacement'] passed via ib.interpolate(u).

Measured on skfem 12.0.1.  Three distinct behaviours, of which only the first
is loud:

  * A bare name in the kernel body raises NameError at ASSEMBLY time, naming
    the identifier that is missing.  Note that inside a @BilinearForm the
    names `u` and `v` ARE bound (they are the trial and test arguments), so
    the NameError only appears for some OTHER identifier -- writing `u` and
    meaning "the current displacement" gives you the trial function instead,
    which is a silent type confusion, not a NameError.
  * Reading a key from w that was never passed raises KeyError with the key
    name -- also loud.
  * THE STALE-CLOSURE CASE IS SILENT.  A form built once by a factory that
    captured the iterate by value keeps assembling against that captured
    array forever.  The Newton loop then stops being Newton: it repeats the
    same linearisation, the residual stalls at a fixed non-zero level instead
    of falling quadratically, and nothing is printed.  Passing the iterate
    through assemble(ib, disp=ib.interpolate(u)) fixes it.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology
site -- the frozen branch of newton() stops assembling the factory-frozen
forms and assembles residual_kwarg/tangent_kwarg with
disp=ib.interpolate(u) instead, so the linearisation follows the current
iterate.  The unmutated run matches all 13 expect_in_output strings; under
mutation "frozen_converges=False", "frozen_residual_stalls=True" and
"frozen_answers_differ=True" disappear.
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

MUTATE = os.environ.get("T2_MUTATE") == "1"

MU = 1.0
LAM = 1.0


def F_of_grad(du):
    ident = np.zeros_like(du)
    ident[0, 0] = 1.0
    ident[1, 1] = 1.0
    return ident + du


def pk1(F):
    Fit = transpose(inv(F))
    with np.errstate(all="ignore"):
        return MU * (F - Fit) + LAM * np.log(det(F)) * Fit


# --- the CORRECT pattern: the iterate arrives through w ----------------
@LinearForm
def residual_kwarg(v, w):
    return ddot(pk1(F_of_grad(w["disp"].grad)), grad(v))


@BilinearForm
def tangent_kwarg(u, v, w):
    F = F_of_grad(w["disp"].grad)
    Fit = transpose(inv(F))
    with np.errstate(all="ignore"):
        lnJ = np.log(det(F))
    dF = grad(u)
    T = np.einsum("ij...,kj...,kl...->il...", Fit, dF, Fit)
    return ddot(MU * dF + (MU - LAM * lnJ) * T
                + LAM * ddot(Fit, dF) * Fit, grad(v))


# --- the STALE pattern: a factory captures the iterate by value --------
def residual_frozen(basis, u_captured):
    @LinearForm
    def form(v, w):
        return ddot(pk1(F_of_grad(basis.interpolate(u_captured).grad)),
                    grad(v))
    return form


def tangent_frozen(basis, u_captured):
    @BilinearForm
    def form(u, v, w):
        F = F_of_grad(basis.interpolate(u_captured).grad)
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


def newton(frozen, nit=6, stretch=0.10):
    m, ib, D, right = setup()
    xbc = ib.zeros()
    xbc[right.nodal["u^1"]] = stretch
    free = np.setdiff1d(np.arange(ib.N), D)
    u = ib.zeros()
    R_frozen = residual_frozen(ib, u) if frozen else None
    K_frozen = tangent_frozen(ib, u) if frozen else None
    hist, msgs = [], []
    for _ in range(nit):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # MUTATION: under T2_MUTATE the frozen branch is replaced by the
            # documented fix, the kwarg pattern below.
            if frozen and not MUTATE:
                R = R_frozen.assemble(ib)
                K = K_frozen.assemble(ib)
            else:
                w = ib.interpolate(u)
                R = residual_kwarg.assemble(ib, disp=w)
                K = tangent_kwarg.assemble(ib, disp=w)
            u = u + solve(*condense(K, -R, x=xbc - u, D=D))
            msgs += [str(c.message) for c in caught]
        # the residual is always measured with the CORRECT kernel, so the
        # comparison is between two solvers, not two error measures
        R_true = residual_kwarg.assemble(ib, disp=ib.interpolate(u))
        hist.append(float(np.linalg.norm(R_true[free])))
    return u, hist, sorted(set(msgs))


def main() -> int:
    ok = True
    m, ib, D, _ = setup()
    print(f"dofs_N={ib.N} elements={m.t.shape[1]}")

    # --- a bare undefined name in the kernel body ----------------------
    @BilinearForm
    def bare_name(u, v, w):
        return current_displacement * u * v          # noqa: F821

    err = None
    try:
        bare_name.assemble(ib)
    except NameError as e:
        err = e
    print(f"bare_name_raises_NameError={isinstance(err, NameError)}")
    print(f"bare_name_message={str(err)!r}")
    if err is None:
        print("FAIL: a bare undefined name did not raise NameError",
              file=sys.stderr)
        ok = False

    # --- but `u` itself IS bound, so writing `u` is a silent confusion --
    @BilinearForm
    def u_is_the_trial_function(u, v, w):
        return ddot(grad(u), grad(v))

    shadow_err = None
    try:
        K = u_is_the_trial_function.assemble(ib)
        shape = K.shape
    except NameError as e:
        shadow_err = e
        shape = None
    print(f"name_u_inside_form_raises={shadow_err is not None}")
    print(f"name_u_is_bound_to_trial_function={shadow_err is None}")
    print(f"trial_function_form_assembles_shape={shape}")
    if shadow_err is not None:
        print("FAIL: `u` was NOT bound inside the form body", file=sys.stderr)
        ok = False

    # --- a missing assemble kwarg is loud ------------------------------
    kerr = None
    try:
        residual_kwarg.assemble(ib)
    except KeyError as e:
        kerr = e
    print(f"missing_kwarg_raises_KeyError={isinstance(kerr, KeyError)}")
    print(f"missing_kwarg_names_the_key={'disp' in str(kerr)}")
    if kerr is None:
        print("FAIL: assembling without the state kwarg did not raise",
              file=sys.stderr)
        ok = False

    # --- the silent one: a frozen closure ------------------------------
    u_ok, h_ok, msg_ok = newton(frozen=False)
    u_bad, h_bad, msg_bad = newton(frozen=True)
    print(f"kwarg_residual_history={[f'{x:.2e}' for x in h_ok]}")
    print(f"frozen_residual_history={[f'{x:.2e}' for x in h_bad]}")
    print(f"kwarg_converges={bool(h_ok[-1] < 1e-12)}")
    print(f"frozen_converges={bool(h_bad[-1] < 1e-12)}")
    print(f"frozen_residual_stalls="
          f"{bool(h_bad[-1] > 1e-6 and abs(h_bad[-1] - h_bad[-2]) < 1e-12)}")
    print(f"frozen_warnings={msg_bad!r}")
    print(f"frozen_failure_is_silent={not msg_bad}")
    print(f"frozen_solution_finite={bool(np.isfinite(u_bad).all())}")
    rel = float(np.linalg.norm(u_bad - u_ok) / np.linalg.norm(u_ok))
    print(f"frozen_vs_kwarg_relative_difference={rel:.4e}")
    print(f"frozen_answers_differ={rel > 1e-4}")
    if h_ok[-1] >= 1e-12:
        print("FAIL: the kwarg pattern did not converge", file=sys.stderr)
        ok = False
    if h_bad[-1] < 1e-12:
        print("FAIL: the frozen closure converged, so it is not stale",
              file=sys.stderr)
        ok = False
    if msg_bad:
        print(f"FAIL: the frozen closure warned {msg_bad!r}, so it is not "
              f"the silent shape documented here", file=sys.stderr)
        ok = False
    if rel <= 1e-4:
        print("FAIL: the frozen closure gave the same answer",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
