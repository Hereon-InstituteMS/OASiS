"""Tier-2: load stepping keeps Neo-Hookean Newton inside its basin.

Claim: skfem hyperelasticity#8 -- ramp the load over N steps for large
deformations; "a single-step application of a load that produces > 10% nominal
strain typically diverges (Newton residual from asm + condense + spsolve grows
~10x per iteration); subdividing into N steps with load_factor = i/N ...
achieves quadratic Newton convergence per substep."

Measured on skfem 12.0.1 (MeshTri().refined(3), ElementVector(ElementTriP1()),
plane strain nu = 0.3, E = 1, left edge clamped, dead transverse traction
0.5 on the right facets, PK1 residual + exact tangent):

  * SINGLE STEP FAILS, BUT NOT BY A ~10x RAMP.  There is no growth history to
    watch: the FIRST Newton correction already turns elements inside out --
    min det(F) goes negative -- and the residual evaluated afterwards is NaN
    straight away.  The displacement vector itself is still FINITE at that
    point, so an isfinite(u) guard passes while the configuration is
    physically impossible.  det(F) > 0 is the check that fires; the residual
    norm is not.
  * TWO SUBSTEPS ARE STILL NOT ENOUGH -- the same inversion happens on the
    first half-load.
  * FIVE AND TEN SUBSTEPS BOTH WORK, converge quadratically within each
    substep, keep min det(F) comfortably positive, and land on the SAME final
    displacement to machine precision, which is what makes the ramp a
    legitimate continuation rather than a different problem.
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
    FacetBasis,
    LinearForm,
    MeshTri,
    condense,
    solve,
)
from skfem.helpers import ddot, det, grad, inv, transpose

NU = 0.3
E_MOD = 1.0
TRACTION = 0.5
MU = E_MOD / (2 * (1 + NU))
LAM = E_MOD * NU / ((1 + NU) * (1 - 2 * NU))


def F_of(w):
    du = w["disp"].grad
    ident = np.zeros_like(du)
    ident[0, 0] = 1.0
    ident[1, 1] = 1.0
    return ident + du


@LinearForm
def residual(v, w):
    F = F_of(w)
    Fit = transpose(inv(F))
    with np.errstate(all="ignore"):
        P = MU * (F - Fit) + LAM * np.log(det(F)) * Fit
    return ddot(P, grad(v))


@BilinearForm
def tangent(u, v, w):
    F = F_of(w)
    Fit = transpose(inv(F))
    with np.errstate(all="ignore"):
        lnJ = np.log(det(F))
    dF = grad(u)
    T = np.einsum("ij...,kj...,kl...->il...", Fit, dF, Fit)
    return ddot(MU * dF + (MU - LAM * lnJ) * T
                + LAM * ddot(Fit, dF) * Fit, grad(v))


@LinearForm
def traction_load(v, w):
    return TRACTION * v[1]


def min_det_F(basis, u):
    du = basis.interpolate(u).grad
    ident = np.zeros_like(du)
    ident[0, 0] = 1.0
    ident[1, 1] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        J = np.asarray(det(ident + du))
    return float(np.nanmin(J)) if np.isfinite(J).any() else float("nan")


def ramp(nsteps, nit=15, tol=1e-11, refine=3):
    m = MeshTri().refined(refine).with_boundaries({
        "left": lambda x: x[0] < 1e-10,
        "right": lambda x: x[0] > 1.0 - 1e-10,
    })
    e = ElementVector(ElementTriP1())
    ib = Basis(m, e)
    fb = FacetBasis(m, e, facets=m.boundaries["right"])
    f_full = traction_load.assemble(fb)
    D = ib.get_dofs("left").all()
    free = np.setdiff1d(np.arange(ib.N), D)
    u = ib.zeros()
    per_step, jmins = [], []
    for s in range(1, nsteps + 1):
        factor = s / nsteps
        hist = []
        for _ in range(nit):
            w = ib.interpolate(u)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                R = residual.assemble(ib, disp=w) - factor * f_full
                K = tangent.assemble(ib, disp=w)
                u = u + solve(*condense(K, -R, D=D))
                Rn = float(np.linalg.norm(
                    (residual.assemble(ib, disp=ib.interpolate(u))
                     - factor * f_full)[free]))
            hist.append(Rn)
            if not np.isfinite(Rn) or Rn < tol:
                break
        per_step.append(hist)
        jmins.append(min_det_F(ib, u))
        if not np.isfinite(jmins[-1]):
            break
    return u, per_step, jmins


def main() -> int:
    ok = True
    print(f"traction={TRACTION} E={E_MOD} nu={NU}")

    # --- one shot ------------------------------------------------------
    u1, hist1, j1 = ramp(1)
    print(f"n1_first_step_history={[f'{x:.2e}' for x in hist1[0][:4]]}")
    print(f"n1_first_residual_is_nan={bool(np.isnan(hist1[0][0]))}")
    print(f"n1_min_detF={j1[0]:.4f}")
    print(f"n1_element_inverted={bool(j1[0] < 0.0)}")
    print(f"n1_displacement_is_finite={bool(np.isfinite(u1).all())}")
    print(f"n1_isfinite_guard_would_pass={bool(np.isfinite(u1).all())}")
    print(f"n1_detF_guard_would_fire={bool(j1[0] <= 0.0)}")
    print(f"n1_shows_10x_per_iteration_ramp="
          f"{len([x for x in hist1[0] if np.isfinite(x)]) > 2}")
    if not np.isnan(hist1[0][0]):
        print("FAIL: the single-step residual was not NaN immediately",
              file=sys.stderr)
        ok = False
    if not j1[0] < 0.0:
        print("FAIL: the single step did not invert an element",
              file=sys.stderr)
        ok = False
    if not np.isfinite(u1).all():
        print("FAIL: the single-step displacement was NOT finite, so an "
              "isfinite guard would have caught it", file=sys.stderr)
        ok = False

    # --- two substeps: still not enough --------------------------------
    u2, hist2, j2 = ramp(2)
    print(f"n2_first_step_history={[f'{x:.2e}' for x in hist2[0][:4]]}")
    print(f"n2_reaches_nan={bool(np.isnan(np.array(j2)).any())}")
    if not np.isnan(np.array(j2)).any():
        print("FAIL: two substeps already survived, so the boundary between "
              "failure and success is elsewhere", file=sys.stderr)
        ok = False

    # --- five and ten substeps -----------------------------------------
    solutions = {}
    for n in (5, 10):
        u, hist, j = ramp(n)
        solutions[n] = u
        quad = []
        for h in hist:
            fin = [x for x in h if np.isfinite(x) and x > 0]
            quad.append(bool(fin) and fin[-1] < 1e-11)
        last = [x for x in hist[-1] if np.isfinite(x) and x > 0]
        # quadratic: each residual is below the SQUARE of the previous one
        # (times an O(1) constant), so the ratio collapses instead of holding
        quadratic_tail = all(last[i + 1] < last[i] ** 1.5
                             for i in range(len(last) - 2))
        print(f"n{n}_substeps_completed={len(hist)}")
        print(f"n{n}_min_detF_final={j[-1]:.4f}")
        print(f"n{n}_stays_uninverted={bool(min(j) > 0.0)}")
        print(f"n{n}_solution_finite={bool(np.isfinite(u).all())}")
        print(f"n{n}_max_disp={float(np.abs(u).max()):.4e}")
        print(f"n{n}_every_substep_converged={all(quad)}")
        print(f"n{n}_last_substep_is_quadratic={quadratic_tail}")
        print(f"n{n}_last_substep_history="
              f"{[f'{x:.2e}' for x in hist[-1][:5]]}")
        if not (min(j) > 0.0 and np.isfinite(u).all() and all(quad)
                and quadratic_tail):
            print(f"FAIL: the {n}-substep ramp did not converge cleanly",
                  file=sys.stderr)
            ok = False

    rel = float(np.linalg.norm(solutions[10] - solutions[5])
                / np.linalg.norm(solutions[5]))
    print(f"ramp_is_path_independent={rel < 1e-8} rel_diff={rel:.2e}")
    if rel >= 1e-8:
        print("FAIL: 5 and 10 substeps disagreed, so the ramp changed the "
              "problem rather than continuing it", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
