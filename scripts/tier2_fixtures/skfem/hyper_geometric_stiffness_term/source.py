"""Tier-2: dropping the geometric-stiffness term S : grad(du)^T grad(v).

Claim: skfem hyperelasticity#5 -- the geometric term is ESSENTIAL for Newton
convergence under large deformations; dropping it (assembling only the
material stiffness C4) "gives quadratic Newton convergence at small strains
(< 5%) but STAGNATES at large strains in the asm residual (drops by ~10x then
flat-lines around 1e-6 absolute)".

Measured on skfem 12.0.1 with a total-Lagrangian PK2 formulation
(MeshTri().refined(3), ElementVector(ElementTriP1()), mu = lam = 1, left edge
clamped, right edge pulled, residual int S : dE(v) dx):

  * "QUADRATIC AT SMALL STRAINS" IS FALSE.  At 2 % stretch -- well inside the
    entry's "< 5%" window -- the material-only tangent does NOT converge
    quadratically.  It converges LINEARLY, at a constant residual ratio, and
    needs several times as many iterations as the exact tangent.  It gets to
    the right answer, so the bug is invisible unless you look at the rate.
  * THE LARGE-STRAIN CLAIM IS TRUE IN SHAPE.  At 30 % stretch the residual
    drops about an order of magnitude, flat-lines, and then -- if you keep
    iterating -- goes NaN.  So it is worse than stagnation: the iteration
    eventually destroys itself.
  * THE PLATEAU LEVEL IN THE ENTRY IS WRONG.  The residual does not settle
    "around 1e-6 absolute"; the plateau sits orders of magnitude higher and
    is set by the strain level, not by a universal number.  Watch the shape
    of the history (drop, then flat), not an absolute threshold.
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


def dE(F, G):
    A = np.einsum("ki...,kj...->ij...", F, G)
    return 0.5 * (A + transpose(A))


@LinearForm
def residual(v, w):
    F, _, _, S = kinematics(w)
    return ddot(S, dE(F, grad(v)))


def tangent(with_geometric):
    @BilinearForm
    def form(u, v, w):
        F, Cinv, lnJ, S = kinematics(w)
        dEu, dEv = dE(F, grad(u)), dE(F, grad(v))
        sandwich = np.einsum("ik...,kl...,lj...->ij...", Cinv, dEu, Cinv)
        sandwich = 0.5 * (sandwich + transpose(sandwich))
        material = LAM * ddot(Cinv, dEu) * Cinv \
            + 2.0 * (MU - LAM * lnJ) * sandwich
        out = ddot(material, dEv)
        if with_geometric:
            out = out + ddot(S, np.einsum("ki...,kj...->ij...",
                                          grad(u), grad(v)))
        return out
    return form


def setup(refine=3):
    m = MeshTri().refined(refine)
    ib = Basis(m, ElementVector(ElementTriP1()))
    left = ib.get_dofs(lambda x: x[0] < 1e-10)
    right = ib.get_dofs(lambda x: x[0] > 1.0 - 1e-10)
    return m, ib, np.concatenate([left.all(), right.all()]), right


def newton(with_geometric, stretch, nit, tol=1e-13):
    m, ib, D, right = setup()
    xbc = ib.zeros()
    xbc[right.nodal["u^1"]] = stretch
    free = np.setdiff1d(np.arange(ib.N), D)
    K_form = tangent(with_geometric)
    u = ib.zeros()
    hist = []
    for _ in range(nit):
        w = ib.interpolate(u)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            R = residual.assemble(ib, disp=w)
            K = K_form.assemble(ib, disp=w)
            u = u + solve(*condense(K, -R, x=xbc - u, D=D))
        hist.append(float(np.linalg.norm(
            residual.assemble(ib, disp=ib.interpolate(u))[free])))
        if hist[-1] < tol:
            break
    return u, hist


def main() -> int:
    ok = True
    m, ib, _, _ = setup()
    print(f"dofs_N={ib.N} elements={m.t.shape[1]}")

    # --- small strain: the entry says quadratic; it is not --------------
    u_ok, h_ok = newton(True, 0.02, 20)
    u_bad, h_bad = newton(False, 0.02, 40)
    print(f"small_exact_iters={len(h_ok)}")
    print(f"small_material_only_iters={len(h_bad)}")
    print(f"small_exact_history={[f'{x:.2e}' for x in h_ok[:4]]}")
    print(f"small_material_only_history={[f'{x:.2e}' for x in h_bad[:6]]}")
    r_ok = [h_ok[i + 1] / h_ok[i] for i in range(len(h_ok) - 1) if h_ok[i] > 0]
    r_bad = [h_bad[i + 1] / h_bad[i] for i in range(3, 8)
             if i + 1 < len(h_bad) and h_bad[i] > 0]
    print(f"small_exact_ratios={[f'{x:.2e}' for x in r_ok[:3]]}")
    print(f"small_material_only_ratios={[f'{x:.3f}' for x in r_bad]}")
    print(f"small_exact_is_quadratic="
          f"{bool(len(r_ok) >= 2 and r_ok[1] < r_ok[0] ** 1.5)}")
    spread = (max(r_bad) - min(r_bad)) if r_bad else 1.0
    print(f"small_material_only_is_linear_not_quadratic={spread < 0.05}")
    print(f"small_material_only_costs_more_iterations="
          f"{len(h_bad) > len(h_ok)}")
    print(f"small_material_only_converged={bool(h_bad[-1] < 1e-13)}")
    rel = float(np.linalg.norm(u_bad - u_ok) / np.linalg.norm(u_ok))
    print(f"small_same_answer={rel < 1e-9}")
    if spread >= 0.05 or len(h_bad) <= len(h_ok):
        print("FAIL: at 2% stretch the material-only tangent was NOT a "
              "degraded linear rate", file=sys.stderr)
        ok = False

    # --- large strain: drop, plateau, then NaN --------------------------
    u_ok2, h_ok2 = newton(True, 0.30, 20)
    u_bad2, h_bad2 = newton(False, 0.30, 30)
    print(f"large_exact_iters={len(h_ok2)}")
    print(f"large_exact_converged={bool(h_ok2[-1] < 1e-13)}")
    print(f"large_material_only_history={[f'{x:.2e}' for x in h_bad2[:8]]}")
    finite = [x for x in h_bad2 if np.isfinite(x)]
    plateau = finite[3:8]
    drop = finite[0] / max(plateau) if plateau else 1.0
    print(f"large_material_only_initial_drop_factor={drop:.2f}")
    print(f"large_material_only_drops_about_10x={3.0 < drop < 30.0}")
    flat = max(plateau) / min(plateau) if plateau else 1e9
    print(f"large_material_only_plateau_spread_factor={flat:.3f}")
    print(f"large_material_only_flat_lines={flat < 2.0}")
    print(f"large_material_only_plateau_level={min(plateau):.2e}")
    print(f"large_material_only_plateau_near_1e_minus_6="
          f"{1e-7 < min(plateau) < 1e-5}")
    print(f"large_material_only_reached_nan="
          f"{bool(np.isnan(np.array(h_bad2)).any())}")
    print(f"large_material_only_converged={bool(finite[-1] < 1e-13)}")
    if not (3.0 < drop < 30.0):
        print("FAIL: the material-only residual did not drop about 10x",
              file=sys.stderr)
        ok = False
    if flat >= 2.0:
        print("FAIL: the material-only residual did not flat-line",
              file=sys.stderr)
        ok = False
    if 1e-7 < min(plateau) < 1e-5:
        print("FAIL: the plateau really did sit near 1e-6, so the entry's "
              "absolute threshold was usable after all", file=sys.stderr)
        ok = False
    if not np.isnan(np.array(h_bad2)).any():
        print("FAIL: the large-strain material-only run stayed finite",
              file=sys.stderr)
        ok = False
    if h_ok2[-1] >= 1e-13:
        print("FAIL: the exact tangent did not converge at large strain",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
