"""Tier-2: the thermal RHS factor is 3K = (3*lam+2*mu), not 2*mu.

Claim: ngsolve thermal_structural#2 — the elasticity RHS is
(3*lam + 2*mu)*alpha*T*Id(dim) contracted with Strain(v); with 2*mu instead, the
free expansion is wrong by (1+nu)/(1-2*nu), which is ~3x for typical metals
(2.50 / 3.25 / 4.50 at nu = 0.25 / 0.30 / 0.35) and NOT the '~30-100x' the older
catalog text claimed.

Wrong variant: assemble the same problem with the 2*mu factor.

Setup (as recorded in the claim): unit_square maxh=0.3, VectorH1(order=2), the
three rigid-body modes removed with a 3-component NumberSpace so the specimen is
genuinely unconstrained, E=210 GPa, nu=0.3, alpha=1.2e-5, dT=100.

Exact plane-strain free expansion for a uniform temperature rise:
    sigma = 0  =>  2 mu e + 2 lam e = (3 lam + 2 mu) alpha dT
    e = (3 lam + 2 mu) / (2 (lam + mu)) * alpha dT = (1 + nu) alpha dT
and with the 2*mu factor the same algebra gives e = (1 - 2 nu) alpha dT.

Observed on NGSolve 6.2.2604 (2026-08-03):
    (3 lam + 2 mu) RHS -> eps_xx = eps_yy = 1.560000e-03 = (1+nu) alpha dT
    2 mu           RHS -> eps_xx = eps_yy = 4.800000e-04 = (1-2nu) alpha dT
    ratio 0.3077 = (1-2nu)/(1+nu), so the error factor is 3.25x, not 30-100x.
Both variants leave the body stress-free, which is why the mistake is silent:
a uniform eigenstrain on a free body always relaxes to zero total stress, so
only the STRAIN magnitude reveals it.

Mutation control: T2_MUTATE=1 sets WRONG_RHS_FACTOR to '3*lam+2*mu', putting
the correct 3K prefactor in the slot where 2*mu was, so
'wrong_factor_matches_1_minus_2nu_law=True',
'wrong_factor_misses_exact_expansion=True' and
'error_factor_matches_closed_form=True' disappear.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import ngsolve as ngs
from netgen.geom2d import unit_square

# WRONG: the thermal-RHS prefactor used by the failing variant
MUTATE = os.environ.get("T2_MUTATE") == "1"

# Mutation: T2_MUTATE=1 uses the correct 3K = 3*lam + 2*mu prefactor.
WRONG_RHS_FACTOR = "3*lam+2*mu" if MUTATE else "2*mu"

E, NU, ALPHA, DT = 210e3, 0.3, 1.2e-5, 100.0
MU = E / (2.0 * (1.0 + NU))
LAM = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
EXACT_3K = (1.0 + NU) * ALPHA * DT          # 1.560000e-03
EXACT_2MU = (1.0 - 2.0 * NU) * ALPHA * DT   # 4.800000e-04


def free_expansion(factor_name: str) -> tuple[float, float, float]:
    """Unconstrained uniformly heated square; returns eps_xx, eps_yy, |sigma|."""
    factor = (3.0 * LAM + 2.0 * MU) if factor_name == "3*lam+2*mu" else 2.0 * MU
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.3))
    Vu = ngs.VectorH1(mesh, order=2)
    N = ngs.NumberSpace(mesh)
    X = Vu * N * N * N                       # 2 translations + 1 rotation
    (uu, l1, l2, l3), (vv, m1, m2, m3) = X.TnT()
    x, y = ngs.x, ngs.y

    def strain(w):
        return 0.5 * (ngs.Grad(w) + ngs.Grad(w).trans)

    def stress(w):
        return 2 * MU * strain(w) + LAM * ngs.Trace(strain(w)) * ngs.Id(2)

    a = ngs.BilinearForm(X, symmetric=True)
    a += ngs.InnerProduct(stress(uu), strain(vv)) * ngs.dx
    a += (l1 * vv[0] + l2 * vv[1] + l3 * (y * vv[0] - x * vv[1])) * ngs.dx
    a += (m1 * uu[0] + m2 * uu[1] + m3 * (y * uu[0] - x * uu[1])) * ngs.dx
    a.Assemble()
    eps_th = factor * ALPHA * DT * ngs.Id(2)
    f = ngs.LinearForm(X)
    f += ngs.InnerProduct(eps_th, strain(vv)) * ngs.dx
    f.Assemble()
    gf = ngs.GridFunction(X)
    gf.vec.data = a.mat.Inverse(X.FreeDofs(), inverse="umfpack") * f.vec
    gu = gf.components[0]
    exx = float(ngs.Integrate(ngs.Grad(gu)[0, 0], mesh))
    eyy = float(ngs.Integrate(ngs.Grad(gu)[1, 1], mesh))
    resid = stress(gu) - eps_th
    smax = float(abs(ngs.Integrate(ngs.InnerProduct(resid, resid), mesh))) ** 0.5
    return exx, eyy, smax


def main() -> int:
    ok = True
    print(f"ngsolve_version={ngs.__version__}")
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=0.3))
    print(f"structural_space_type={ngs.VectorH1(mesh, order=2).type} "
          f"constraint_space_type={ngs.NumberSpace(mesh).type}")
    print(f"mu={MU:.6g} lam={LAM:.6g} three_k={3 * LAM + 2 * MU:.6g} "
          f"E_over_1_minus_2nu={E / (1 - 2 * NU):.6g}")
    print(f"exact_plane_strain_expansion={EXACT_3K:.6e}")

    # --- WRONG variant: 2*mu prefactor ------------------------------------
    exx_b, eyy_b, sig_b = free_expansion(WRONG_RHS_FACTOR)
    print(f"wrong_factor={WRONG_RHS_FACTOR} eps_xx={exx_b:.6e} "
          f"eps_yy={eyy_b:.6e} residual_stress={sig_b:.3e}")
    print(f"wrong_factor_matches_1_minus_2nu_law="
          f"{abs(exx_b - EXACT_2MU) / EXACT_2MU < 1e-6}")
    print(f"wrong_factor_misses_exact_expansion="
          f"{abs(exx_b - EXACT_3K) / EXACT_3K > 0.1}")
    print(f"wrong_factor_still_stress_free={sig_b < 1e-6}")
    if abs(exx_b - EXACT_2MU) / EXACT_2MU >= 1e-6:
        print(f"FAIL: the 2*mu RHS gave eps_xx={exx_b:.6e}, not the predicted "
              f"(1-2nu) alpha dT = {EXACT_2MU:.6e}", file=sys.stderr)
        ok = False
    if abs(exx_b - EXACT_3K) / EXACT_3K <= 0.1:
        print("FAIL: the 2*mu RHS reproduced the correct free expansion",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: 3K = 3*lam + 2*mu ---------------------------------
    exx_g, eyy_g, sig_g = free_expansion("3*lam+2*mu")
    rel = abs(exx_g - EXACT_3K) / EXACT_3K
    print(f"right_factor=3*lam+2*mu eps_xx={exx_g:.6e} eps_yy={eyy_g:.6e} "
          f"residual_stress={sig_g:.3e} rel_dev={rel:.3e}")
    print(f"three_k_reproduces_exact_expansion_to_6_digits={rel < 1e-6}")
    print(f"three_k_expansion_is_isotropic="
          f"{abs(exx_g - eyy_g) / EXACT_3K < 1e-6}")
    print(f"three_k_leaves_body_stress_free={sig_g < 1e-6}")
    if rel >= 1e-6:
        print(f"FAIL: the (3 lam + 2 mu) RHS missed the exact plane-strain free "
              f"expansion by {rel:.3e}", file=sys.stderr)
        ok = False

    # --- the size of the mistake ------------------------------------------
    err_factor = exx_g / exx_b
    print(f"error_factor={err_factor:.4f} "
          f"predicted_1_plus_nu_over_1_minus_2nu="
          f"{(1 + NU) / (1 - 2 * NU):.4f}")
    print(f"error_factor_matches_closed_form="
          f"{abs(err_factor - (1 + NU) / (1 - 2 * NU)) < 1e-4}")
    print(f"error_factor_below_10x={err_factor < 10.0}")
    print(f"error_factor_in_old_30_to_100x_claim={30.0 <= err_factor <= 100.0}")
    if abs(err_factor - (1 + NU) / (1 - 2 * NU)) >= 1e-4:
        print(f"FAIL: measured error factor {err_factor:.4f} does not match "
              f"(1+nu)/(1-2nu)", file=sys.stderr)
        ok = False
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
