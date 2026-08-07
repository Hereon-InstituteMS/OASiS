"""Tier-2: gamma below the domain threshold gives no pattern — but the quoted inequality is wrong.

Claim: skfem reaction_diffusion#7 — pattern formation requires gamma large
enough relative to domain size. "gamma * L^2 < the critical Turing wavelength
k_c^2 -> no pattern (homogeneous state stable on the domain); pattern emerges
only when gamma * L^2 > pi^2 * (a+b)^2 for Schnakenberg."

Wrong variant: gamma = 12 on the unit square, which the quoted inequality says
should pattern (12 > pi^2 * (a+b)^2 = 9.8696).
Right variant: gamma = 20, above the real threshold.

Observed on scikit-fem 12.0.1 (2026-08-06). 16x16 ElementQuad1 (289 DOFs),
a=0.1 b=0.9 d_u=1 d_v=40, L=1, backward Euler dt=0.1, 80 steps, identical seed
and 1e-2 perturbation:

  gamma =  8   spread 3.6e-14   no pattern
  gamma = 10   spread 4.2e-11   no pattern   <- above the quoted threshold
  gamma = 12   spread 6.7e-07   no pattern   <- above the quoted threshold
  gamma = 20   spread 3.3e+00   PATTERN

The qualitative claim holds: too small a gamma leaves the homogeneous state
stable on the domain. The quoted inequality does NOT. The correct criterion is
that the smallest non-zero wavenumber the domain admits, k^2 = pi^2/L^2 for
Neumann modes, must fall inside the unstable band [k_-^2, k_+^2], which for
these parameters is gamma * [0.033726, 0.741274]. That gives
gamma_crit = pi^2 / 0.741274 = 13.3144 on the unit square, not 9.8696; the runs
at gamma = 10 and 12 sit in the gap between the two numbers and produce no
pattern, which is what falsifies the quoted form.

This fixture is deliberately kept apart from reaction_diffusion#0: dt=0.1 is
small enough that backward Euler cannot mask a growing mode here. Backward Euler
amplifies whenever 0 < lambda*dt < 2, and the largest growth rate at gamma = 20
is 4.41, so lambda*dt = 0.44. The fixture asserts that for every gamma it runs.

Cost: four 80-step runs on a 16x16 grid, about 6 s. 80 steps (T=8) is what the
gamma = 12 run needs to fall six orders below its seed.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology
site -- GAMMA_WRONG, the sub-threshold gamma = 12, is raised to GAMMA_RIGHT =
20, above the real domain threshold 13.31. The sub-threshold run then
patterns and 'gamma_12_spread_lt_1e_5=True',
'gamma_12_spread_monotonically_decaying=True',
'empirical_threshold_between_12_and_20=True' and
'quoted_inequality_falsified=True' disappear from the output.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

from skfem import (Basis, BilinearForm, ElementQuad1, LinearForm, MeshQuad,
                   asm)
from skfem.models.poisson import laplace, mass

MUTATE = os.environ.get("T2_MUTATE") == "1"

A_P, B_P = 0.1, 0.9
D_U, D_V = 1.0, 40.0
L_DOMAIN = 1.0
NX = 16
DT = 0.1
N_STEPS = 80
SEED = 42
PERT = 1e-2

GAMMA_RIGHT = 20.0      # above the real threshold 13.31
# THE PATHOLOGY: gamma = 12 sits below the real domain threshold (though above
# the claim's formula).  Under T2_MUTATE the documented fix is applied here --
# gamma is raised past the real threshold.
GAMMA_WRONG = 12.0 if not MUTATE else GAMMA_RIGHT


def forms(gamma: float):
    @LinearForm
    def res_u(v, w):
        return (A_P - w["uu"] + w["uu"] ** 2 * w["vv"]) * v

    @LinearForm
    def res_v(v, w):
        return (B_P - w["uu"] ** 2 * w["vv"]) * v

    return res_u, res_v


@BilinearForm
def jac_uu(u, v, w):
    return (-1.0 + 2.0 * w["uu"] * w["vv"]) * u * v


@BilinearForm
def jac_uv(u, v, w):
    return (w["uu"] ** 2) * u * v


@BilinearForm
def jac_vu(u, v, w):
    return (-2.0 * w["uu"] * w["vv"]) * u * v


@BilinearForm
def jac_vv(u, v, w):
    return (-w["uu"] ** 2) * u * v


def reaction_jacobian_at_ss():
    u_ss, v_ss = A_P + B_P, B_P / (A_P + B_P) ** 2
    return (-1.0 + 2.0 * u_ss * v_ss, u_ss ** 2,
            -2.0 * u_ss * v_ss, -u_ss ** 2)


def band_coefficients():
    """k_-^2 and k_+^2 divided by gamma — the unstable band, per unit gamma."""
    f_u, f_v, g_u, g_v = reaction_jacobian_at_ss()
    det = f_u * g_v - f_v * g_u
    d = D_V / D_U
    root = np.sqrt((d * f_u + g_v) ** 2 - 4.0 * d * det)
    return ((d * f_u + g_v - root) / (2.0 * d),
            (d * f_u + g_v + root) / (2.0 * d))


def growth_rate(gamma: float, nmax: int = 12):
    f_u, f_v, g_u, g_v = reaction_jacobian_at_ss()
    det = f_u * g_v - f_v * g_u
    d = D_V / D_U
    best = -np.inf
    for n in range(nmax):
        for mm in range(nmax):
            k2 = np.pi ** 2 * (n * n + mm * mm) / L_DOMAIN ** 2
            trace = gamma * (f_u + g_v) - (1.0 + d) * k2
            h = d * k2 ** 2 - gamma * (d * f_u + g_v) * k2 + gamma ** 2 * det
            disc = trace * trace - 4.0 * h
            lam = 0.5 * (trace + np.sqrt(disc)) if disc >= 0.0 else 0.5 * trace
            best = max(best, lam)
    return best


def schnakenberg(gamma: float):
    res_u, res_v = forms(gamma)
    mesh = MeshQuad.init_tensor(np.linspace(0.0, L_DOMAIN, NX + 1),
                                np.linspace(0.0, L_DOMAIN, NX + 1))
    basis = Basis(mesh, ElementQuad1())
    K = laplace.assemble(basis)
    M = mass.assemble(basis)
    N = basis.N
    rng = np.random.default_rng(SEED)
    u = np.full(N, A_P + B_P) + PERT * rng.standard_normal(N)
    v = np.full(N, B_P / (A_P + B_P) ** 2) + PERT * rng.standard_normal(N)
    Ku, Kv = D_U * K + M / DT, D_V * K + M / DT
    trace = [float(u.max() - u.min())]
    for _step in range(N_STEPS):
        u_old, v_old = u.copy(), v.copy()
        u_new, v_new = u_old.copy(), v_old.copy()
        for _it in range(25):
            U, V = basis.interpolate(u_new), basis.interpolate(v_new)
            R = np.concatenate([
                M @ (u_new - u_old) / DT + D_U * K @ u_new
                - gamma * asm(res_u, basis, uu=U, vv=V),
                M @ (v_new - v_old) / DT + D_V * K @ v_new
                - gamma * asm(res_v, basis, uu=U, vv=V)])
            J = bmat([[Ku - gamma * asm(jac_uu, basis, uu=U, vv=V),
                       -gamma * asm(jac_uv, basis, uu=U, vv=V)],
                      [-gamma * asm(jac_vu, basis, uu=U, vv=V),
                       Kv - gamma * asm(jac_vv, basis, uu=U, vv=V)]],
                     format="csr")
            dx = spsolve(J, -R)
            u_new += dx[:N]
            v_new += dx[N:]
            if np.linalg.norm(dx) < 1e-10:
                break
        u, v = u_new, v_new
        trace.append(float(u.max() - u.min()))
    return basis, trace


def main() -> int:
    ok = True

    k_minus, k_plus = band_coefficients()
    smallest_mode = np.pi ** 2 / L_DOMAIN ** 2
    gamma_crit = smallest_mode / k_plus
    claim_value = np.pi ** 2 * (A_P + B_P) ** 2
    print(f"element={type(ElementQuad1()).__name__} domain_L={L_DOMAIN:.1f}")
    print(f"unstable_band_per_gamma=[{k_minus:.6f}, {k_plus:.6f}]")
    print(f"smallest_nonzero_neumann_mode_k2={smallest_mode:.4f}")
    print(f"analytic_gamma_crit={gamma_crit:.4f}")
    print(f"claim_formula_pi2_times_a_plus_b_squared={claim_value:.4f}")
    print(f"analytic_gamma_crit_between_13_and_14={13.0 < gamma_crit < 14.0}")
    print(f"claim_formula_below_analytic_threshold={claim_value < gamma_crit}")
    if not (13.0 < gamma_crit < 14.0 and claim_value < gamma_crit):
        print("FAIL: the dispersion-relation threshold moved; the recorded "
              "comparison against the quoted formula is stale", file=sys.stderr)
        ok = False

    results = {}
    for gamma in (8.0, 10.0, GAMMA_WRONG, GAMMA_RIGHT):
        lam = growth_rate(gamma)
        basis, trace = schnakenberg(gamma)
        results[gamma] = (lam, trace)
        print(f"gamma={gamma:g} lambda_max={lam:+.4g} lambda_max_times_dt="
              f"{lam * DT:+.4f} final_spread={trace[-1]:.4e}")

    print(f"n_dofs={basis.N} n_elements={basis.mesh.nelements} steps={N_STEPS}")
    unmaskable = all(lam * DT < 2.0 for lam, _ in results.values())
    print(f"dt_cannot_mask_growth={unmaskable}")
    if not unmaskable:
        print("FAIL: dt is large enough for backward Euler to damp a growing "
              "mode, so this fixture cannot separate gamma from dt",
              file=sys.stderr)
        ok = False

    s8 = results[8.0][1][-1]
    s10 = results[10.0][1][-1]
    s_wrong = results[GAMMA_WRONG][1][-1]
    s_right = results[GAMMA_RIGHT][1][-1]
    decaying = all(results[GAMMA_WRONG][1][i + 1] < results[GAMMA_WRONG][1][i]
                   for i in range(1, N_STEPS))
    print(f"gamma_8_spread_lt_1e_10={s8 < 1e-10}")
    print(f"gamma_10_above_claim_formula={10.0 > claim_value}")
    print(f"gamma_10_spread_lt_1e_6={s10 < 1e-6}")
    print(f"gamma_12_above_claim_formula={GAMMA_WRONG > claim_value}")
    print(f"gamma_12_spread_lt_1e_5={s_wrong < 1e-5}")
    print(f"gamma_12_spread_monotonically_decaying={decaying}")
    print(f"gamma_20_spread_gt_1p0={s_right > 1.0}")
    print(f"empirical_threshold_between_12_and_20="
          f"{s_wrong < 1e-5 < 1.0 < s_right}")
    print(f"quoted_inequality_falsified="
          f"{GAMMA_WRONG > claim_value and s_wrong < 1e-5}")
    if not (s8 < 1e-10 and s10 < 1e-6 and s_wrong < 1e-5 and decaying):
        print("FAIL: a sub-threshold gamma produced a pattern — the claim's "
              "pathology is absent", file=sys.stderr)
        ok = False
    if s_right <= 1.0:
        print("FAIL: the supra-threshold gamma did not pattern, so there is no "
              "healthy reference", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
