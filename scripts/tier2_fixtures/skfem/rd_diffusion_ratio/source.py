"""Tier-2: d_v = d_u kills the Turing mechanism; the critical ratio is 8.57, not 10.

Claim: skfem reaction_diffusion#4 — the Turing instability requires
d_v >> d_u (fast inhibitor, slow activator). Setting d_v = d_u (or d_u > d_v)
kills the cross-diffusion mechanism: amplitudes decay back to the homogeneous
steady state with no pattern. The condition needs d_v/d_u above ~10 for
Schnakenberg.

Wrong variant: d_v = d_u = 1, everything else identical.
Right variant: the shipped d_v = 40.

Observed on scikit-fem 12.0.1 (2026-08-06). 16x16 ElementQuad1 (289 DOFs),
a=0.1 b=0.9 gamma=1000, backward Euler dt=2e-3, 40 steps, same seed and same
1e-2 perturbation in every run:

  d_v/d_u =  1   spread 8.9e-16   (decayed to machine noise)
  d_v/d_u =  8   spread 1.1e-03   (still decaying)
  d_v/d_u = 10   spread 9.8e-01   (pattern)
  d_v/d_u = 40   spread 4.1e+00   (pattern)

so the mechanism switches on between 8 and 10, which brackets the analytic
threshold this fixture also computes from the dispersion relation: the Turing
conditions for (a, b) = (0.1, 0.9) are d*f_u + g_v > 0 and
(d*f_u + g_v)^2 > 4*d*det, whose root is d_crit = 8.5676. The claim's "~10" is
therefore a safe rule of thumb rather than the exact number; d = 10 does pattern,
d = 8 does not.

At d_v = d_u the first Turing condition already fails: d*f_u + g_v = -0.2 < 0,
and the largest growth rate over all Neumann modes of the unit square is -100,
i.e. every mode decays. At d_v = 40 it is +31 and +411 respectively.

Cost: four 40-step runs on a 16x16 grid, about 5 s. 40 steps is what the d_v = 8
and d_v = 10 runs need to separate; the d_v = 40 pattern has been saturated since
step 8.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology
site -- D_V_WRONG, the equal diffusivity d_v = d_u = 1, is set to the shipped
D_V_RIGHT = 40, so the wrong variant becomes the healthy one. The Turing
mechanism switches back on and 'equal_diffusivity_fails_turing_condition=True',
'lambda_max_at_equal_diffusivity_lt_0=True' and
'equal_diffusivity_spread_lt_1e_8=True' disappear from the output.
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
D_U = 1.0
GAMMA = 1000.0
NX = 16
DT = 2e-3
N_STEPS = 40
SEED = 42
PERT = 1e-2

D_V_RIGHT = 40.0        # the shipped value
# THE PATHOLOGY: equal diffusivities.  Under T2_MUTATE the documented fix is
# applied here -- the inhibitor gets the shipped fast diffusivity back.
D_V_WRONG = 1.0 if not MUTATE else D_V_RIGHT


@LinearForm
def res_u(v, w):
    return (A_P - w["uu"] + w["uu"] ** 2 * w["vv"]) * v


@LinearForm
def res_v(v, w):
    return (B_P - w["uu"] ** 2 * w["vv"]) * v


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


def growth_rate(d_ratio: float, nmax: int = 12):
    """Largest linear growth rate over the Neumann modes of the unit square."""
    f_u, f_v, g_u, g_v = reaction_jacobian_at_ss()
    det = f_u * g_v - f_v * g_u
    best = -np.inf
    for n in range(nmax):
        for mm in range(nmax):
            k2 = np.pi ** 2 * (n * n + mm * mm)
            trace = GAMMA * (f_u + g_v) - (1.0 + d_ratio) * k2
            h = (d_ratio * k2 ** 2 - GAMMA * (d_ratio * f_u + g_v) * k2
                 + GAMMA ** 2 * det)
            disc = trace * trace - 4.0 * h
            lam = 0.5 * (trace + np.sqrt(disc)) if disc >= 0.0 else 0.5 * trace
            best = max(best, lam)
    return best


def critical_ratio():
    """Root of (d*f_u + g_v)^2 = 4*d*det — the exact Turing threshold."""
    f_u, f_v, g_u, g_v = reaction_jacobian_at_ss()
    det = f_u * g_v - f_v * g_u
    aa = f_u ** 2
    bb = 2.0 * f_u * g_v - 4.0 * det
    cc = g_v ** 2
    return (-bb + np.sqrt(bb * bb - 4.0 * aa * cc)) / (2.0 * aa)


def schnakenberg(d_v: float):
    mesh = MeshQuad.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                               np.linspace(0.0, 1.0, NX + 1))
    basis = Basis(mesh, ElementQuad1())
    K = laplace.assemble(basis)
    M = mass.assemble(basis)
    N = basis.N
    rng = np.random.default_rng(SEED)
    u = np.full(N, A_P + B_P) + PERT * rng.standard_normal(N)
    v = np.full(N, B_P / (A_P + B_P) ** 2) + PERT * rng.standard_normal(N)
    Ku, Kv = D_U * K + M / DT, d_v * K + M / DT
    for _step in range(N_STEPS):
        u_old, v_old = u.copy(), v.copy()
        u_new, v_new = u_old.copy(), v_old.copy()
        for _it in range(25):
            U, V = basis.interpolate(u_new), basis.interpolate(v_new)
            R = np.concatenate([
                M @ (u_new - u_old) / DT + D_U * K @ u_new
                - GAMMA * asm(res_u, basis, uu=U, vv=V),
                M @ (v_new - v_old) / DT + d_v * K @ v_new
                - GAMMA * asm(res_v, basis, uu=U, vv=V)])
            J = bmat([[Ku - GAMMA * asm(jac_uu, basis, uu=U, vv=V),
                       -GAMMA * asm(jac_uv, basis, uu=U, vv=V)],
                      [-GAMMA * asm(jac_vu, basis, uu=U, vv=V),
                       Kv - GAMMA * asm(jac_vv, basis, uu=U, vv=V)]],
                     format="csr")
            dx = spsolve(J, -R)
            u_new += dx[:N]
            v_new += dx[N:]
            if np.linalg.norm(dx) < 1e-10:
                break
        u, v = u_new, v_new
    return basis, u, float(u.max() - u.min())


def main() -> int:
    ok = True
    f_u, f_v, g_u, g_v = reaction_jacobian_at_ss()
    print(f"element={type(ElementQuad1()).__name__}")
    print(f"f_u={f_u:.1f} f_v={f_v:.1f} g_u={g_u:.1f} g_v={g_v:.1f}")

    cond_wrong = (D_V_WRONG / D_U) * f_u + g_v
    cond_right = (D_V_RIGHT / D_U) * f_u + g_v
    print(f"turing_condition_at_ratio_1={cond_wrong:.1f}")
    print(f"turing_condition_at_ratio_40={cond_right:.1f}")
    print(f"equal_diffusivity_fails_turing_condition={cond_wrong <= 0.0}")
    print(f"ratio_40_passes_turing_condition={cond_right > 0.0}")

    d_crit = critical_ratio()
    print(f"analytic_critical_ratio={d_crit:.4f}")
    print(f"critical_ratio_between_8_and_9={8.0 < d_crit < 9.0}")

    lam_wrong = growth_rate(D_V_WRONG / D_U)
    lam_right = growth_rate(D_V_RIGHT / D_U)
    print(f"lambda_max_at_ratio_1={lam_wrong:.4g} "
          f"lambda_max_at_ratio_40={lam_right:.4g}")
    print(f"lambda_max_at_equal_diffusivity_lt_0={lam_wrong < 0.0}")
    print(f"lambda_max_at_ratio_40_gt_0={lam_right > 0.0}")
    if not (cond_wrong <= 0.0 < cond_right and lam_wrong < 0.0 < lam_right
            and 8.0 < d_crit < 9.0):
        print("FAIL: the linear Turing analysis no longer separates equal from "
              "unequal diffusivities", file=sys.stderr)
        ok = False

    # --- WRONG variant: d_v = d_u ------------------------------------------
    basis, _u_w, spread_w = schnakenberg(D_V_WRONG)
    print(f"n_dofs={basis.N} n_elements={basis.mesh.nelements} steps={N_STEPS}")
    print(f"equal_diffusivity_spread={spread_w:.4e}")
    print(f"equal_diffusivity_spread_lt_1e_8={spread_w < 1e-8}")
    if spread_w >= 1e-8:
        print("FAIL: equal diffusivities still produced structure — the claim's "
              "pathology is absent", file=sys.stderr)
        ok = False

    # --- RIGHT variant: the shipped ratio ----------------------------------
    _b, u_r, spread_r = schnakenberg(D_V_RIGHT)
    print(f"ratio_40 u=[{u_r.min():.5f}, {u_r.max():.5f}]")
    print(f"ratio_40_spread={spread_r:.4e}")
    print(f"ratio_40_spread_gt_1p0={spread_r > 1.0}")
    if spread_r <= 1.0:
        print("FAIL: the d_v/d_u = 40 run did not pattern, so the fixture has "
              "no healthy reference", file=sys.stderr)
        ok = False

    # --- bracket the threshold the claim quotes as "~10" -------------------
    _b8, _u8, spread_8 = schnakenberg(8.0)
    _b10, _u10, spread_10 = schnakenberg(10.0)
    print(f"ratio_8_spread={spread_8:.4e} ratio_10_spread={spread_10:.4e}")
    print(f"ratio_8_spread_lt_1e_2={spread_8 < 1e-2}")
    print(f"ratio_10_spread_gt_0p1={spread_10 > 0.1}")
    print(f"empirical_threshold_between_8_and_10="
          f"{spread_8 < 1e-2 < 0.1 < spread_10}")
    print(f"analytic_threshold_inside_empirical_bracket="
          f"{8.0 < d_crit < 10.0}")
    if not (spread_8 < 1e-2 < 0.1 < spread_10):
        print("FAIL: the ratio 8 / ratio 10 runs no longer bracket the Turing "
              "threshold", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
