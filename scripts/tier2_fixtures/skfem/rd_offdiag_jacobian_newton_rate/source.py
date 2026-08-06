"""Tier-2: dropping the off-diagonal Jacobian blocks costs Newton its quadratic rate.

Claim: skfem reaction_diffusion#1 — the coupled Schnakenberg system needs the
full block Jacobian [[J_uu, J_uv], [J_vu, J_vv]], four BilinearForm + asm calls
per Newton step. Assembling only the diagonal blocks and dropping the
off-diagonal asm output gives a linear-rate Newton instead of quadratic; the
off-diagonal terms scale with df_u/dv and df_v/du, which are non-zero for any
coupled reaction.

Wrong variant: identical residual, identical diagonal blocks, but
``bmat([[J_uu, None], [None, J_vv]])`` — the two off-diagonal asm calls are
simply not made.
Right variant: all four blocks.

Observed on scikit-fem 12.0.1 (2026-08-06), one backward-Euler step at dt=1e-4
from a state perturbed by 0.3 about (u_ss, v_ss), 16x16 ElementQuad1 grid:

  full   ||R|| = 5.12e+02  3.93e-01  2.73e-05  4.19e-13   (2 updates)
  diag   ||R|| = 5.12e+02  5.99e+00  1.02e+00  4.24e-02 ... 1.90e-09 (11 updates)

The full sequence's successive-residual ratios collapse (6.9e-05 -> 1.5e-08,
3.7 decades gained), the textbook signature of quadratic convergence. The
diagonal-only ratios oscillate in [4.2e-02, 2.5e-01] forever and never drop
below 1e-2: a fixed linear rate of about 0.13 per iteration, 5.5x the iteration
count for the same tolerance.

The fixture also pins the off-diagonal blocks themselves: at the homogeneous
steady state df_u/dv = u_ss^2 = 1.0 exactly and df_v/du = -2 u_ss v_ss = -1.8
exactly, so ``asm(J_uv)`` there must equal the plain mass matrix entry for
entry — it does, to 1e-16. A library change that silently zeroed a
variable-coefficient mass form would break that check.

The residual and the Jacobian are both assembled by quadrature from
``basis.interpolate(u)``, which makes them an exactly consistent pair (verified
here by the FD check in rd_reaction_block_mass_vs_stiffness). That matters: the
shipped template mixes a nodal residual ``M @ f(u_nodal)`` with a
consistent-coefficient Jacobian ``asm(mass_pointwise, c=df_nodal)``, which are
NOT derivative-consistent, so the shipped four-block Newton is itself only
linear. See the authoring report.

Cost: one time step on a 16x16 grid, well under a second.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology
site -- the off_diagonal flag inside newton() is forced True, so the two
dropped asm(jac_uv) / asm(jac_vu) calls are made again and bmat gets all four
blocks. The wrong variant recovers the quadratic rate and
'diag_updates_ge_8=True', 'diag_min_ratio_gt_1e_2=True' and
'diag_ratio_decades_lt_1=True' disappear from the output.
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
GAMMA = 1000.0
NX = 16
DT = 1e-4
PERT = 0.3
SEED = 7
MAXIT = 25
RTOL = 1e-9


@LinearForm
def res_u(v, w):
    return (A_P - w["uu"] + w["uu"] ** 2 * w["vv"]) * v


@LinearForm
def res_v(v, w):
    return (B_P - w["uu"] ** 2 * w["vv"]) * v


@BilinearForm
def jac_uu(u, v, w):          # df_u/du = -1 + 2 u v
    return (-1.0 + 2.0 * w["uu"] * w["vv"]) * u * v


@BilinearForm
def jac_uv(u, v, w):          # df_u/dv = u^2
    return (w["uu"] ** 2) * u * v


@BilinearForm
def jac_vu(u, v, w):          # df_v/du = -2 u v
    return (-2.0 * w["uu"] * w["vv"]) * u * v


@BilinearForm
def jac_vv(u, v, w):          # df_v/dv = -u^2
    return (-w["uu"] ** 2) * u * v


def newton(off_diagonal: bool):
    """One backward-Euler step; returns the residual history."""
    # THE PATHOLOGY: the two off-diagonal asm calls are not made.  Under
    # T2_MUTATE the documented fix is applied here -- all four blocks are
    # assembled, which is the edit the entry tells a user to make.
    off_diagonal = off_diagonal or MUTATE
    mesh = MeshQuad.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                                np.linspace(0.0, 1.0, NX + 1))
    basis = Basis(mesh, ElementQuad1())
    K = laplace.assemble(basis)
    M = mass.assemble(basis)
    N = basis.N

    rng = np.random.default_rng(SEED)
    u_old = np.full(N, A_P + B_P) + PERT * rng.standard_normal(N)
    v_old = np.full(N, B_P / (A_P + B_P) ** 2) + PERT * rng.standard_normal(N)
    Ku = D_U * K + M / DT
    Kv = D_V * K + M / DT

    u_new, v_new = u_old.copy(), v_old.copy()
    history = []
    shape = None
    for _it in range(MAXIT):
        U = basis.interpolate(u_new)
        V = basis.interpolate(v_new)
        R = np.concatenate([
            M @ (u_new - u_old) / DT + D_U * K @ u_new
            - GAMMA * asm(res_u, basis, uu=U, vv=V),
            M @ (v_new - v_old) / DT + D_V * K @ v_new
            - GAMMA * asm(res_v, basis, uu=U, vv=V)])
        history.append(float(np.linalg.norm(R)))
        if len(history) > 1 and history[-1] < RTOL * history[1]:
            break
        if history[-1] > 1e14:
            break
        A_uu = Ku - GAMMA * asm(jac_uu, basis, uu=U, vv=V)
        A_vv = Kv - GAMMA * asm(jac_vv, basis, uu=U, vv=V)
        if off_diagonal:
            A_uv = -GAMMA * asm(jac_uv, basis, uu=U, vv=V)
            A_vu = -GAMMA * asm(jac_vu, basis, uu=U, vv=V)
        else:
            A_uv = None
            A_vu = None
        J = bmat([[A_uu, A_uv], [A_vu, A_vv]], format="csr")
        shape = J.shape
        dx = spsolve(J, -R)
        u_new += dx[:N]
        v_new += dx[N:]
    return basis, history, shape


def ratios(history):
    """Successive-residual ratios, starting from the first asymptotic residual.

    history[0] is taken at u_new == u_old, where the M/dt term vanishes
    identically; it is not part of the asymptotic sequence.
    """
    return [history[i + 1] / history[i] for i in range(1, len(history) - 1)]


def main() -> int:
    ok = True

    # --- the off-diagonal blocks are not optional: pin them exactly ---------
    mesh = MeshQuad.init_tensor(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    basis = Basis(mesh, ElementQuad1())
    M = mass.assemble(basis)
    u_ss, v_ss = A_P + B_P, B_P / (A_P + B_P) ** 2
    U = basis.interpolate(np.full(basis.N, u_ss))
    V = basis.interpolate(np.full(basis.N, v_ss))
    B_uv = asm(jac_uv, basis, uu=U, vv=V)
    B_vu = asm(jac_vu, basis, uu=U, vv=V)
    print(f"dfu_dv_at_ss={u_ss ** 2:.1f} dfv_du_at_ss={-2.0 * u_ss * v_ss:.1f}")
    dev_uv = float(abs(B_uv - M).max())
    dev_vu = float(abs(B_vu + 1.8 * M).max())
    print(f"offdiag_uv_minus_mass_matrix_maxabs={dev_uv:.3e}")
    print(f"offdiag_vu_plus_1p8_mass_maxabs={dev_vu:.3e}")
    print(f"offdiag_blocks_match_mass_matrix={dev_uv < 1e-15 and dev_vu < 1e-15}")
    print(f"offdiag_blocks_nonzero="
          f"{abs(B_uv).max() > 0.0 and abs(B_vu).max() > 0.0}")
    if not (dev_uv < 1e-15 and dev_vu < 1e-15):
        print("FAIL: the reaction off-diagonal blocks no longer reduce to the "
              "mass matrix at the analytic steady state", file=sys.stderr)
        ok = False

    # --- RIGHT variant: all four blocks ------------------------------------
    b_full, h_full, shape_full = newton(off_diagonal=True)
    r_full = ratios(h_full)
    n_full = len(h_full) - 2
    print(f"n_dofs={b_full.N} jacobian_shape={shape_full[0]}x{shape_full[1]}")
    print("full_residuals=" + " ".join(f"{x:.3e}" for x in h_full))
    print("full_ratios=" + " ".join(f"{x:.3e}" for x in r_full))
    print(f"full_updates_from_r1={n_full}")
    print(f"full_updates_le_3={n_full <= 3}")
    print(f"full_min_ratio_lt_1e_6={min(r_full) < 1e-6}")
    dec_full = np.log10(r_full[0] / r_full[-1]) if len(r_full) > 1 else 0.0
    print(f"full_ratio_decades_gained={dec_full:.2f}")
    print(f"full_ratio_decades_ge_3={dec_full >= 3.0}")
    if not (n_full <= 3 and min(r_full) < 1e-6 and dec_full >= 3.0):
        print("FAIL: the four-block Jacobian did not give quadratic Newton "
              "convergence, so the fixture has no healthy reference",
              file=sys.stderr)
        ok = False

    # --- WRONG variant: diagonal blocks only -------------------------------
    _b_diag, h_diag, shape_diag = newton(off_diagonal=False)
    r_diag = ratios(h_diag)
    n_diag = len(h_diag) - 2
    print("diag_residuals=" + " ".join(f"{x:.3e}" for x in h_diag))
    print("diag_ratios=" + " ".join(f"{x:.3e}" for x in r_diag))
    print(f"diag_updates_from_r1={n_diag}")
    print(f"diag_updates_ge_8={n_diag >= 8}")
    print(f"diag_min_ratio={min(r_diag):.3e}")
    print(f"diag_min_ratio_gt_1e_2={min(r_diag) > 1e-2}")
    dec_diag = np.log10(r_diag[0] / r_diag[-1]) if len(r_diag) > 1 else 0.0
    print(f"diag_ratio_decades_gained={dec_diag:.2f}")
    print(f"diag_ratio_decades_lt_1={dec_diag < 1.0}")
    print(f"diag_over_full_update_count={n_diag / max(n_full, 1):.2f}")
    if not (n_diag >= 8 and min(r_diag) > 1e-2 and dec_diag < 1.0):
        print("FAIL: dropping the off-diagonal blocks did not degrade Newton to "
              "a linear rate — the claim's pathology is absent", file=sys.stderr)
        ok = False

    print(f"same_jacobian_shape_both_variants={shape_full == shape_diag}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
