"""Tier-2: the reaction Jacobian is a MASS form; the stiffness pattern is spurious diffusion.

Claim: skfem reaction_diffusion#2 — reaction Jacobian blocks are assembled as
mass matrices with a pointwise coefficient. Assembling them via the stiffness
pattern instead produces spurious diffusion in J_uv / J_vu, and the sub-block
differs from a reference that uses M @ diag(df_u/dv).

Wrong variant: J_uv from ``w["c"] * dot(grad(u), grad(v))``.
Right variant: J_uv from ``w["c"] * u * v``.

Observed on scikit-fem 12.0.1 (2026-08-06), ElementQuad1:

  * The stiffness-pattern block annihilates constants exactly:
    max|row sum| / max|entry| = 2.5e-16, and S @ ones = 0 to 1e-16. So it
    reports ZERO reaction coupling for a spatially uniform change in v — the
    one direction in which df_u/dv = u^2 is certainly non-zero. The mass-form
    block reports 1.0 there, and the total of its entries equals the exact
    integral of the coefficient over the domain.
  * Its norm relative to the mass block scales like 1/h^2 — 89, 375, 1526, 6134
    for nx = 4, 8, 16, 32, i.e. x4 per mesh halving. That IS a diffusion
    operator, which is exactly what "spurious diffusion" means, and it says the
    defect gets worse, not better, under refinement.
  * Finite-difference directional derivative of the residual: the mass form is
    derivative-consistent to 8.7e-10 (FD truncation), the stiffness form is
    wrong by a relative 2.8e+02.
  * Consequence in the solver: with the stiffness pattern in J_uv, Newton on one
    backward-Euler step never converges in 25 iterations and ends with a
    residual 3 orders ABOVE where it started; with the mass pattern it converges
    in 2 updates to 4.2e-13.

On the claim's "reference scipy implementation that uses M @ diag(df_u/dv)":
M @ diag(c) and asm(mass_pointwise, c) are NOT the same matrix. Each is the
exact derivative of a DIFFERENT residual — M @ diag(c) of the nodal residual
M @ f(u_nodal), asm(mass_pointwise, c) of the quadrature residual
asm(f(u_h) * v). The fixture prints the gap for a smooth coefficient
(1.7e-01, 4.9e-02, 1.6e-02, 8.2e-03 at nx = 4, 8, 16, 32): they agree only in
the limit h -> 0, so "differs from the M @ diag reference" is not by itself
evidence of a bug.

Cost: assembly-only checks plus one 25-iteration Newton step on a 16x16 grid,
about 2 s.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.sparse import bmat, diags
from scipy.sparse.linalg import spsolve

from skfem import (Basis, BilinearForm, ElementQuad1, LinearForm, MeshQuad,
                   asm)
from skfem.helpers import dot, grad
from skfem.models.poisson import laplace, mass

A_P, B_P = 0.1, 0.9
D_U, D_V = 1.0, 40.0
GAMMA = 1000.0
NX = 16
DT = 1e-4
PERT = 0.3
SEED = 7


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
def jac_vv(u, v, w):
    return (-w["uu"] ** 2) * u * v


@BilinearForm
def jac_vu(u, v, w):
    return (-2.0 * w["uu"] * w["vv"]) * u * v


@BilinearForm
def jac_uv_mass(u, v, w):
    """RIGHT: df_u/dv = u^2 enters as a mass form with a pointwise coefficient."""
    return (w["uu"] ** 2) * u * v


@BilinearForm
def jac_uv_stiffness(u, v, w):
    """WRONG: the same coefficient dropped into the stiffness pattern."""
    return (w["uu"] ** 2) * dot(grad(u), grad(v))


@BilinearForm
def coeff_mass(u, v, w):
    return w["c"] * u * v


@BilinearForm
def coeff_stiffness(u, v, w):
    return w["c"] * dot(grad(u), grad(v))


def grid(nx):
    mesh = MeshQuad.init_tensor(np.linspace(0.0, 1.0, nx + 1),
                                np.linspace(0.0, 1.0, nx + 1))
    return Basis(mesh, ElementQuad1())


def newton(off_diagonal_form):
    """One backward-Euler step; returns the residual history."""
    basis = grid(NX)
    K = laplace.assemble(basis)
    M = mass.assemble(basis)
    N = basis.N
    rng = np.random.default_rng(SEED)
    u_old = np.full(N, A_P + B_P) + PERT * rng.standard_normal(N)
    v_old = np.full(N, B_P / (A_P + B_P) ** 2) + PERT * rng.standard_normal(N)
    Ku, Kv = D_U * K + M / DT, D_V * K + M / DT
    u_new, v_new = u_old.copy(), v_old.copy()
    history = []
    for _it in range(25):
        U, V = basis.interpolate(u_new), basis.interpolate(v_new)
        R = np.concatenate([
            M @ (u_new - u_old) / DT + D_U * K @ u_new
            - GAMMA * asm(res_u, basis, uu=U, vv=V),
            M @ (v_new - v_old) / DT + D_V * K @ v_new
            - GAMMA * asm(res_v, basis, uu=U, vv=V)])
        history.append(float(np.linalg.norm(R)))
        if len(history) > 1 and history[-1] < 1e-9 * history[1]:
            break
        if history[-1] > 1e14:
            break
        J = bmat([[Ku - GAMMA * asm(jac_uu, basis, uu=U, vv=V),
                   -GAMMA * asm(off_diagonal_form, basis, uu=U, vv=V)],
                  [-GAMMA * asm(jac_vu, basis, uu=U, vv=V),
                   Kv - GAMMA * asm(jac_vv, basis, uu=U, vv=V)]], format="csr")
        dx = spsolve(J, -R)
        u_new += dx[:N]
        v_new += dx[N:]
    return history


def main() -> int:
    ok = True

    # --- 1. the stiffness pattern annihilates constants ---------------------
    basis = grid(8)
    N = basis.N
    rng = np.random.default_rng(3)
    u_state = np.full(N, A_P + B_P) + 0.2 * rng.standard_normal(N)
    v_state = np.full(N, B_P / (A_P + B_P) ** 2) + 0.2 * rng.standard_normal(N)
    U, V = basis.interpolate(u_state), basis.interpolate(v_state)
    B_mass = asm(jac_uv_mass, basis, uu=U, vv=V)
    B_stiff = asm(jac_uv_stiffness, basis, uu=U, vv=V)
    one = np.ones(N)
    act_mass = float(np.abs(B_mass @ one).max())
    act_stiff = float(np.abs(B_stiff @ one).max())
    print(f"element={type(ElementQuad1()).__name__} n_dofs={N}")
    print(f"mass_block_action_on_uniform_max={act_mass:.4e}")
    print(f"stiffness_block_action_on_uniform_max={act_stiff:.4e}")
    print(f"stiffness_block_kills_uniform_coupling={act_stiff < 1e-14}")
    print(f"mass_block_keeps_uniform_coupling={act_mass > 1e-3}")
    if not (act_stiff < 1e-14 and act_mass > 1e-3):
        print("FAIL: the stiffness-pattern block no longer annihilates a "
              "uniform perturbation", file=sys.stderr)
        ok = False

    # total of the mass-form entries == exact integral of the coefficient
    smooth = grid(16)
    xs, ys = smooth.doflocs
    cvec = 1.0 + 0.5 * np.sin(2.0 * np.pi * xs) * np.cos(2.0 * np.pi * ys)
    C = smooth.interpolate(cvec)
    Mc = asm(coeff_mass, smooth, c=C)
    Sc = asm(coeff_stiffness, smooth, c=C)
    total = float(Mc.sum())
    print(f"sum_of_mass_form_entries={total:.10f} exact_integral_of_c=1.0")
    print(f"mass_form_total_equals_integral={abs(total - 1.0) < 1e-6}")
    rowsum_rel = float(np.abs(np.array(Sc.sum(axis=1)).ravel()).max()
                       / abs(Sc).max())
    print(f"stiffness_form_max_rowsum_over_max_entry={rowsum_rel:.3e}")
    print(f"stiffness_form_rowsums_are_zero={rowsum_rel < 1e-12}")
    if not (abs(total - 1.0) < 1e-6 and rowsum_rel < 1e-12):
        print("FAIL: the mass/stiffness algebraic identities no longer hold",
              file=sys.stderr)
        ok = False

    # --- 2. the stiffness pattern is a diffusion operator: 1/h^2 growth -----
    ratios = []
    gaps = []
    for nx in (4, 8, 16, 32):
        b = grid(nx)
        x2, y2 = b.doflocs
        c2 = 1.0 + 0.5 * np.sin(2.0 * np.pi * x2) * np.cos(2.0 * np.pi * y2)
        C2 = b.interpolate(c2)
        m_c = asm(coeff_mass, b, c=C2)
        s_c = asm(coeff_stiffness, b, c=C2)
        m_plain = mass.assemble(b)
        ratios.append(float(abs(s_c).max() / abs(m_c).max()))
        gaps.append(float(abs(m_c - m_plain @ diags(c2)).max() / abs(m_c).max()))
    print("stiffness_over_mass_norm_ratio=" + " ".join(f"{r:.1f}" for r in ratios))
    growth = [ratios[i + 1] / ratios[i] for i in range(len(ratios) - 1)]
    print("norm_ratio_growth_per_halving=" + " ".join(f"{g:.2f}" for g in growth))
    quad = all(3.0 < g < 5.0 for g in growth)
    print(f"norm_ratio_grows_like_one_over_h_squared={quad}")
    if not quad:
        print("FAIL: the stiffness-pattern block does not scale like 1/h^2, so "
              "it cannot be identified as spurious diffusion", file=sys.stderr)
        ok = False
    print("mass_pointwise_vs_M_at_diag_gap=" + " ".join(f"{g:.2e}" for g in gaps))
    print(f"mass_pointwise_differs_from_M_at_diag={gaps[0] > 1e-2}")
    print(f"gap_shrinks_under_refinement={gaps[-1] < gaps[0] / 10.0}")

    # --- 3. finite-difference consistency of each candidate block ----------
    eps = 1e-7
    direction = rng.standard_normal(N)

    def reaction_residual(u_vec, v_vec):
        return -GAMMA * asm(res_u, basis, uu=basis.interpolate(u_vec),
                            vv=basis.interpolate(v_vec))

    fd = (reaction_residual(u_state, v_state + eps * direction)
          - reaction_residual(u_state, v_state)) / eps
    err_mass = float(np.linalg.norm(fd - (-GAMMA * (B_mass @ direction)))
                     / np.linalg.norm(fd))
    err_stiff = float(np.linalg.norm(fd - (-GAMMA * (B_stiff @ direction)))
                      / np.linalg.norm(fd))
    print(f"fd_relerr_mass_form={err_mass:.3e}")
    print(f"fd_relerr_stiffness_form={err_stiff:.3e}")
    print(f"mass_form_is_derivative_consistent={err_mass < 1e-6}")
    print(f"stiffness_form_relerr_gt_1={err_stiff > 1.0}")
    if not (err_mass < 1e-6 and err_stiff > 1.0):
        print("FAIL: the finite-difference check no longer separates the mass "
              "form from the stiffness form", file=sys.stderr)
        ok = False

    # --- 4. consequence in the Newton solve --------------------------------
    h_mass = newton(jac_uv_mass)
    h_stiff = newton(jac_uv_stiffness)
    conv_mass = h_mass[-1] < 1e-9 * h_mass[1]
    conv_stiff = h_stiff[-1] < 1e-9 * h_stiff[1]
    print("mass_form_newton_residuals=" + " ".join(f"{x:.3e}" for x in h_mass))
    print(f"mass_form_newton_updates={len(h_mass) - 2}")
    print(f"mass_form_newton_converged={conv_mass}")
    print("stiffness_form_newton_residuals="
          + " ".join(f"{x:.3e}" for x in h_stiff[:6]))
    print(f"stiffness_form_newton_iterations_used={len(h_stiff)}")
    print(f"stiffness_form_newton_converged={conv_stiff}")
    print(f"stiffness_form_final_residual_above_start="
          f"{h_stiff[-1] > h_stiff[1]}")
    if not conv_mass:
        print("FAIL: the mass-form Jacobian no longer converges, so there is no "
              "healthy reference", file=sys.stderr)
        ok = False
    if conv_stiff or not h_stiff[-1] > h_stiff[1]:
        print("FAIL: the stiffness-pattern Jacobian converged — the claim's "
              "pathology is absent", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
