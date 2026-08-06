"""Tier-2: u_ss = a+b, v_ss = b/(a+b)^2 — and comparing against it catches a mistyped a or b.

Claim: skfem reaction_diffusion#5 — the Schnakenberg homogeneous steady state is
u_ss = a + b, v_ss = b/(a+b)^2. Initial conditions that do not match it produce a
long startup transient before patterns emerge, and comparing the simulated u
against the analytic steady state catches a mis-typed (a, b): a 10% deviation in
u_ss means a or b is wrong by a similar factor.

Three things are exercised, all on scikit-fem 12.0.1 (2026-08-06), ElementQuad1
on a 4x4 grid (25 DOFs), backward Euler dt=2e-4, gamma=1000. d_v = d_u here on
purpose: the uniform state must be linearly STABLE so the measurement isolates
the (a, b) parameterisation. The Turing mechanism is exercised in
rd_diffusion_ratio.

1. The formula. At (u_ss, v_ss) the assembled reaction load is 9.8e-18 —
   machine zero, so the state is an exact equilibrium of the discrete reaction
   term. Dropping the square, v = b/(a+b), leaves it at 9.8e-18 too for the
   SHIPPED parameters (a, b) = (0.1, 0.9), because a + b = 1 there and the
   square is invisible. At (a, b) = (0.05, 1.0) the correct load is 0.0 and the
   dropped-square load is 1.1e-02. Recorded because it means the shipped
   parameter set cannot detect that particular typo at all.

2. Wrong initial condition. Swapping a and b in the v formula gives
   v0 = a/(a+b)^2 = 0.1 instead of 0.9. Peak |mean(u) - u_ss| / u_ss during the
   run is 6.8e-01 — a 68% excursion — and it takes 200 steps (T = 0.04, about
   40 reaction times 1/gamma) to fall back to 2.5e-05. With the correct v0 the
   deviation is exactly 0.0 at every step.

3. Mistyped parameter. With b = 0.99 in the equations but the initial condition
   set to the believed (1.0, 0.9), the run settles at mean(u) = 1.0900009885 —
   exactly a + b_typo. The 10% error in b shows up as a 9.0001% deviation from
   the expected u_ss = 1.0, which is the claim's diagnostic.

Cost: three uniform-state runs on a 4x4 grid plus assembly-only checks, ~6 s.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

from skfem import (Basis, BilinearForm, ElementQuad1, LinearForm, MeshQuad,
                   asm)
from skfem.models.poisson import laplace, mass

A_P, B_P = 0.1, 0.9
D_U = D_V = 1.0          # uniform state must be stable; see the docstring
GAMMA = 1000.0
NX = 4
DT = 2e-4
N_STEPS = 200

U0_BELIEVED = A_P + B_P                     # 1.0
V0_RIGHT = B_P / (A_P + B_P) ** 2           # 0.9
V0_WRONG = A_P / (A_P + B_P) ** 2           # 0.1 — a and b swapped

B_TYPO = 0.99                               # b mistyped by +10%


def forms(a: float, b: float):
    @LinearForm
    def res_u(v, w):
        return (a - w["uu"] + w["uu"] ** 2 * w["vv"]) * v

    @LinearForm
    def res_v(v, w):
        return (b - w["uu"] ** 2 * w["vv"]) * v

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


def grid():
    mesh = MeshQuad.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                                np.linspace(0.0, 1.0, NX + 1))
    return Basis(mesh, ElementQuad1())


def reaction_load(a: float, b: float, u_val: float, v_val: float) -> float:
    """||assembled reaction load|| at a uniform state — zero iff it is an equilibrium."""
    basis = grid()
    res_u, res_v = forms(a, b)
    U = basis.interpolate(np.full(basis.N, u_val))
    V = basis.interpolate(np.full(basis.N, v_val))
    return max(float(np.linalg.norm(asm(res_u, basis, uu=U, vv=V))),
               float(np.linalg.norm(asm(res_v, basis, uu=U, vv=V))))


def relax(b_in_equations: float, u0: float, v0: float):
    """Uniform-state relaxation; returns the trace of mean(u)."""
    basis = grid()
    res_u, res_v = forms(A_P, b_in_equations)
    K = laplace.assemble(basis)
    M = mass.assemble(basis)
    N = basis.N
    u = np.full(N, u0)
    v = np.full(N, v0)
    Ku, Kv = D_U * K + M / DT, D_V * K + M / DT
    trace = [float(u.mean())]
    for _step in range(N_STEPS):
        u_old, v_old = u.copy(), v.copy()
        u_new, v_new = u_old.copy(), v_old.copy()
        for _it in range(25):
            U, V = basis.interpolate(u_new), basis.interpolate(v_new)
            R = np.concatenate([
                M @ (u_new - u_old) / DT + D_U * K @ u_new
                - GAMMA * asm(res_u, basis, uu=U, vv=V),
                M @ (v_new - v_old) / DT + D_V * K @ v_new
                - GAMMA * asm(res_v, basis, uu=U, vv=V)])
            J = bmat([[Ku - GAMMA * asm(jac_uu, basis, uu=U, vv=V),
                       -GAMMA * asm(jac_uv, basis, uu=U, vv=V)],
                      [-GAMMA * asm(jac_vu, basis, uu=U, vv=V),
                       Kv - GAMMA * asm(jac_vv, basis, uu=U, vv=V)]],
                     format="csr")
            dx = spsolve(J, -R)
            u_new += dx[:N]
            v_new += dx[N:]
            if np.linalg.norm(dx) < 1e-12:
                break
        u, v = u_new, v_new
        trace.append(float(u.mean()))
    return basis, trace


def main() -> int:
    ok = True
    print(f"element={type(ElementQuad1()).__name__} n_dofs={grid().N}")
    print(f"u_ss={U0_BELIEVED:.1f} v_ss={V0_RIGHT:.1f}")

    # --- 1. the formula ----------------------------------------------------
    load_ok = reaction_load(A_P, B_P, U0_BELIEVED, V0_RIGHT)
    print(f"reaction_load_at_analytic_ss={load_ok:.4e}")
    print(f"reaction_load_at_analytic_ss_lt_1e_15={load_ok < 1e-15}")
    if load_ok >= 1e-15:
        print("FAIL: (a+b, b/(a+b)^2) is no longer an equilibrium of the "
              "assembled reaction term", file=sys.stderr)
        ok = False

    load_nosq_shipped = reaction_load(A_P, B_P, U0_BELIEVED, B_P / (A_P + B_P))
    a2, b2 = 0.05, 1.0
    load_ok_2 = reaction_load(a2, b2, a2 + b2, b2 / (a2 + b2) ** 2)
    load_nosq_2 = reaction_load(a2, b2, a2 + b2, b2 / (a2 + b2))
    print(f"dropped_square_load_shipped_params={load_nosq_shipped:.4e}")
    print(f"dropped_square_load_at_a_plus_b_ne_1={load_nosq_2:.4e}")
    print(f"second_param_set_analytic_load={load_ok_2:.4e}")
    print(f"dropped_square_invisible_at_a_plus_b_eq_1="
          f"{load_nosq_shipped < 1e-15}")
    print(f"dropped_square_detected_when_a_plus_b_ne_1="
          f"{load_ok_2 < 1e-15 < load_nosq_2 and load_nosq_2 > 1e-3}")
    if not (load_nosq_shipped < 1e-15 and load_nosq_2 > 1e-3):
        print("FAIL: the dropped-square probe no longer behaves as recorded",
              file=sys.stderr)
        ok = False

    # --- 2. RIGHT vs WRONG initial condition -------------------------------
    _b, trace_right = relax(B_P, U0_BELIEVED, V0_RIGHT)
    dev_right = [abs(m - U0_BELIEVED) / U0_BELIEVED for m in trace_right]
    print(f"correct_ic_peak_rel_dev={max(dev_right):.3e}")
    print(f"correct_ic_peak_rel_dev_lt_1e_12={max(dev_right) < 1e-12}")
    if max(dev_right) >= 1e-12:
        print("FAIL: the analytic steady state did not stay put, so there is no "
              "healthy reference", file=sys.stderr)
        ok = False

    _b2, trace_wrong = relax(B_P, U0_BELIEVED, V0_WRONG)
    dev_wrong = [abs(m - U0_BELIEVED) / U0_BELIEVED for m in trace_wrong]
    print(f"wrong_ic_v0={V0_WRONG:.1f} (a and b swapped in v_ss)")
    print(f"wrong_ic_peak_rel_dev={max(dev_wrong):.4e}")
    print(f"wrong_ic_final_rel_dev={dev_wrong[-1]:.3e}")
    print(f"wrong_ic_peak_rel_dev_gt_0p1={max(dev_wrong) > 0.1}")
    print(f"wrong_ic_returns_to_ss={dev_wrong[-1] < 1e-3}")
    steps_over_1pct = sum(1 for d in dev_wrong if d > 0.01)
    print(f"wrong_ic_steps_above_1pct={steps_over_1pct} "
          f"of {len(dev_wrong)} (transient lasts many reaction times 1/gamma)")
    if not (max(dev_wrong) > 0.1 and dev_wrong[-1] < 1e-3):
        print("FAIL: the mismatched initial condition produced no startup "
              "transient — the claim's pathology is absent", file=sys.stderr)
        ok = False

    # --- 3. a mistyped b, detected by comparing against a+b ----------------
    _b3, trace_typo = relax(B_TYPO, U0_BELIEVED, V0_RIGHT)
    final = trace_typo[-1]
    pct = (final - U0_BELIEVED) / U0_BELIEVED * 100.0
    print(f"mistyped_b={B_TYPO} final_mean_u={final:.10f} "
          f"a_plus_b_typo={A_P + B_TYPO:.4f}")
    print(f"mistyped_b_final_matches_a_plus_b_typo="
          f"{abs(final - (A_P + B_TYPO)) < 1e-5}")
    print(f"mistyped_b_rel_dev_pct={pct:.4f}")
    print(f"mistyped_b_rel_dev_pct_between_5_and_15={5.0 < pct < 15.0}")
    if not (abs(final - (A_P + B_TYPO)) < 1e-5 and 5.0 < pct < 15.0):
        print("FAIL: a 10% error in b no longer shows up as a comparable "
              "deviation of u from the analytic steady state", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
