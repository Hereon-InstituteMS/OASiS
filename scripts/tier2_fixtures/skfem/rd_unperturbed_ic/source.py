"""Tier-2: starting exactly at (u_ss, v_ss) gives no pattern — it is an exact discrete equilibrium.

Claim: skfem reaction_diffusion#3 — the initial condition must perturb the
homogeneous steady state to trigger the Turing instability. Starting from
exactly (u_ss, v_ss) gives no pattern formation; add a perturbation of
amplitude ~1e-3 * u_ss.

Wrong variant: u = u_ss, v = v_ss with NO perturbation.
Right variant: the same run with a 1e-3 random perturbation.

Observed on scikit-fem 12.0.1 (2026-08-06). 16x16 ElementQuad1 (289 DOFs),
Schnakenberg a=0.1 b=0.9 d_u=1 d_v=40 gamma=1000, backward Euler dt=2e-3,
10 steps:

  unperturbed  max(u) - min(u) = 2.97e-08
  perturbed    max(u) - min(u) = 4.48e+00      ratio 1.5e+08

and the uniform state is an EXACT discrete equilibrium: the assembled residual
there is ~1e-13, because laplace @ ones = 0 to round-off and f(u_ss, v_ss) = 0
identically.

Refinement of the claim, printed by the fixture: "solution stays uniform
throughout the simulation" holds in exact arithmetic only. In floating point the
unperturbed run is seeded by round-off, and the fixture's own trace shows that
seed growing geometrically at the same rate as the deliberate one — 9.7e-15,
5.2e-14, 2.5e-13, ... about x5.5 per step, which is the backward-Euler
amplification 1/(1 - lambda*dt) of the fastest-growing mode. During authoring
the identical run reached spread 1.7e-01 at step 19 and 4.0e+01 at step 20. The
correct statement is that an unperturbed start DELAYS onset by
log(perturbation / machine-eps) / log(amplification) steps — here about 17 —
not that it prevents it.

Cost: two 10-step runs on a 16x16 grid, under 2 s. Ten steps is the smallest
count that has the seeded run saturated (it saturates at step 8) while the
round-off seed is still 8 orders below it.

Mutation control: T2_MUTATE=1 applies the claim's own remedy at the pathology
site -- PERT_WRONG, the amplitude of the perturbation added to the "wrong"
run's initial condition, becomes PERT_RIGHT (1e-3) instead of 0.0. That run is
then no longer started exactly at (u_ss, v_ss), so it is not an equilibrium and
it forms the same pattern as the seeded run. Re-run: T2_MUTATE=1 python
source.py
"""
from __future__ import annotations

import os
import sys

MUTATE = os.environ.get("T2_MUTATE") == "1"

import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

from skfem import (Basis, BilinearForm, ElementQuad1, LinearForm, MeshQuad,
                   asm)
from skfem.models.poisson import laplace, mass

A_P, B_P = 0.1, 0.9
D_U, D_V = 1.0, 40.0
GAMMA = 1000.0
NX = 16
DT = 2e-3
N_STEPS = 10
SEED = 42

PERT_RIGHT = 1e-3       # the claim's recommended ~1e-3 * u_ss
# exactly (u_ss, v_ss) — the claim's mistake; under mutation it is perturbed
PERT_WRONG = 0.0 if not MUTATE else PERT_RIGHT


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


def schnakenberg(perturbation: float):
    """Backward Euler + Newton; returns (basis, u, spread trace, |R| at t=0)."""
    mesh = MeshQuad.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                                np.linspace(0.0, 1.0, NX + 1))
    basis = Basis(mesh, ElementQuad1())
    K = laplace.assemble(basis)
    M = mass.assemble(basis)
    N = basis.N
    rng = np.random.default_rng(SEED)
    u = np.full(N, A_P + B_P) + perturbation * rng.standard_normal(N)
    v = np.full(N, B_P / (A_P + B_P) ** 2) + perturbation * rng.standard_normal(N)
    Ku, Kv = D_U * K + M / DT, D_V * K + M / DT

    U0, V0 = basis.interpolate(u), basis.interpolate(v)
    r0 = float(np.linalg.norm(np.concatenate([
        D_U * K @ u - GAMMA * asm(res_u, basis, uu=U0, vv=V0),
        D_V * K @ v - GAMMA * asm(res_v, basis, uu=U0, vv=V0)])))

    trace = [float(u.max() - u.min())]
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
            if np.linalg.norm(dx) < 1e-10:
                break
        u, v = u_new, v_new
        trace.append(float(u.max() - u.min()))
    return basis, u, trace, r0


def main() -> int:
    ok = True

    # --- WRONG variant: no perturbation at all -----------------------------
    basis, _u_w, trace_w, r0_w = schnakenberg(PERT_WRONG)
    spread_w = trace_w[-1]
    print(f"element={type(ElementQuad1()).__name__} n_dofs={basis.N} "
          f"n_elements={basis.mesh.nelements}")
    print(f"u_ss={A_P + B_P:.1f} v_ss={B_P / (A_P + B_P) ** 2:.1f}")
    print(f"initial_spread_unperturbed={trace_w[0]:.1e}")
    print(f"residual_at_uniform_state={r0_w:.3e}")
    print(f"uniform_state_is_exact_discrete_equilibrium={r0_w < 1e-10}")
    print("unperturbed_spread_trace=" + " ".join(f"{s:.2e}" for s in trace_w))
    print(f"unperturbed_spread={spread_w:.4e}")
    print(f"unperturbed_spread_lt_1e_5={spread_w < 1e-5}")
    if r0_w >= 1e-10:
        print("FAIL: (u_ss, v_ss) is no longer an exact discrete equilibrium",
              file=sys.stderr)
        ok = False
    if spread_w >= 1e-5:
        print("FAIL: the unperturbed run developed structure — the claim's "
              "pathology is absent", file=sys.stderr)
        ok = False

    # honest caveat: the round-off seed IS growing, at the amplification rate
    growing = all(trace_w[i + 1] > trace_w[i] for i in range(2, len(trace_w) - 1))
    late = [trace_w[i + 1] / trace_w[i] for i in range(3, len(trace_w) - 1)]
    print(f"unperturbed_seed_growth_per_step={np.mean(late):.2f}")
    print(f"unperturbed_seed_grows_from_roundoff={growing}")

    # --- RIGHT variant: the recommended 1e-3 perturbation ------------------
    _b2, u_r, trace_r, _r0 = schnakenberg(PERT_RIGHT)
    spread_r = trace_r[-1]
    print(f"perturbation_amplitude={PERT_RIGHT:g}")
    print(f"perturbed u=[{u_r.min():.5f}, {u_r.max():.5f}]")
    print(f"perturbed_spread={spread_r:.4e}")
    print(f"perturbed_spread_gt_1p0={spread_r > 1.0}")
    if spread_r <= 1.0:
        print("FAIL: the perturbed run did not form a pattern, so the fixture "
              "cannot attribute the flat state to the initial condition",
              file=sys.stderr)
        ok = False

    ratio = spread_r / max(spread_w, 1e-300)
    print(f"spread_ratio={ratio:.3e}")
    print(f"spread_ratio_gt_1e6={ratio > 1e6}")
    if ratio <= 1e6:
        print("FAIL: perturbed and unperturbed runs did not separate",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
