"""Tier-2: rc=0 from a reaction-diffusion run is NOT evidence of a Turing pattern.

Claim: skfem reaction_diffusion#0 — the shipped ``reaction_diffusion_2d``
template (Schnakenberg, a=0.1, b=0.9, d_u=1, d_v=40, gamma=1000, ElementQuad1 on
a 32x32 quad grid = 1089 DOFs, backward Euler dt=0.5 to T=50, u/v seeded with a
1e-2 random perturbation about (u_ss, v_ss) = (1.0, 0.9)) exits cleanly while
decaying straight back to the uniform state: max(u) - min(u) at T=50 is 3.7e-14,
machine noise, even though d_v/d_u = 40.

Wrong variant: the shipped time scheme verbatim — backward Euler at dt=0.5.
Right variant: byte-identical mesh, element, parameters, seed and assembly; only
dt is changed to 0.002. A saturated Turing pattern forms, spread ~4.3.

Observed on scikit-fem 12.0.1 (2026-08-06): shipped run ends at
u_max=1.0000000000000104, u_min=0.999999999999973, spread 3.741e-14 — the
claim's numbers reproduce exactly. dt=0.002 ends at u in [0.12, 4.69].

Mechanism (printed by the fixture from the dispersion relation, so a library
change in either direction is caught): for these parameters the homogeneous
state is linearly UNSTABLE, fastest-growing Neumann mode k^2 = pi^2*(3^2+3^2)
with growth rate lambda = +411. Backward Euler multiplies that mode by
1/(1 - lambda*dt) per step. At dt=0.5 that is 1/(1-205.5) = -0.0049: the mode
that must grow is damped by a factor 200 per step. At dt=0.002 it is
1/(1-0.822) = +5.6: it grows. Unconditional stability is precisely the defect —
it stabilises modes that the physics requires to be unstable.

Cost: the wrong variant runs the full shipped 100 steps (~2.5 s) so the claim's
figures are reproduced literally. The right variant needs only 20 steps of
dt=0.002 (T=0.04, ~6 s): gamma=1000 makes the reaction time scale 1e-3, the seed
saturates by step 8, and the pattern is stationary afterwards (checked out to
T=0.2 during authoring: spread 4.27 -> 4.33 -> 4.31).
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

from skfem import Basis, BilinearForm, ElementQuad1, MeshQuad, asm
from skfem.models.poisson import laplace, mass

A_P, B_P = 0.1, 0.9
D_U, D_V = 1.0, 40.0
GAMMA = 1000.0
NX = 32
SEED = 42
PERT = 1e-2
NEWTON_TOL = 1e-8

# The shipped template's time discretisation.  Mutating this pair is what the
# mutation check does: with (0.002, 20) the "wrong" branch also patterns.
DT_SHIPPED, N_STEPS_SHIPPED = 0.5, 100
DT_FIXED, N_STEPS_FIXED = 0.002, 20


@BilinearForm
def mass_pointwise(u, v, w):
    """Mass matrix with a pointwise coefficient — the shipped template's form."""
    return w["c"] * u * v


def growth_rate(d_u: float, d_v: float, gamma: float, nmax: int = 12):
    """Largest linear growth rate over the Neumann modes of the unit square."""
    u_ss, v_ss = A_P + B_P, B_P / (A_P + B_P) ** 2
    f_u = -1.0 + 2.0 * u_ss * v_ss
    f_v = u_ss ** 2
    g_u = -2.0 * u_ss * v_ss
    g_v = -u_ss ** 2
    det = f_u * g_v - f_v * g_u
    d = d_v / d_u
    best, mode = -np.inf, None
    for n in range(nmax):
        for mm in range(nmax):
            k2 = np.pi ** 2 * (n * n + mm * mm)
            trace = gamma * (f_u + g_v) - (1.0 + d) * k2
            h = d * k2 ** 2 - gamma * (d * f_u + g_v) * k2 + gamma ** 2 * det
            disc = trace * trace - 4.0 * h
            lam = 0.5 * (trace + np.sqrt(disc)) if disc >= 0.0 else 0.5 * trace
            if lam > best:
                best, mode = lam, (n, mm)
    return best, mode


def schnakenberg(dt: float, n_steps: int):
    """The shipped assembly/solve pipeline, run for n_steps of size dt."""
    mesh = MeshQuad.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                                np.linspace(0.0, 1.0, NX + 1))
    basis = Basis(mesh, ElementQuad1())
    K = laplace.assemble(basis)
    M = mass.assemble(basis)
    N = basis.N

    rng = np.random.default_rng(SEED)
    u_ss, v_ss = A_P + B_P, B_P / (A_P + B_P) ** 2
    u = np.full(N, u_ss) + PERT * rng.standard_normal(N)
    v = np.full(N, v_ss) + PERT * rng.standard_normal(N)

    Ku = D_U * K + M / dt
    Kv = D_V * K + M / dt

    for _ in range(n_steps):
        u_old, v_old = u.copy(), v.copy()
        u_new, v_new = u_old.copy(), v_old.copy()
        for _nit in range(20):
            f_u = A_P - u_new + u_new ** 2 * v_new
            f_v = B_P - u_new ** 2 * v_new
            J = bmat(
                [[Ku - GAMMA * asm(mass_pointwise, basis,
                                   c=(-1.0 + 2.0 * u_new * v_new)),
                  -GAMMA * asm(mass_pointwise, basis, c=u_new ** 2)],
                 [-GAMMA * asm(mass_pointwise, basis, c=(-2.0 * u_new * v_new)),
                  Kv - GAMMA * asm(mass_pointwise, basis, c=(-u_new ** 2))]],
                format="csr")
            R = np.concatenate([
                M @ (u_new - u_old) / dt + D_U * K @ u_new - GAMMA * M @ f_u,
                M @ (v_new - v_old) / dt + D_V * K @ v_new - GAMMA * M @ f_v])
            dx = spsolve(J, -R)
            u_new += dx[:N]
            v_new += dx[N:]
            if np.linalg.norm(dx) < NEWTON_TOL:
                break
        u, v = u_new, v_new

    return basis, u, v


def main() -> int:
    ok = True

    lam, mode = growth_rate(D_U, D_V, GAMMA)
    print(f"element={type(ElementQuad1()).__name__}")
    print(f"u_ss={A_P + B_P:.1f} v_ss={B_P / (A_P + B_P) ** 2:.1f}")
    print(f"fastest_growing_mode=({mode[0]},{mode[1]}) lambda_max={lam:.4g}")
    print(f"lambda_max_gt_0={lam > 0.0}")
    if lam <= 0.0:
        print("FAIL: the parameter set is not linearly Turing-unstable, so the "
              "premise of the claim does not hold", file=sys.stderr)
        ok = False

    amp_shipped = 1.0 / (1.0 - lam * DT_SHIPPED)
    amp_fixed = 1.0 / (1.0 - lam * DT_FIXED)
    print(f"backward_euler_amplification_at_dt_{DT_SHIPPED}={amp_shipped:.4g}")
    print(f"backward_euler_amplification_at_dt_{DT_FIXED}={amp_fixed:.4g}")
    print(f"be_amp_at_shipped_dt_lt_1={abs(amp_shipped) < 1.0}")
    print(f"be_amp_at_fixed_dt_gt_1={abs(amp_fixed) > 1.0}")
    if not abs(amp_shipped) < 1.0:
        print("FAIL: backward Euler at the shipped dt does not damp the "
              "unstable mode", file=sys.stderr)
        ok = False
    if not abs(amp_fixed) > 1.0:
        print("FAIL: backward Euler at the reduced dt does not amplify the "
              "unstable mode", file=sys.stderr)
        ok = False

    # --- WRONG variant: the shipped dt -------------------------------------
    basis, u_w, _v_w = schnakenberg(DT_SHIPPED, N_STEPS_SHIPPED)
    spread_w = float(u_w.max() - u_w.min())
    print(f"n_dofs={basis.N} n_elements={basis.mesh.nelements}")
    print(f"shipped dt={DT_SHIPPED} steps={N_STEPS_SHIPPED} "
          f"u_max={u_w.max()!r} u_min={u_w.min()!r}")
    print(f"shipped_spread={spread_w:.6e}")
    print(f"shipped_spread_lt_1e_10={spread_w < 1e-10}")
    print(f"shipped_min_max_identical_to_4dp="
          f"{f'{u_w.min():.4f}' == f'{u_w.max():.4f}'}")
    if spread_w >= 1e-10:
        print("FAIL: the shipped time scheme did not decay to machine noise — "
              "the claim's pathology is absent", file=sys.stderr)
        ok = False

    # --- RIGHT variant: same everything, dt small enough -------------------
    _basis2, u_f, _v_f = schnakenberg(DT_FIXED, N_STEPS_FIXED)
    spread_f = float(u_f.max() - u_f.min())
    print(f"fixed dt={DT_FIXED} steps={N_STEPS_FIXED} "
          f"u=[{u_f.min():.5f}, {u_f.max():.5f}]")
    print(f"fixed_spread={spread_f:.6e}")
    print(f"fixed_spread_gt_1p0={spread_f > 1.0}")
    if spread_f <= 1.0:
        print("FAIL: the reduced time step did not produce a pattern either, so "
              "the fixture cannot attribute the flat state to backward Euler",
              file=sys.stderr)
        ok = False

    ratio = spread_f / max(spread_w, 1e-300)
    print(f"spread_ratio={ratio:.3e}")
    print(f"spread_ratio_gt_1e10={ratio > 1e10}")
    if ratio <= 1e10:
        print("FAIL: the two schemes did not separate by 10 orders of magnitude",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
