"""Tier-2: zero-flux BCs are natural; imposing u=0 on the boundary instead destroys the pattern.

Claim: skfem reaction_diffusion#6 — Neumann (zero-flux) BCs are natural in the
weak form, no explicit enforcement needed. Applying a Dirichlet condition with
value 0 instead silences the natural BC and imposes a far stronger constraint
(u = 0 on the boundary, not just du/dn = 0); the pattern is pulled toward zero at
the boundary instead of bulging outward.

Wrong variant: constrain every boundary DOF of both species to 0 with
``skfem.condense(J, -R, D=boundary_dofs)``.
Right variant: assemble nothing on the boundary at all and solve the full system.

Observed on scikit-fem 12.0.1 (2026-08-06). Schnakenberg a=0.1 b=0.9 d_u=1
d_v=40 gamma=1000 on a 16x16 ElementQuad1 grid (289 DOFs, 64 boundary DOFs),
backward Euler dt=2e-3, 20 steps:

  natural Neumann   u on boundary in [1.18e-01, 4.35e+00], spread 4.23
  Dirichlet u=0     u on boundary identically 0, whole field in [0, 0.125],
                    spread 0.125

so the constraint does not merely flatten the boundary, it suppresses the
pattern everywhere on this domain — the amplitude drops by a factor 34.

The "natural" half of the claim is pinned separately and exactly, on pure
diffusion where the answer is known: with no boundary treatment the total mass
1^T M u is conserved to a relative -1.2e-15 over 50 steps, which is what
"zero flux is already in the weak form" means. With the boundary DOFs condensed
to 0, 70.1% of the mass drains out over the same 50 steps.

API note, printed by the fixture: scikit-fem 12.0.1 has no ``DirichletBC``
symbol — ``hasattr(skfem, "DirichletBC")`` is False. The essential-BC API is
``condense`` / ``enforce`` / ``penalize``. The claim's wording is borrowed from
another library.

Cost: two 20-step Schnakenberg runs plus two 50-step linear diffusion runs on a
16x16 grid, about 3 s.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

import skfem
from skfem import (Basis, BilinearForm, ElementQuad1, LinearForm, MeshQuad,
                   asm, condense, solve)
from skfem.models.poisson import laplace, mass

A_P, B_P = 0.1, 0.9
D_U, D_V = 1.0, 40.0
GAMMA = 1000.0
NX = 16
DT = 2e-3
N_STEPS = 20
SEED = 42
PERT = 1e-2


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


def grid():
    mesh = MeshQuad.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                                np.linspace(0.0, 1.0, NX + 1))
    return Basis(mesh, ElementQuad1())


def schnakenberg(dirichlet: bool):
    basis = grid()
    K = laplace.assemble(basis)
    M = mass.assemble(basis)
    N = basis.N
    boundary = basis.get_dofs().flatten()
    both = np.concatenate([boundary, boundary + N])
    rng = np.random.default_rng(SEED)
    u = np.full(N, A_P + B_P) + PERT * rng.standard_normal(N)
    v = np.full(N, B_P / (A_P + B_P) ** 2) + PERT * rng.standard_normal(N)
    if dirichlet:
        u[boundary] = 0.0
        v[boundary] = 0.0
    Ku, Kv = D_U * K + M / DT, D_V * K + M / DT
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
            if dirichlet:
                dx = solve(*condense(J, -R, D=both))
            else:
                dx = spsolve(J, -R)
            u_new += dx[:N]
            v_new += dx[N:]
            if np.linalg.norm(dx) < 1e-10:
                break
        u, v = u_new, v_new
    return basis, u, boundary


def diffusion_mass_balance(dirichlet: bool, n_steps: int = 50):
    """Pure diffusion: does the discretisation conserve total mass?"""
    basis = grid()
    K = laplace.assemble(basis)
    M = mass.assemble(basis)
    boundary = basis.get_dofs().flatten()
    xs, ys = basis.doflocs
    u = 1.0 + np.exp(-40.0 * ((xs - 0.5) ** 2 + (ys - 0.5) ** 2))
    if dirichlet:
        u[boundary] = 0.0
    A = M + 1e-3 * K
    ones = np.ones(basis.N)
    m0 = float(ones @ (M @ u))
    for _ in range(n_steps):
        rhs = M @ u
        u = solve(*condense(A, rhs, D=boundary)) if dirichlet else spsolve(A, rhs)
    m1 = float(ones @ (M @ u))
    return (m1 - m0) / m0


def main() -> int:
    ok = True

    print(f"skfem_version={skfem.__version__}")
    print(f"skfem_has_DirichletBC={hasattr(skfem, 'DirichletBC')}")
    print(f"skfem_has_condense={hasattr(skfem, 'condense')} "
          f"skfem_has_enforce={hasattr(skfem, 'enforce')}")
    if hasattr(skfem, "DirichletBC"):
        print("FAIL: skfem grew a DirichletBC symbol; the API note is stale",
              file=sys.stderr)
        ok = False

    # --- the natural BC, pinned on pure diffusion --------------------------
    rel_neumann = diffusion_mass_balance(dirichlet=False)
    rel_dirichlet = diffusion_mass_balance(dirichlet=True)
    print(f"diffusion_mass_rel_change_neumann={rel_neumann:.3e}")
    print(f"diffusion_mass_rel_change_dirichlet={rel_dirichlet:.3e}")
    print(f"neumann_mass_rel_change_lt_1e_12={abs(rel_neumann) < 1e-12}")
    print(f"dirichlet_mass_rel_change_gt_0p5={abs(rel_dirichlet) > 0.5}")
    if not abs(rel_neumann) < 1e-12:
        print("FAIL: the untouched weak form no longer conserves mass, so "
              "zero flux is not natural in it", file=sys.stderr)
        ok = False
    if not abs(rel_dirichlet) > 0.5:
        print("FAIL: condensing the boundary DOFs to zero no longer drains "
              "mass — the two conditions are not being distinguished",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: natural (nothing done on the boundary) -------------
    basis, u_n, boundary = schnakenberg(dirichlet=False)
    spread_n = float(u_n.max() - u_n.min())
    bmax_n = float(np.abs(u_n[boundary]).max())
    print(f"n_dofs={basis.N} n_boundary_dofs={boundary.size}")
    print(f"neumann u=[{u_n.min():.5f}, {u_n.max():.5f}] "
          f"boundary=[{u_n[boundary].min():.5f}, {u_n[boundary].max():.5f}]")
    print(f"neumann_pattern_spread={spread_n:.4f}")
    print(f"neumann_boundary_max_u_gt_1={bmax_n > 1.0}")
    print(f"neumann_pattern_spread_gt_1p0={spread_n > 1.0}")
    if not (bmax_n > 1.0 and spread_n > 1.0):
        print("FAIL: the natural-BC run did not produce a pattern reaching the "
              "boundary, so there is no healthy reference", file=sys.stderr)
        ok = False

    # --- WRONG variant: u = v = 0 imposed on the boundary ------------------
    _b, u_d, bnd = schnakenberg(dirichlet=True)
    spread_d = float(u_d.max() - u_d.min())
    bmax_d = float(np.abs(u_d[bnd]).max())
    print(f"dirichlet u=[{u_d.min():.5f}, {u_d.max():.5f}] "
          f"boundary_max_abs={bmax_d:.3e}")
    print(f"dirichlet_pattern_spread={spread_d:.4f}")
    print(f"dirichlet_boundary_u_exactly_zero={bmax_d == 0.0}")
    print(f"dirichlet_pattern_spread_lt_0p3={spread_d < 0.3}")
    print(f"amplitude_suppression_factor={spread_n / max(spread_d, 1e-300):.1f}")
    print(f"amplitude_suppressed_gt_10x={spread_n / max(spread_d, 1e-300) > 10.0}")
    if not (bmax_d == 0.0 and spread_d < 0.3):
        print("FAIL: the Dirichlet run did not clamp the boundary and suppress "
              "the pattern — the claim's pathology is absent", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
