"""Tier-2 numerical-comparison fixture: forward-Euler CFL violation.

Solves the 1-D heat equation ∂u/∂t = α ∂²u/∂x² on [0, 1] with
Dirichlet u=0 at both endpoints and IC u(x) = sin(πx), using
forward Euler time-stepping. Two runs:

  * dt_stable  = 0.4 * h²/(2α)  — well below the CFL bound.
    Solution decays toward 0 monotonically; final |u|_max < 1.
  * dt_unstable = 5.0 * h²/(2α) — far above the CFL bound.
    Forward Euler amplification factor exceeds 1; the
    solution grows exponentially. After 20 steps, |u|_max is
    O(10²–10³), > 100× the initial bump.

A post-execution critic recognising the catalog Signal
('forward Euler with too-large dt produces oscillating
solution that grows exponentially') matches against the
numerical observation, not against an exception string —
this is the first Tier-2 fixture that verifies a
*numerical-comparison* Signal rather than an exception class.
"""
from __future__ import annotations

import sys

import numpy as np

from skfem import (
    Basis,
    BilinearForm,
    ElementLineP1,
    MeshLine,
)
from skfem.helpers import dot, grad


@BilinearForm
def stiffness(u, v, w):
    return dot(grad(u), grad(v))


@BilinearForm
def mass(u, v, w):
    return u * v


def _evolve(K, M_lump, u, dt: float, alpha: float,
            n_steps: int, interior) -> float:
    """Run n_steps of forward Euler and return |u|_inf."""
    for _ in range(n_steps):
        rhs = -alpha * (K @ u)
        u[interior] = u[interior] + dt * rhs[interior] / M_lump[interior]
    return float(np.abs(u).max())


def main() -> int:
    n = 50
    mesh = MeshLine(np.linspace(0, 1, n))
    ib = Basis(mesh, ElementLineP1())
    K = stiffness.assemble(ib)
    M = mass.assemble(ib)
    M_lump = np.array(M.sum(axis=1)).flatten()

    x = mesh.p.flatten()
    u0 = np.sin(np.pi * x)
    u0[0] = u0[-1] = 0.0  # enforce Dirichlet

    alpha = 1.0
    h = 1.0 / (n - 1)
    dt_cfl = h * h / (2.0 * alpha)

    D = ib.get_dofs().flatten()
    interior = sorted(set(range(ib.N)) - set(D))

    u_stable = _evolve(K, M_lump, u0.copy(),
                       0.4 * dt_cfl, alpha, 20, interior)
    u_unstable = _evolve(K, M_lump, u0.copy(),
                         5.0 * dt_cfl, alpha, 20, interior)

    print(f"forward-Euler heat eq, 20 steps, h = {h:.4f}")
    print(f"  dt = 0.4 * CFL  → |u|_max = {u_stable:.3e}")
    print(f"  dt = 5.0 * CFL  → |u|_max = {u_unstable:.3e}")

    if u_unstable < 10.0:
        print("FIXTURE WARNING: unstable run did not blow up — "
              "CFL stability bound no longer reproducible",
              file=sys.stderr)
        return 2

    ratio = u_unstable / max(u_stable, 1e-12)
    print(f"  unstable / stable ratio = {ratio:.1e}",
          file=sys.stderr)
    print(
        f"CFL violated: dt={5.0*dt_cfl:.4e} exceeds bound "
        f"{dt_cfl:.4e}; forward-Euler is unstable here, "
        f"linfty_norm grows by factor {ratio:.1e}.",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
