"""Tier-2: high cell Peclet makes plain CG undershoot; SUPG does not.

Claim: skfem convection_diffusion#3 -- standard CG at Pe > ~10 develops
oscillations that do not damp under refinement; SUPG or a DG element removes
them.

The test problem is an outflow boundary layer: b = (1, 0), eps = 1e-3, u = 0 at
the inlet and u = 1 at the outlet, so the exact solution is monotone in [0, 1]
and ANY negative value is a discretisation artefact rather than physics.

Wrong variant: the plain Galerkin form, swept over four meshes.
Right variant: the same form plus the standard SUPG term with
tau = (h/2|b|)(coth(Pe_h) - 1/Pe_h).

Only booleans and signs are asserted; no undershoot magnitude is pinned.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementQuad1,
    MeshQuad,
    condense,
    solve,
)
from skfem.helpers import dot, grad

EPS = 1e-3
BX, BY = 1.0, 0.0
LEVELS = (10, 20, 40, 80)


def solve_layer(nx: int, supg: bool):
    m = MeshQuad.init_tensor(
        np.linspace(0.0, 1.0, nx + 1), np.linspace(0.0, 1.0, 3),
    ).with_boundaries({
        "inlet": lambda x: x[0] < 1e-10,
        "outlet": lambda x: x[0] > 1.0 - 1e-10,
    })
    basis = Basis(m, ElementQuad1())
    h = 1.0 / nx
    speed = float(np.hypot(BX, BY))
    peclet = speed * h / (2.0 * EPS)
    tau = 0.0
    if supg:
        tau = (h / (2.0 * speed)) * (1.0 / np.tanh(peclet) - 1.0 / peclet)

    @BilinearForm
    def form(u, v, w):
        conv = BX * u.grad[0] + BY * u.grad[1]
        out = EPS * dot(grad(u), grad(v)) + conv * v
        if tau > 0.0:
            out = out + tau * conv * (BX * v.grad[0] + BY * v.grad[1])
        return out

    x = basis.zeros()
    inlet = basis.get_dofs("inlet").flatten()
    outlet = basis.get_dofs("outlet").flatten()
    x[outlet] = 1.0
    u = solve(*condense(form.assemble(basis), basis.zeros(), x=x,
                        D=np.concatenate([inlet, outlet])))
    return peclet, float(u.min())


def main() -> int:
    ok = True
    cg, supg, peclets = [], [], []
    for nx in LEVELS:
        pe, umin_cg = solve_layer(nx, supg=False)
        _, umin_supg = solve_layer(nx, supg=True)
        peclets.append(pe)
        cg.append(umin_cg)
        supg.append(umin_supg)
        print(f"nx={nx:3d} cell_peclet={pe:8.2f} cg_min={umin_cg:+.5f} "
              f"supg_min={umin_supg:+.5f}")

    tol = -1e-6
    cg_bad = all(v < tol for v in cg)
    supg_good = all(v >= tol for v in supg)
    print(f"cg_undershoots_at_every_level={cg_bad}")
    print(f"supg_clean_at_every_level={supg_good}")
    print(f"cg_undershoot_survives_to_finest={cg[-1] < tol}")
    print(f"supg_fixes_coarsest_mesh={cg[0] < tol <= supg[0]}")
    print(f"cell_peclet_above_one_at_every_level="
          f"{all(p > 1.0 for p in peclets)}")
    print(f"cg_undershoot_shrinks_with_refinement={cg[-1] > cg[0]}")

    if not cg_bad:
        print(f"FAIL: plain CG stayed monotone somewhere in the sweep: {cg}",
              file=sys.stderr)
        ok = False
    if not supg_good:
        print(f"FAIL: SUPG undershot somewhere in the sweep: {supg}",
              file=sys.stderr)
        ok = False
    if not all(p > 1.0 for p in peclets):
        print(f"FAIL: the sweep dropped below cell Peclet 1: {peclets}",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
