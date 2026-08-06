"""Tier-2: isolated subdomains leave an O(1) interface temperature jump.

Claim: skfem heat#2 -- conjugate heat transfer couples fluid and solid
subdomains across a shared interface; use skfem.subdomains and a matching Basis
on each. Solving each region in isolation gives interface temperature jumps of
the order of the imposed temperature difference rather than 0.

Wrong variant: assemble and solve each subdomain separately, which leaves the
interface as an insulated (natural) boundary. Each region then runs isothermal
at its own Dirichlet value and the interface jump is the full 100 K.

Right variant: one assembly over both subdomains. The interface DOFs are shared,
so continuity holds by construction and the interface temperature lands between
the two Dirichlet values, pulled towards the high-conductivity side.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import Basis, ElementQuad1, MeshQuad, condense, solve
from skfem.models.poisson import laplace

K_FLUID, K_SOLID = 1.0, 100.0
T_HOT, T_COLD = 100.0, 0.0


def main() -> int:
    ok = True
    m = MeshQuad.init_tensor(
        np.linspace(0.0, 1.0, 17), np.linspace(0.0, 1.0, 5),
    ).with_subdomains({
        "fluid": lambda x: x[0] < 0.5,
        "solid": lambda x: x[0] > 0.5,
    }).with_boundaries({
        "left": lambda x: x[0] < 1e-10,
        "right": lambda x: x[0] > 1.0 - 1e-10,
    })
    print(f"subdomain_sizes={{'fluid': {len(m.subdomains['fluid'])}, "
          f"'solid': {len(m.subdomains['solid'])}}}")

    basis = Basis(m, ElementQuad1())
    left = basis.get_dofs("left").flatten()
    right = basis.get_dofs("right").flatten()
    interface = basis.get_dofs(
        lambda p: np.abs(p[0] - 0.5) < 1e-10).flatten()
    print(f"n_interface_dofs={len(interface)}")

    def sub_stiffness(tag: str, k: float):
        return k * laplace.assemble(
            Basis(m, ElementQuad1(), elements=m.subdomains[tag]))

    # --- RIGHT variant: one monolithic system ---------------------------
    K = sub_stiffness("fluid", K_FLUID) + sub_stiffness("solid", K_SOLID)
    x = basis.zeros()
    x[left] = T_HOT
    x[right] = T_COLD
    u = solve(*condense(K, basis.zeros(), x=x,
                        D=np.concatenate([left, right])))
    iface = u[interface]
    spread = float(iface.max() - iface.min())
    single_valued = spread < 1e-9
    between = bool(T_COLD < iface.mean() < T_HOT)
    print(f"monolithic_interface_is_single_valued={single_valued}")
    print(f"monolithic_interface_between_the_two_bc_values={between}")
    if not (single_valued and between):
        print(f"FAIL: monolithic interface values {iface!r} (spread "
              f"{spread!r})", file=sys.stderr)
        ok = False

    # --- WRONG variant: each region alone, interface insulated ----------
    isolated = {}
    for tag, k, bc_dofs, bc_val in (("fluid", K_FLUID, left, T_HOT),
                                    ("solid", K_SOLID, right, T_COLD)):
        Ki = sub_stiffness(tag, k)
        touched = np.unique(Ki.nonzero()[0])
        untouched = np.setdiff1d(np.arange(basis.N), touched)
        xi = basis.zeros()
        xi[bc_dofs] = bc_val
        ui = solve(*condense(Ki, basis.zeros(), x=xi,
                             D=np.unique(np.concatenate([bc_dofs, untouched]))))
        isolated[tag] = ui[interface]
        print(f"isolated_{tag}_interface_mean={float(ui[interface].mean()):.6f}")

    jump = float(np.abs(isolated["fluid"] - isolated["solid"]).max())
    full = abs(jump - (T_HOT - T_COLD)) < 1e-6
    print(f"isolated_interface_jump_is_full_delta_t={full}")
    ratio_ok = jump / max(spread, 1e-300) > 1e12
    print(f"isolated_jump_over_monolithic_spread_gt_1e12={ratio_ok}")
    if not (full and ratio_ok):
        print(f"FAIL: isolated interface jump {jump!r} is not the full "
              f"{T_HOT - T_COLD!r} K, or the monolithic spread {spread!r} is "
              f"not negligible against it", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
