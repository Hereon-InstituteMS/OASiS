"""Tier-2: a DUNE-fem DirichletBC that is not in the scheme's list is
silently ignored, and scheme.solve() still reports converged=True.

This is the most expensive DUNE-fem trap found by execution, and it is
a direct FEniCS-transfer error: in dolfinx you build the BC object and
hand it to the solver/assembler separately, so building a DirichletBC
and then constructing the scheme without it looks natural. In dune-fem
the BC only exists if it is an element of the list passed to
``galerkin([...])`` — dune/fem/scheme/_schemes.py decides at scheme-
construction time whether to wrap the operator in a DirichletWrapper
(``integrands.hasDirichletBoundary``). Nothing warns.

What the run then solves is the PURE NEUMANN problem, which is singular
(the solution is defined only up to a constant, and the data are not
compatible). CG does not error out: it grinds through tens of thousands
of iterations, returns, and the info dict says ``converged: True``.

The fixture runs three variants of the SAME weak form and prints the
observables a caller could actually check:

  good     galerkin([a == b, dbc])                    -> correct
  omitted  galerkin(a == b, space=space)              -> silent garbage
  never    galerkin([a == b, DirichletBC(space, 0,
                     conditional(x[0] < -1.0, 1, 0))]) -> silent garbage

The third variant is the same failure reached a different way: the BC
IS in the list, but its subdomain conditional is false everywhere on
[0,1]^2, and nothing checks that a subdomain selects any facet.

Verified by execution against dune-fem 2.12.0.2 on 2026-08-03:
  good    converged=True  linear_iterations=1      L2 err 1.899705e-03
  omitted converged=True  linear_iterations=23935  L2 err 7.510512e+14
  never   converged=True  linear_iterations=23935  L2 err 7.510512e+14
"""
from __future__ import annotations

import math
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np                                        # noqa: E402
from dune.grid import structuredGrid                      # noqa: E402
from dune.fem import integrate                            # noqa: E402
from dune.fem.space import lagrange                       # noqa: E402
from dune.fem.scheme import galerkin                      # noqa: E402
from dune.ufl import DirichletBC                          # noqa: E402
from ufl import (TrialFunction, TestFunction,             # noqa: E402
                 SpatialCoordinate, conditional, dot, grad, dx, sin, pi)

N = 16


def main() -> int:
    gridView = structuredGrid([0, 0], [1, 1], [N, N])
    space = lagrange(gridView, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    x = SpatialCoordinate(space)
    u_ex = sin(pi * x[0]) * sin(pi * x[1])
    a = dot(grad(u), grad(v)) * dx
    b = 2 * pi ** 2 * u_ex * v * dx

    def run(label, scheme):
        uh = space.interpolate(0, name="uh")
        info = scheme.solve(target=uh)
        err = math.sqrt(integrate((uh - u_ex) ** 2,
                                  gridView=gridView, order=6))
        peak = float(np.abs(np.array(uh.as_numpy)).max())
        print(f"{label}_converged={bool(info['converged'])} "
              f"{label}_liniter={int(info['linear_iterations'])} "
              f"{label}_l2err={err:.6e} {label}_peak={peak:.6e}")
        return bool(info["converged"]), int(info["linear_iterations"]), err

    good_conv, good_it, good_err = run(
        "good", galerkin([a == b, DirichletBC(space, 0)], solver="cg"))
    om_conv, om_it, om_err = run(
        "omitted", galerkin(a == b, space=space, solver="cg"))
    nv_conv, nv_it, nv_err = run(
        "never",
        galerkin([a == b,
                  DirichletBC(space, 0, conditional(x[0] < -1.0, 1, 0))],
                 solver="cg"))

    fail = []
    # The correct pattern must actually be correct.
    if not good_conv or good_err > 3e-3:
        fail.append(f"good variant broken: converged={good_conv} "
                    f"l2err={good_err:.3e} (expected <= 3e-3)")
    # Both broken variants must reproduce the trap: reported converged,
    # yet catastrophically wrong.
    for label, conv, it, err in (("omitted", om_conv, om_it, om_err),
                                 ("never", nv_conv, nv_it, nv_err)):
        if not conv:
            fail.append(f"{label}: expected the trap (converged=True "
                        f"despite a singular system) but got "
                        f"converged=False — dune-fem now detects it")
        if err < 1e3:
            fail.append(f"{label}: expected a catastrophic L2 error "
                        f"but got {err:.3e} — the BC appears to be "
                        f"applied after all")
        if it <= 100 * good_it + 100:
            fail.append(f"{label}: expected an iteration blow-up but "
                        f"got {it} vs good {good_it}")

    if not fail:
        print("dune_silent_bc_trap_reproduced=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
