"""Tier-2 Layer-C: DUNE-fem Poisson numerical-correctness gate.

Mirrors fenics / skfem / ngsolve poisson_mms_convergence. Runs the
catalog-recommended DUNE-fem API surface end-to-end against a
manufactured solution and asserts the MEASURED errors and orders,
not merely that the script runs.

DUNE-fem catalog API surface exercised here
(src/backends/dune/generators/poisson.py + verified_api.py):
  * structuredGrid([0,0], [1,1], [n,n])          -> cube cells
  * lagrange(gridView, order=k)
  * TrialFunction / TestFunction / SpatialCoordinate on the SPACE
  * galerkin([a == b, DirichletBC(space, u_exact)], solver="cg")
  * scheme.solve(target=uh) -> info dict with 'converged'
  * dune.fem.integrate(expr, gridView=..., order=...) for the norms

MMS:
  u*(x, y) = sin(pi x) sin(pi y) + 0.3 x^2 y   on [0, 1]^2
  f        = -div(grad(u*))                    (built symbolically by
                                                UFL — nothing to
                                                transcribe by hand)
  u = u* on the whole boundary (exact Dirichlet data)

Expected behaviour for Lagrange order k under uniform refinement:
  L2 error   = O(h^(k+1))
  H1 seminorm= O(h^k)

Measured on dune-fem 2.12.0.2 (conda env dune-fem-env, 2026-08-03),
levels n = 4, 8, 16, 32:

  k=1: L2 EOC 1.989 / 1.997 / 1.999   H1 EOC 0.995 / 0.999 / 1.000
  k=2: L2 EOC 2.979 / 2.995 / 2.999   H1 EOC 1.998 / 2.000 / 2.000
  k=3: L2 EOC 3.985 / 3.996 / 3.999   H1 EOC 2.996 / 2.999 / 3.000

  L2 at n=32: k=1 4.556318e-04, k=2 3.846536e-06, k=3 2.180413e-08

The gate below re-measures k=1 and k=2 on n = 8, 16, 32 (k=3 is
dropped only to keep the fixture cheap) and fails if either the
absolute error floor or the observed order band is missed. The bands
are deliberately tight — this is a Cartesian grid, so unlike the
Netgen-meshed ngsolve fixture there is no mesh-topology jitter and the
orders land on theory to three digits.

MUTATION CONTROL. This fixture is a correctness GATE, not a pitfall
reproduction, so the control has to show the gate can fail rather than
that a pathology can be removed. T2_MUTATE=1 loosens the linear
tolerance from 1e-12 to 1e-2, which is exactly the defect the gate
exists to catch: the algebraic error then swamps the discretisation
error, the observed orders leave their bands, and 'dune_poisson_mms_
gate=OK' is no longer printed while FAIL: lines appear. Only a solver
parameter changes, so no new module is compiled.
"""
from __future__ import annotations

import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dune.grid import structuredGrid                       # noqa: E402
from dune.fem import integrate                             # noqa: E402
from dune.fem.space import lagrange                        # noqa: E402
from dune.fem.scheme import galerkin                       # noqa: E402
from dune.ufl import DirichletBC                           # noqa: E402
from ufl import (TrialFunction, TestFunction,              # noqa: E402
                 SpatialCoordinate, dot, grad, div, dx, sin, pi)

# Absolute L2 floors at the finest level (n = 32), taken from the
# 2026-08-03 measurement with 30% headroom.
L2_FLOOR = {1: 6.0e-04, 2: 5.0e-06}
# Observed-order bands. Cartesian refinement -> no topology jitter.
EOC_BAND = {1: (1.90, 2.10), 2: (2.90, 3.10)}
H1_BAND = {1: (0.90, 1.10), 2: (1.90, 2.10)}
LEVELS = [8, 16, 32]


def solve_level(n: int, order: int) -> tuple[float, float, bool]:
    gridView = structuredGrid([0, 0], [1, 1], [n, n])
    space = lagrange(gridView, order=order)
    u, v = TrialFunction(space), TestFunction(space)
    x = SpatialCoordinate(space)
    u_ex = sin(pi * x[0]) * sin(pi * x[1]) + 0.3 * x[0] ** 2 * x[1]
    f = -div(grad(u_ex))
    a = dot(grad(u), grad(v)) * dx
    b = f * v * dx
    scheme = galerkin([a == b, DirichletBC(space, u_ex)], solver="cg",
                      parameters={"linear.tolerance":
                                  1e-2 if MUTATE else 1e-12,
                                  "linear.maxiterations": 20000})
    uh = space.interpolate(0, name="uh")
    info = scheme.solve(target=uh)
    q = 2 * order + 4
    e_l2 = math.sqrt(integrate((uh - u_ex) ** 2, gridView=gridView, order=q))
    e_h1 = math.sqrt(integrate(dot(grad(uh - u_ex), grad(uh - u_ex)),
                               gridView=gridView, order=q))
    return e_l2, e_h1, bool(info["converged"])


def main() -> int:
    import dune.fem  # noqa: F401  (import proves the env is usable)
    try:
        from importlib.metadata import version
        print(f"dune_fem_version={version('dune-fem')}")
    except Exception:                                    # pragma: no cover
        print("dune_fem_version=unknown")

    if MUTATE:
        print("mutation=the_linear_tolerance_is_loosened_to_1e-2")
    fail: list[str] = []
    for k in sorted(L2_FLOOR):
        l2, h1 = [], []
        for n in LEVELS:
            e_l2, e_h1, conv = solve_level(n, k)
            l2.append(e_l2)
            h1.append(e_h1)
            print(f"P{k}_n{n}_l2err={e_l2:.6e} h1err={e_h1:.6e} "
                  f"converged={conv}")
            if not conv:
                fail.append(f"P{k} n={n} solver reported not converged")
        print(f"P{k}_finest_l2err={l2[-1]:.6e}_floor={L2_FLOOR[k]:.1e}")
        if l2[-1] > L2_FLOOR[k]:
            fail.append(f"P{k} finest L2 {l2[-1]:.3e} > "
                        f"{L2_FLOOR[k]:.1e}")
        for i in range(1, len(LEVELS)):
            eoc_l2 = math.log(l2[i - 1] / l2[i]) / math.log(2.0)
            eoc_h1 = math.log(h1[i - 1] / h1[i]) / math.log(2.0)
            print(f"P{k}_eoc_n{LEVELS[i-1]}_to_n{LEVELS[i]}="
                  f"{eoc_l2:.3f}_expected={k+1} "
                  f"h1={eoc_h1:.3f}_expected={k}")
            lo, hi = EOC_BAND[k]
            if not (lo <= eoc_l2 <= hi):
                fail.append(f"P{k} L2 EOC {eoc_l2:.3f} outside "
                            f"[{lo}, {hi}]")
            lo, hi = H1_BAND[k]
            if not (lo <= eoc_h1 <= hi):
                fail.append(f"P{k} H1 EOC {eoc_h1:.3f} outside "
                            f"[{lo}, {hi}]")

    if not fail:
        print("dune_poisson_mms_gate=OK")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
