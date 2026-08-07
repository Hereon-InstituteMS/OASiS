"""Tier-2: on a saddle-point system the Krylov methods WORK — the
signal is cost and accuracy, not failure.

  stokes#1          the direct solver is the right default, but the
                    claim that Krylov methods cannot solve a
                    saddle-point system was refuted: all three converge.
                    solver=('suitesparse','umfpack') takes 1 linear
                    iteration to near machine precision, while cg,
                    gmres and bicgstab take four- to five-figure
                    iteration counts and land orders of magnitude
                    further from the reference. Also: the default
                    preconditioner IS 'none' — passing it explicitly
                    reproduces the default count exactly.
  navier_stokes#5   the same measurement, stated for the Newton tangent:
                    read linear_iterations out of the dict solve()
                    returns and judge the solver by that, not by whether
                    it finishes.

Taylor-Hood Poiseuille flow on a small grid. The iteration counts are
asserted by ORDER OF MAGNITUDE, never by digits: the catalog itself
records that an earlier revision's six figures did not reproduce.

Verified by execution against dune-fem 2.12.0.2.

MUTATION CONTROL. T2_MUTATE=1 runs the three Krylov labels with the
direct UMFPACK solver — the pathology (an unpreconditioned Krylov
method on a saddle-point system) removed. Every count drops to 1, so
'krylov_costs_orders_of_magnitude_more=True' is no longer printed and
FAIL: lines appear. The direct scheme is one the fixture already
builds.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dune.grid import structuredGrid                            # noqa: E402
from dune.fem.space import lagrange, composite                  # noqa: E402
from dune.fem.scheme import galerkin                            # noqa: E402
from dune.ufl import Constant, DirichletBC                      # noqa: E402
from ufl import (TrialFunction, TestFunction,                    # noqa: E402
                 SpatialCoordinate, as_vector, div, grad, inner,
                 dx, conditional, lt, gt, Or)

N = 8
TOL = 1e-8


def main() -> int:
    fail: list[str] = []
    gridView = structuredGrid([0, 0], [1, 1], [N, N])
    W = composite(lagrange(gridView, dimRange=2, order=2),
                  lagrange(gridView, order=1))
    t, s = TrialFunction(W), TestFunction(W)
    u, p = as_vector([t[0], t[1]]), t[2]
    v, q = as_vector([s[0], s[1]]), s[2]
    x = SpatialCoordinate(W)

    a = (inner(grad(u), grad(v)) - p * div(v) - q * div(u)) * dx
    L = Constant(0.0, name="zero") * q * dx

    inflow_profile = x[1] * (1 - x[1])
    walls = Or(lt(x[1], TOL), gt(x[1], 1 - TOL))
    inlet = lt(x[0], TOL)
    bcs = [DirichletBC(W, [inflow_profile, 0, None], inlet),
           DirichletBC(W, [0, 0, None], walls)]

    print(f"taylor_hood_dofs={W.size}")
    print(f"taylor_hood_dimRange={W.dimRange}")

    if MUTATE:
        print("mutation=every_krylov_label_uses_the_direct_solver")
    results = {}
    for label, solver in (("umfpack", ("suitesparse", "umfpack")),
                          ("cg", "cg"),
                          ("gmres", "gmres"),
                          ("bicgstab", "bicgstab")):
        if MUTATE:
            solver = ("suitesparse", "umfpack")
        scheme = galerkin([a == L] + bcs, solver=solver,
                          parameters={"linear.tolerance": 1e-12,
                                      "linear.maxiterations": 200000})
        wh = W.interpolate([0, 0, 0], name=f"wh_{label}")
        info = scheme.solve(target=wh)
        vals = np.array(wh.as_numpy)
        results[label] = (bool(info["converged"]),
                          int(info["linear_iterations"]), vals)
        print(f"{label}_converged={bool(info['converged'])}")
        print(f"{label}_linear_iterations="
              f"{int(info['linear_iterations'])}")

    ref = results["umfpack"][2]
    ref_norm = float(np.linalg.norm(ref))
    for label in ("cg", "gmres", "bicgstab"):
        conv, iters, vals = results[label]
        rel = float(np.linalg.norm(vals - ref)) / ref_norm
        print(f"{label}_relative_to_direct={rel:.3e}")
        if not conv:
            fail.append(f"{label} did NOT converge; the catalog "
                        f"explicitly retracted the 'Krylov cannot solve "
                        f"it' wording, so a failure here would mean the "
                        f"retraction is wrong")
        if iters < 100:
            fail.append(f"{label} converged in {iters} linear "
                        f"iterations; the claim is that the Krylov "
                        f"methods need three or more orders of "
                        f"magnitude more work than the direct solve")

    direct_conv, direct_iters, _ = results["umfpack"]
    print(f"direct_linear_iterations={direct_iters}")
    print(f"direct_takes_one_iteration={direct_iters == 1}")
    worst = max(results[k][1] for k in ("cg", "gmres", "bicgstab"))
    print(f"worst_krylov_iterations={worst}")
    print(f"krylov_costs_orders_of_magnitude_more="
          f"{worst > 1000 * max(direct_iters, 1)}")
    print(f"all_krylov_methods_do_converge="
          f"{all(results[k][0] for k in ('cg', 'gmres', 'bicgstab'))}")
    if direct_iters != 1:
        fail.append(f"the direct solve reported {direct_iters} linear "
                    f"iterations, not 1")
    if not worst > 1000 * max(direct_iters, 1):
        fail.append(f"the worst Krylov count was {worst} against "
                    f"{direct_iters} for the direct solve; the claim is "
                    f"a gap of three to five orders of magnitude")

    # ── the default preconditioner IS 'none' ───────────────────────
    counts = {}
    for pc in ("none", "ssor", "jacobi"):
        scheme = galerkin([a == L] + bcs, solver="gmres",
                          parameters={"linear.tolerance": 1e-12,
                                      "linear.maxiterations": 200000,
                                      "linear.preconditioning.method": pc})
        wh = W.interpolate([0, 0, 0], name=f"pc_{pc}")
        info = scheme.solve(target=wh)
        counts[pc] = int(info["linear_iterations"])
        print(f"gmres_with_{pc}_iterations={counts[pc]}")
    default_count = results["gmres"][1]
    print(f"gmres_default_iterations={default_count}")
    print(f"default_preconditioner_is_none="
          f"{counts['none'] == default_count}")
    print(f"a_preconditioner_reduces_the_count="
          f"{min(counts['ssor'], counts['jacobi']) < default_count}")
    if counts["none"] != default_count:
        fail.append(f"explicitly asking for 'none' gave "
                    f"{counts['none']} iterations against the default "
                    f"{default_count}; the claim is that they are "
                    f"EXACTLY equal, which is how the default was "
                    f"identified")

    if not fail:
        print("dune_saddle_point_solver_cost_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
