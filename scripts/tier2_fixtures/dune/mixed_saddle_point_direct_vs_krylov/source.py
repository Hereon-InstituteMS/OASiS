"""Tier-2: on a mixed (saddle-point) system an unpreconditioned Krylov
solver is not a drop-in replacement for the direct one.

The claim (dune.mixed_methods #6) is that plain GMRES on the raw
indefinite matrix converges, slowly, and to a much looser tolerance
than a direct factorisation, and that UMFPACK is the right default at
this size. Measured here, all three parts hold, and two things the
claim does not mention turn out to matter more than the tolerance gap:

  * the GMRES iteration count grows far faster than the problem, so
    "slowly" is a structural statement about the count, not a wall
    clock;
  * two other stock Krylov choices do not merely converge slowly.
    BiCGStab exhausts its cap and hands back the zero vector; CG
    reports converged=True and hands back NaN. An indefinite matrix
    is outside CG's contract, and nothing in the returned info says
    so.

The flux space is Lagrange, not Raviart-Thomas, because
mixed_methods #0 records that raviartThomas cannot be a leg of
composite() on this dune-fem — the claim's own advice.

Cost is asserted through iteration counts and residual norms only.
Nothing here is timed: a wall-clock assertion goes red when the machine
is busy and says nothing about the solver.
"""
from __future__ import annotations

import math
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                            # noqa: E402
from dune.fem.space import lagrange, composite                  # noqa: E402
from dune.fem.scheme import galerkin                            # noqa: E402
from dune.ufl import DirichletBC                                # noqa: E402
from ufl import (TrialFunction, TestFunction, SpatialCoordinate,  # noqa: E402
                 as_vector, div, inner, dx, sin, pi)

PARAMS = {"linear.tolerance": 1e-12, "linear.maxiterations": 20000,
          "linear.verbose": False}
LEVELS = (8, 16)


def mixed_pieces(n):
    """sigma = -grad u, -div sigma = f, as one indefinite system."""
    gridView = structuredGrid([0, 0], [1, 1], [n, n])
    flux = lagrange(gridView, dimRange=2, order=2)
    potential = lagrange(gridView, order=1)
    W = composite(flux, potential)
    t, s = TrialFunction(W), TestFunction(W)
    sig, u = as_vector([t[0], t[1]]), t[2]
    tau, v = as_vector([s[0], s[1]]), s[2]
    x = SpatialCoordinate(W)
    u_exact = sin(pi * x[0]) * sin(pi * x[1])
    f = 2 * pi * pi * u_exact
    a = (inner(sig, tau) + u * div(tau) + v * div(sig)) * dx
    L = (-f) * v * dx
    bc = DirichletBC(W, [None, None, 0], None)
    return gridView, W, a, L, bc


def solve_with(n, solver, tag):
    gridView, W, a, L, bc = mixed_pieces(n)
    scheme = galerkin([a == L, bc], solver=solver, parameters=PARAMS)
    wh = W.interpolate([0, 0, 0], name=f"{tag}{n}")
    info = scheme.solve(target=wh)
    residual = wh.copy()
    scheme(wh, residual)
    res = float(np.abs(np.array(residual.as_numpy)).max())
    values = np.array(wh.as_numpy)
    return {
        "converged": bool(info.get("converged")),
        "iterations": int(info.get("linear_iterations", -1)),
        "residual": res,
        "values": values,
        "dofs": W.size,
    }


def main() -> int:
    fail: list[str] = []
    direct, krylov = {}, {}

    for n in LEVELS:
        direct[n] = solve_with(n, ("suitesparse", "umfpack"), "direct")
        krylov[n] = solve_with(n, "gmres", "gmres")
        print(f"n{n}_dofs={direct[n]['dofs']}")
        print(f"n{n}_direct_converged={direct[n]['converged']}")
        print(f"n{n}_direct_residual={direct[n]['residual']:.6e}")
        print(f"n{n}_gmres_converged={krylov[n]['converged']}")
        print(f"n{n}_gmres_linear_iterations={krylov[n]['iterations']}")
        print(f"n{n}_gmres_residual={krylov[n]['residual']:.6e}")

    # 1. Both solve the same system and land on the same answer — the
    #    Krylov solve is looser, not wrong.
    same = True
    for n in LEVELS:
        scale = float(np.abs(direct[n]["values"]).max())
        gap = float(np.abs(direct[n]["values"]
                           - krylov[n]["values"]).max())
        print(f"n{n}_direct_vs_gmres_max_difference={gap:.6e}")
        print(f"n{n}_solution_scale={scale:.6e}")
        same = same and gap < 1e-6 * scale
    print(f"gmres_reaches_the_same_solution={same}")
    if not same:
        fail.append("GMRES did not reach the direct solver's answer; the "
                    "claim is about tolerance and cost, not correctness, "
                    "so a wrong answer would mean this fixture is "
                    "measuring something else")

    # 2. The direct residual is orders of magnitude tighter.
    gaps = []
    for n in LEVELS:
        gaps.append(krylov[n]["residual"] / direct[n]["residual"])
        print(f"n{n}_gmres_over_direct_residual={gaps[-1]:.6e}")
    tighter = all(g > 1e2 for g in gaps)
    print(f"direct_residual_is_orders_tighter={tighter}")
    if not tighter:
        fail.append(f"the direct residual was not orders of magnitude "
                    f"tighter than the GMRES one (ratios {gaps}); that "
                    f"gap is the claim")

    # 3. "Slowly" as a structure, not a clock: the iteration count grows
    #    faster than the problem does.
    coarse, fine = LEVELS[0], LEVELS[-1]
    iter_growth = (krylov[fine]["iterations"]
                   / max(krylov[coarse]["iterations"], 1))
    dof_growth = direct[fine]["dofs"] / direct[coarse]["dofs"]
    print(f"gmres_iteration_growth={iter_growth:.4f}")
    print(f"dof_growth={dof_growth:.4f}")
    superlinear = iter_growth > dof_growth
    print(f"gmres_iterations_grow_faster_than_the_problem={superlinear}")
    if not superlinear:
        fail.append(f"the GMRES iteration count grew by {iter_growth:.2f} "
                    f"against a dof growth of {dof_growth:.2f}; without "
                    f"super-linear growth an unpreconditioned Krylov "
                    f"solver would be a reasonable default and the claim "
                    f"would not hold")

    # 4. Two other stock Krylov choices on the same matrix. One fails
    #    loudly, one reports success and returns NaN.
    n = LEVELS[0]
    quiet_failures = {}
    for name in ("bicgstab", "cg"):
        try:
            out = solve_with(n, name, name)
        except Exception as exc:                              # noqa: BLE001
            print(f"{name}_raised={type(exc).__name__}")
            quiet_failures[name] = ("raised", False)
            continue
        values = out["values"]
        has_nan = bool(np.isnan(values).any())
        all_zero = bool(np.abs(values).max() == 0.0)
        print(f"{name}_converged={out['converged']}")
        print(f"{name}_residual={out['residual']:.6e}")
        print(f"{name}_returned_nan={has_nan}")
        print(f"{name}_returned_zero_vector={all_zero}")
        usable = out["converged"] and not has_nan and not all_zero
        quiet_failures[name] = (out["converged"], usable)

    cg_reported, cg_usable = quiet_failures.get("cg", (False, False))
    cg_lies = cg_reported and not cg_usable
    print(f"cg_reports_success_on_an_indefinite_matrix={cg_reported}")
    print(f"cg_result_is_unusable={not cg_usable}")
    print(f"cg_failure_is_silent={cg_lies}")
    if not cg_lies:
        fail.append("CG on the indefinite mixed matrix did not report "
                    "convergence while returning an unusable field; that "
                    "silent failure is the strongest reason the claim's "
                    "'use a direct solver' is not merely a performance "
                    "preference")

    _, bicg_usable = quiet_failures.get("bicgstab", (False, False))
    print(f"bicgstab_result_is_unusable={not bicg_usable}")
    if bicg_usable:
        fail.append("BiCGStab produced a usable answer on the raw "
                    "indefinite matrix; this fixture records that it "
                    "does not")

    print(f"saddle_point_needs_direct_or_preconditioned="
          f"{tighter and superlinear and cg_lies}")

    if not fail:
        print("dune_mixed_saddle_point_solver_gate=OK")
        return 0
    for reason in fail:
        print(f"FAIL: {reason}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
