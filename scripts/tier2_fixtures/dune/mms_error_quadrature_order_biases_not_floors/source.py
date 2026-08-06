"""Tier-2: what an under-integrated error norm actually does to a
convergence study — and what it does not do.

The claim (dune.poisson_mms #2) predicts that an error norm integrated
below 2*order shows "EOCs that DRIFT AWAY from k+1 as levels refine
(error floor)", and marks itself "theory-derived; not provoked live on
this install". Provoked live, the prediction is wrong in the part that
matters and right in a way nobody would guard against.

The SAME discrete solution uh is measured at several quadrature orders,
so nothing but the error norm changes between the columns.

  * There is no error floor at any order tried. Every column keeps
    converging, decade after decade, to the finest level. A gate that
    watched for a stalling error would never fire.
  * The order does drift away from k+1 — UPWARD. A rule of degree 0 or
    1 samples the error where it superconverges, so the reported order
    is k+2, which reads as "better than theory", not as a defect.
  * The reported error is BIASED, and the bias survives refinement.
    A degree-2 rule over-reports by a fixed fraction at every level. A
    degree-0 rule under-reports, and the under-report gets WORSE as
    the mesh refines.

The optimistic direction is the dangerous one: an error norm that is
too small by two orders of magnitude, with an order that looks better
than theory, is exactly what a convergence gate is supposed to catch
and exactly what it would wave through.
"""
from __future__ import annotations

import math
import sys
import warnings

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                            # noqa: E402
from dune.fem import integrate                                  # noqa: E402
from dune.fem.space import lagrange                             # noqa: E402
from dune.fem.scheme import galerkin                            # noqa: E402
from dune.ufl import DirichletBC                                # noqa: E402
from ufl import (TrialFunction, TestFunction, SpatialCoordinate,  # noqa: E402
                 dot, grad, div, dx, sin, pi)

ORDER = 2                       # Lagrange degree; theory gives L2 order 3
LEVELS = (4, 8, 16, 32)
GOOD = 2 * ORDER + 4            # what the catalog template uses
QUADS = (0, 2, GOOD)


def solve_and_measure(n):
    """One solve, several error norms — only the quadrature changes."""
    gridView = structuredGrid([0, 0], [1, 1], [n, n])
    space = lagrange(gridView, order=ORDER)
    u, v = TrialFunction(space), TestFunction(space)
    x = SpatialCoordinate(space)
    u_exact = sin(pi * x[0]) * sin(pi * x[1]) + 0.3 * x[0] ** 2 * x[1]
    f = -div(grad(u_exact))
    scheme = galerkin(
        [dot(grad(u), grad(v)) * dx == f * v * dx,
         DirichletBC(space, u_exact)],
        solver="cg",
        parameters={"linear.tolerance": 1e-12,
                    "linear.maxiterations": 20000})
    uh = space.interpolate(0, name=f"uh{n}")
    info = scheme.solve(target=uh)
    norms = {q: math.sqrt(abs(integrate((uh - u_exact) ** 2,
                                        gridView=gridView, order=q)))
             for q in QUADS}
    return norms, bool(info["converged"])


def eocs(series):
    return [math.log(a / b) / math.log(2.0)
            for a, b in zip(series, series[1:])]


def main() -> int:
    fail: list[str] = []
    table = {}
    for n in LEVELS:
        norms, converged = solve_and_measure(n)
        table[n] = norms
        print(f"n{n}_converged={converged}")
        print("n%d_errors=%s" % (n, " ".join(
            f"q{q}:{norms[q]:.6e}" for q in QUADS)))
        if not converged:
            fail.append(f"the solve at n={n} did not converge, so the "
                        f"error columns are not comparable")

    order = {q: eocs([table[n][q] for n in LEVELS]) for q in QUADS}
    for q in QUADS:
        print(f"q{q}_eocs={','.join(f'{e:.4f}' for e in order[q])}")

    # 1. Control: the well-integrated norm sits on theory.
    good_ok = all(abs(e - (ORDER + 1)) < 0.15 for e in order[GOOD])
    print(f"well_integrated_norm_is_on_theory={good_ok}")
    if not good_ok:
        fail.append(f"the q={GOOD} norm did not give order {ORDER + 1} "
                    f"({order[GOOD]}); without a trustworthy control the "
                    f"other columns mean nothing")

    # 2. NO error floor anywhere — the claim's prediction, falsified.
    #    A floor shows up as an order collapsing toward zero at the
    #    finest levels; every column instead keeps its order to the end.
    floors = {q: order[q][-1] < order[GOOD][-1] - 0.5 for q in QUADS}
    no_floor = not any(floors.values())
    for q in QUADS:
        print(f"q{q}_final_eoc={order[q][-1]:.4f}_first_eoc="
              f"{order[q][0]:.4f}")
    print(f"no_error_floor_at_any_quadrature_order={no_floor}")
    if not no_floor:
        fail.append(f"an error floor DID appear ({floors}); this fixture "
                    f"records that the claim's predicted floor does not "
                    f"happen, so a floor would mean the record is wrong")

    # 3. The drift is real but UPWARD: the lowest rule reports better
    #    than theory, not worse.
    lowest = QUADS[0]
    upward = order[lowest][-1] > (ORDER + 1) + 0.5
    print(f"lowest_order_norm_reports_better_than_theory={upward}")
    print(f"lowest_order_norm_final_eoc={order[lowest][-1]:.4f}"
          f"_theory={ORDER + 1}")
    if not upward:
        fail.append(f"the q={lowest} norm did not report an order above "
                    f"{ORDER + 1} ({order[lowest][-1]:.3f}); the drift "
                    f"this fixture records is a spurious "
                    f"SUPERconvergence, not a loss of order")

    # 4. The bias, and how it behaves under refinement. Reported as a
    #    relative gap against the well-integrated norm at each level.
    bias = {q: [(table[n][q] - table[n][GOOD]) / table[n][GOOD]
                for n in LEVELS] for q in QUADS}
    for q in QUADS:
        print(f"q{q}_relative_bias="
              f"{','.join(f'{b:+.4f}' for b in bias[q])}")

    mid = QUADS[1]
    steady = abs(bias[mid][-1] - bias[mid][0]) < 0.05
    positive = all(b > 0.05 for b in bias[mid])
    print(f"mid_order_bias_is_positive_and_steady={steady and positive}")
    if not (steady and positive):
        fail.append(f"the q={mid} norm's bias was not a steady "
                    f"over-report ({bias[mid]}); a bias that washes out "
                    f"under refinement would make the quadrature order "
                    f"harmless, which is not what is measured")

    under = all(b < -0.5 for b in bias[lowest])
    worsens = bias[lowest][-1] < bias[lowest][0]
    print(f"lowest_order_norm_under_reports_the_error={under}")
    print(f"lowest_order_under_report_worsens_with_refinement={worsens}")
    if not (under and worsens):
        fail.append(f"the q={lowest} norm did not under-report the error "
                    f"by more than half at every level, worsening under "
                    f"refinement ({bias[lowest]}); that optimistic bias "
                    f"is the actual hazard this claim should describe")

    # 5. One line for the corrected reading of the claim.
    corrected = no_floor and upward and under and worsens
    print(f"under_integration_biases_the_norm_it_does_not_floor_it="
          f"{corrected}")
    if not corrected:
        fail.append("the corrected reading — bias, not floor; optimistic, "
                    "not pessimistic — was not observed")

    if not fail:
        print("dune_mms_error_quadrature_gate=OK")
        return 0
    for reason in fail:
        print(f"FAIL: {reason}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
