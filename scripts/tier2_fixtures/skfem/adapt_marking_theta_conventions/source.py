"""Tier-2: "Dorfler / max-strategy" with theta = 0.5 are OPPOSITE strategies.

Claim: skfem adaptive_poisson#4 -- "Dorfler / max-strategy marking with
theta=0.5 is the standard sweet spot -- theta -> 1 gives near-uniform
refinement (slow convergence per DOF), theta -> 0 gives 1-element-at-a-time
refinement (many iterations needed) ... healthy adaptive convergence shows N
growing ~1.3-2x per step while max_eta drops ~30-50%."

Measured on skfem 12.0.1, L-shaped domain (MeshTri.init_lshaped().refined(3),
384 elements), ElementTriP1, harmonic problem with the exact r^(2/3) corner
solution imposed on the boundary, elemental indicator from the interior-facet
gradient jump:

  * THE ENTRY NAMES TWO STRATEGIES WHOSE theta RUNS IN OPPOSITE DIRECTIONS.
    Dorfler marks the smallest set whose indicators carry a theta-fraction of
    the total, so theta -> 0 marks ONE element and theta -> 1 marks nearly
    all.  The max-strategy marks every element with eta_K > theta * max eta,
    so theta -> 0 marks ALL and theta -> 1 marks one.  On the SAME indicator
    field the two counts move in opposite directions as theta rises.  Writing
    "Dorfler / max-strategy with theta = 0.5" therefore does not name a
    setting: it names two different ones.
  * so the entry's own guidance is right for Dorfler and backwards for the
    max-strategy, and a reader who implements the max-strategy from that
    sentence gets near-uniform refinement exactly where they were told to
    expect one element at a time.
  * the DOF-growth rule of thumb is checked directly on a real Dorfler loop.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology
site -- the max-strategy threshold is complemented, eta_K > (1 - theta) * max
eta, so its theta finally runs the same way round as Dorfler's -- and the
opposite-direction finding disappears.  Re-run with
T2_MUTATE=1 python source.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from skfem import (
    Basis,
    ElementTriP1,
    Functional,
    InteriorFacetBasis,
    MeshTri,
    condense,
    solve,
)
from skfem.models.poisson import laplace

MUTATE = os.environ.get("T2_MUTATE") == "1"


def exact(x):
    r = np.sqrt(x[0] ** 2 + x[1] ** 2)
    th = np.arctan2(x[1], x[0])
    th = np.where(th < 0.0, th + 2.0 * np.pi, th)
    return r ** (2.0 / 3.0) * np.sin(2.0 / 3.0 * th)


def solve_l(m):
    ib = Basis(m, ElementTriP1())
    K = laplace.assemble(ib)
    D = ib.get_dofs().all()
    x = ib.zeros()
    x[D] = exact(ib.doflocs[:, D])
    return ib, solve(*condense(K, ib.zeros(), x=x, D=D))


def indicator(m, ib, u):
    e = ElementTriP1()
    f0 = InteriorFacetBasis(m, e, side=0)
    f1 = InteriorFacetBasis(m, e, side=1)

    @Functional
    def jump(w):
        return w.h * ((w["g0"][0] - w["g1"][0]) * w.n[0]
                      + (w["g0"][1] - w["g1"][1]) * w.n[1]) ** 2

    ef = jump.elemental(f0, g0=f0.interpolate(u).grad,
                        g1=f1.interpolate(u).grad)
    a = np.zeros(m.t.shape[1])
    np.add.at(a, f0.tind, ef)
    np.add.at(a, f1.tind, ef)
    return np.sqrt(a)


def dorfler(eta, theta):
    order = np.argsort(eta)[::-1]
    cum = np.cumsum(eta[order] ** 2)
    k = int(np.searchsorted(cum, theta * cum[-1])) + 1
    return order[:k]


def max_strategy(eta, theta):
    # The pathology: this theta is the bulk-threshold fraction, so it runs
    # the OPPOSITE way round to Dorfler's theta.  T2_MUTATE=1 applies the
    # documented fix -- read the max-strategy parameter as the complement,
    # eta_K > (1 - theta) * max eta -- so both conventions agree in direction.
    thr = theta if not MUTATE else 1.0 - theta
    return np.nonzero(eta > thr * eta.max())[0]


def main() -> int:
    ok = True
    m = MeshTri.init_lshaped().refined(3)
    ib, u = solve_l(m)
    eta = indicator(m, ib, u)
    ne = m.t.shape[1]
    print(f"elements={ne} dofs_N={ib.N}")
    print(f"indicator_is_positive={bool((eta >= 0).all())}")

    dor, mx = [], []
    for theta in (0.1, 0.5, 0.9):
        d = len(dorfler(eta, theta))
        s = len(max_strategy(eta, theta))
        dor.append(d)
        mx.append(s)
        print(f"theta{theta}_dorfler_marks={d} max_strategy_marks={s}")
    print(f"dorfler_marks_more_as_theta_rises="
          f"{dor[0] < dor[1] <= dor[2] and dor[0] < dor[2]}")
    print(f"max_strategy_marks_fewer_as_theta_rises="
          f"{mx[0] > mx[1] >= mx[2] and mx[0] > mx[2]}")
    print(f"the_two_conventions_run_in_opposite_directions="
          f"{(dor[2] > dor[0]) and (mx[2] < mx[0])}")
    print(f"theta_0p5_marks_different_counts={dor[1] != mx[1]}")
    if not ((dor[2] > dor[0]) and (mx[2] < mx[0])):
        print("FAIL: the two marking conventions did not run in opposite "
              "directions", file=sys.stderr)
        ok = False

    # --- what theta -> 0 and theta -> 1 mean, per strategy ---------------
    print(f"dorfler_theta_tiny_marks={len(dorfler(eta, 1e-6))}")
    print(f"dorfler_theta_one_marks={len(dorfler(eta, 1.0))}")
    print(f"max_theta_tiny_marks={len(max_strategy(eta, 1e-6))}")
    print(f"max_theta_one_marks={len(max_strategy(eta, 1.0))}")
    print(f"dorfler_theta_tiny_is_one_element={len(dorfler(eta, 1e-6)) == 1}")
    print(f"dorfler_theta_one_is_everything="
          f"{len(dorfler(eta, 1.0)) == ne}")
    print(f"max_theta_tiny_is_everything="
          f"{len(max_strategy(eta, 1e-6)) > 0.9 * ne}")
    print(f"max_theta_one_is_nothing={len(max_strategy(eta, 1.0)) == 0}")
    if len(dorfler(eta, 1e-6)) != 1 or len(dorfler(eta, 1.0)) != ne:
        print("FAIL: Dorfler's limits are not one element and everything",
              file=sys.stderr)
        ok = False
    if not (len(max_strategy(eta, 1e-6)) > 0.9 * ne):
        print("FAIL: the max-strategy at tiny theta did not mark nearly all",
              file=sys.stderr)
        ok = False

    # --- the DOF-growth rule of thumb, on a real Dorfler loop -----------
    mm = MeshTri.init_lshaped().refined(2)
    ns, maxeta = [], []
    for _ in range(7):
        ibx, ux = solve_l(mm)
        ex = indicator(mm, ibx, ux)
        ns.append(ibx.N)
        maxeta.append(float(ex.max()))
        mm = mm.refined(dorfler(ex, 0.5))
    growth = [ns[i + 1] / ns[i] for i in range(len(ns) - 1)]
    print(f"dorfler_theta0p5_dof_counts={ns}")
    print(f"dorfler_theta0p5_dof_growth={[f'{g:.3f}' for g in growth]}")
    print(f"dorfler_theta0p5_max_eta={[f'{x:.4e}' for x in maxeta]}")
    print(f"dof_growth_within_1p3_to_2x="
          f"{all(1.3 <= g <= 2.0 for g in growth)}")
    print(f"dof_growth_is_far_below_the_rule_of_thumb="
          f"{all(g < 1.3 for g in growth)}")
    print(f"max_eta_drops_30_to_50_percent_per_step="
          f"{all(0.5 <= maxeta[i + 1] / maxeta[i] <= 0.7
                 for i in range(len(maxeta) - 1))}")
    if all(1.3 <= g <= 2.0 for g in growth):
        print("FAIL: theta = 0.5 Dorfler DID grow N by 1.3-2x per step, so "
              "the rule of thumb needs no qualification", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
