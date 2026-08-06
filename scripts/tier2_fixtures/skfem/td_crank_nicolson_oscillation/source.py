"""Tier-2: Crank-Nicolson rings at sharp transients; backward Euler does not.

Claim: skfem time_dependent#1 -- Crank-Nicolson (theta = 0.5) is 2nd order but
can oscillate: "temperature/concentration shows 10-30% over/undershoot at
sharp transients that does not damp with mesh refinement; switching to
theta = 1 (BE) removes the oscillation at the cost of 1st-order phase error."

Measured on skfem 12.0.1, heat equation with a square-pulse initial condition
on MeshTri.init_tensor n x n, ElementTriP1, homogeneous Dirichlet, ten steps:

  * THE "DOES NOT DAMP WITH REFINEMENT" PART IS TRUE, but only once the
    comparison is made at a FIXED dt/h^2.  Refining the mesh at a fixed dt
    makes dt/h^2 grow, and refining with dt ~ h^2 keeps the undershoot
    roughly constant -- it is the ratio that governs the ringing, not h.
  * "SWITCHING TO theta = 1 REMOVES THE OSCILLATION" IS TRUE.  Backward
    Euler's worst undershoot over all nine cases is at the rounding level --
    more than four orders of magnitude below Crank-Nicolson's SMALLEST --
    while Crank-Nicolson undershoots visibly at every mesh and every dt.
  * THE 10-30% FIGURE IS NOT A PROPERTY OF THE SCHEME.  The measured
    undershoot is a few percent at dt/h^2 = 0.5 and rises past 60 % at
    dt/h^2 = 50, so it spans an order of magnitude across ordinary choices.
    Test for a NEGATIVE minimum, not for a percentage band.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, mass

STEPS = 10


def pulse(x):
    return np.where((np.abs(x[0] - 0.5) < 0.2) & (np.abs(x[1] - 0.5) < 0.2),
                    1.0, 0.0)


def run(theta, n, dt):
    m = MeshTri.init_tensor(np.linspace(0.0, 1.0, n + 1),
                            np.linspace(0.0, 1.0, n + 1))
    ib = Basis(m, ElementTriP1())
    K = laplace.assemble(ib)
    M = mass.assemble(ib)
    D = ib.get_dofs().all()
    u = ib.project(pulse)
    u[D] = 0.0
    A = (M + theta * dt * K).tocsr()
    B = (M - (1.0 - theta) * dt * K).tocsr()
    worst = 0.0
    for _ in range(STEPS):
        u = solve(*condense(A, B @ u, D=D))
        worst = min(worst, float(u.min()))
    return -worst, ib


def main() -> int:
    ok = True
    table = {}
    for n in (16, 32, 64):
        h2 = (1.0 / n) ** 2
        for ratio in (0.5, 5.0, 50.0):
            for theta, tag in ((0.5, "cn"), (1.0, "be")):
                under, ib = run(theta, n, ratio * h2)
                table[(tag, n, ratio)] = under
                print(f"{tag}_n{n}_dtoverh2_{ratio:g}_undershoot={under:.4f}")
                if tag == "cn" and n == 16 and ratio == 5.0:
                    print(f"cn_reference_dofs_N={ib.N}")

    be_all = [v for (t, _, _), v in table.items() if t == "be"]
    cn_all = [v for (t, _, _), v in table.items() if t == "cn"]
    print(f"be_max_undershoot_over_all_cases={max(be_all):.6f}")
    print(f"be_never_undershoots={max(be_all) < 1e-4}")
    print(f"be_undershoot_below_cn_by_gt_1000x="
          f"{max(be_all) * 1000.0 < min(cn_all)}")
    print(f"cn_min_undershoot_over_all_cases={min(cn_all):.6f}")
    print(f"cn_always_undershoots={min(cn_all) > 1e-3}")
    if max(be_all) >= 1e-4:
        print("FAIL: backward Euler undershot too, so theta=1 does not "
              "remove the oscillation", file=sys.stderr)
        ok = False
    if min(cn_all) <= 1e-3:
        print("FAIL: Crank-Nicolson did not undershoot in every case",
              file=sys.stderr)
        ok = False

    # --- refinement at a FIXED dt/h^2 does not damp the ringing ---------
    for ratio in (5.0, 50.0):
        series = [table[("cn", n, ratio)] for n in (16, 32, 64)]
        damped = series[-1] < 0.5 * series[0]
        print(f"cn_ratio{ratio:g}_series={[f'{x:.4f}' for x in series]}")
        print(f"cn_ratio{ratio:g}_damps_with_refinement={damped}")
        if damped:
            print(f"FAIL: at dt/h^2 = {ratio} the CN undershoot DID damp "
                  f"with refinement", file=sys.stderr)
            ok = False

    # --- the quoted 10-30 % band -----------------------------------------
    span_low = table[("cn", 32, 0.5)]
    span_high = table[("cn", 32, 50.0)]
    print(f"cn_undershoot_low_ratio={span_low:.4f}")
    print(f"cn_undershoot_high_ratio={span_high:.4f}")
    print(f"cn_undershoot_within_10_to_30_percent_band="
          f"{0.10 <= span_low <= 0.30 and 0.10 <= span_high <= 0.30}")
    print(f"cn_undershoot_spans_more_than_10x="
          f"{span_high > 10.0 * span_low}")
    if 0.10 <= span_low <= 0.30 and 0.10 <= span_high <= 0.30:
        print("FAIL: the undershoot really did stay inside 10-30%, so the "
              "quoted band is usable after all", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
