"""Tier-2: backward Euler is first order in time, Crank-Nicolson second.

Claim: skfem time_dependent#0 -- backward Euler (theta = 1) is
unconditionally stable and 1st order in time; halving dt should halve the
error, i.e. slope 1 on a log-log plot.  "If the slope is 2, the scheme is
actually Crank-Nicolson and theta is mis-configured; if the slope is 0, dt is
too large to be in the asymptotic regime."

Measured on skfem 12.0.1, heat equation u_t = Laplace u on the unit square,
MeshTri.init_tensor 24x24 with ElementTriP1, homogeneous Dirichlet, initial
condition sin(pi x) sin(pi y), integrated to T = 0.02:

  * measuring against the ANALYTIC solution does not work for this claim.
    The spatial discretisation error dominates as soon as dt is small, so
    Crank-Nicolson's error saturates at a constant and its measured slope
    collapses toward 0 -- which the entry would misread as "dt too large".
    The reference has to be the SEMI-DISCRETE solution: the same mesh,
    the same theta, with a much smaller dt.
  * against that reference theta = 1 gives a measured slope of ~1 and
    theta = 0.5 gives ~2, across four successive halvings each, so the
    entry's diagnostic is sound once the reference is right.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, mass

T_END = 0.02
NX = 24


def ic(x):
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])


def exact(x, t):
    return ic(x) * np.exp(-2.0 * np.pi ** 2 * t)


def setup():
    m = MeshTri.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                            np.linspace(0.0, 1.0, NX + 1))
    return m, Basis(m, ElementTriP1())


def integrate(theta, nsteps):
    m, ib = setup()
    K = laplace.assemble(ib)
    M = mass.assemble(ib)
    D = ib.get_dofs().all()
    u = ib.project(ic)
    u[D] = 0.0
    dt = T_END / nsteps
    A = (M + theta * dt * K).tocsr()
    B = (M - (1.0 - theta) * dt * K).tocsr()
    for _ in range(nsteps):
        u = solve(*condense(A, B @ u, D=D))
    return u, ib, M


def l2(M, a, b):
    d = a - b
    return float(np.sqrt(d @ (M @ d)))


def slopes(errs):
    return [float(np.log2(errs[i] / errs[i + 1]))
            for i in range(len(errs) - 1)]


def main() -> int:
    ok = True
    m, ib = setup()
    print(f"dofs_N={ib.N} elements={m.t.shape[1]} T={T_END}")

    counts = [4, 8, 16, 32, 64]
    results = {}
    for theta, tag in ((1.0, "be"), (0.5, "cn")):
        uref, _, M = integrate(theta, 1024)
        errs_semi, errs_analytic = [], []
        for n in counts:
            u, ibn, Mn = integrate(theta, n)
            errs_semi.append(l2(M, u, uref))
            errs_analytic.append(l2(Mn, u, ibn.project(
                lambda x: exact(x, T_END))))
        s_semi = slopes(errs_semi)
        s_ana = slopes(errs_analytic)
        results[tag] = (s_semi, s_ana)
        print(f"{tag}_errors_vs_semidiscrete="
              f"{[f'{x:.3e}' for x in errs_semi]}")
        print(f"{tag}_slopes_vs_semidiscrete={[f'{x:.3f}' for x in s_semi]}")
        print(f"{tag}_errors_vs_analytic="
              f"{[f'{x:.3e}' for x in errs_analytic]}")
        print(f"{tag}_slopes_vs_analytic={[f'{x:.3f}' for x in s_ana]}")

    be_semi, be_ana = results["be"]
    cn_semi, cn_ana = results["cn"]
    print(f"be_slope_is_one={all(0.85 < s < 1.15 for s in be_semi)}")
    print(f"cn_slope_is_two={all(1.85 < s < 2.15 for s in cn_semi)}")
    print(f"be_and_cn_are_distinguishable="
          f"{max(be_semi) < min(cn_semi)}")
    if not all(0.85 < s < 1.15 for s in be_semi):
        print("FAIL: backward Euler did not measure first order",
              file=sys.stderr)
        ok = False
    if not all(1.85 < s < 2.15 for s in cn_semi):
        print("FAIL: Crank-Nicolson did not measure second order",
              file=sys.stderr)
        ok = False

    # --- the reference matters: against the analytic solution CN lies ---
    print(f"cn_analytic_slope_collapses={cn_ana[-1] < 0.5}")
    print(f"cn_analytic_last_slope={cn_ana[-1]:.3f}")
    print(f"analytic_reference_would_misdiagnose_cn_as_dt_too_large="
          f"{cn_ana[-1] < 0.5 and cn_semi[-1] > 1.85}")
    if not (cn_ana[-1] < 0.5):
        print("FAIL: the analytic reference did NOT saturate, so the "
              "reference choice does not matter after all", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
