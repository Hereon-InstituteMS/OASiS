"""Tier-2: Crank-Nicolson needs the SYMMETRIC right-hand side.

Claim: skfem heat_transient#2 -- Crank-Nicolson is
(M + 0.5 dt K) u_new = (M - 0.5 dt K) u_old + dt f; the symmetric formula is
required for second-order accuracy.  "Signal: linfty_norm(u_h - u_exact) after
time-stepping decreases by ~4 per dt halving (slope 2); mixing up the
(1-theta) factor on RHS degrades to slope 1 even though theta=0.5 was set."

Measured on skfem 12.0.1, u_t = Laplace u on MeshTri.init_tensor 20x20 with
ElementTriP1, homogeneous Dirichlet, IC sin(pi x) sin(pi y), T = 0.02, errors
against a 1024-step same-mesh reference:

  * CONFIRMED.  With the symmetric right-hand side the L-infinity error falls
    by about 4 per halving; replacing (M - 0.5 dt K) by the bare M -- the
    exact "forgot the (1-theta) factor" slip, with theta still set to 0.5 --
    drops it to about 2 per halving.  The left-hand side is untouched, so
    nothing about the assembled system looks wrong.
  * IT IS SILENT AND IT STILL CONVERGES: the degraded scheme is stable,
    finite, monotone and emits nothing.  Only the ORDER gives it away, which
    means a single-dt run cannot detect it at all.
  * naming what went wrong: the asymmetric variant is NOT numerically
    identical to a backward-Euler run at the same dt, but it tracks one at a
    fixed error ratio across every dt tested, so it is first order with a
    smaller constant -- effectively backward Euler taking a shorter step.
    Reporting it as "Crank-Nicolson, slightly less accurate" is the reading
    to avoid; it has lost an order.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, mass

T_END = 0.02
NX = 20


def integrate(nsteps, variant):
    m = MeshTri.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                            np.linspace(0.0, 1.0, NX + 1))
    ib = Basis(m, ElementTriP1())
    K = laplace.assemble(ib)
    M = mass.assemble(ib)
    D = ib.get_dofs().all()
    u = ib.project(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    u[D] = 0.0
    dt = T_END / nsteps
    theta = 0.5
    A = (M + theta * dt * K).tocsr()
    if variant == "symmetric":
        B = (M - (1.0 - theta) * dt * K).tocsr()
    elif variant == "dropped_factor":
        B = M.tocsr()                       # the (1-theta) term forgotten
    elif variant == "backward_euler":
        A = (M + dt * K).tocsr()
        B = M.tocsr()
    else:
        raise AssertionError(variant)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(nsteps):
            u = solve(*condense(A, B @ u, D=D))
        msgs = sorted({str(c.message) for c in caught})
    return u, ib, D, msgs


def study(variant, counts=(4, 8, 16, 32, 64)):
    uref, ib, D, _ = integrate(1024, variant)
    free = np.setdiff1d(np.arange(ib.N), D)
    errs, allmsgs = [], []
    for n in counts:
        u, _, _, msgs = integrate(n, variant)
        errs.append(float(np.abs(u[free] - uref[free]).max()))
        allmsgs += msgs
    factors = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    return errs, factors, sorted(set(allmsgs)), ib


def main() -> int:
    ok = True
    e_ok, f_ok, m_ok, ib = study("symmetric")
    e_bad, f_bad, m_bad, _ = study("dropped_factor")
    print(f"dofs_N={ib.N}")
    print(f"symmetric_errors={[f'{e:.3e}' for e in e_ok]}")
    print(f"symmetric_factors={[f'{f:.3f}' for f in f_ok]}")
    print(f"dropped_factor_errors={[f'{e:.3e}' for e in e_bad]}")
    print(f"dropped_factor_factors={[f'{f:.3f}' for f in f_bad]}")
    print(f"symmetric_quarters_the_error="
          f"{all(3.6 < f < 4.4 for f in f_ok)}")
    print(f"dropped_factor_only_halves_the_error="
          f"{all(1.8 < f < 2.3 for f in f_bad)}")
    print(f"dropped_factor_still_quarters={all(3.6 < f < 4.4 for f in f_bad)}")
    if not all(3.6 < f < 4.4 for f in f_ok):
        print("FAIL: the symmetric formula did not give second order",
              file=sys.stderr)
        ok = False
    if not all(1.8 < f < 2.3 for f in f_bad):
        print("FAIL: dropping the (1-theta) factor did not degrade to first "
              "order", file=sys.stderr)
        ok = False

    # --- silent, stable, finite -------------------------------------------
    u_bad, _, D2, msgs = integrate(64, "dropped_factor")
    print(f"dropped_factor_warnings={m_bad!r}")
    print(f"dropped_factor_is_silent={not m_bad}")
    print(f"dropped_factor_finite={bool(np.isfinite(u_bad).all())}")
    print(f"dropped_factor_bounded={bool(np.abs(u_bad).max() < 1.0)}")
    if m_bad:
        print(f"FAIL: the degraded scheme warned {m_bad!r}", file=sys.stderr)
        ok = False

    # --- what the degraded scheme actually is ------------------------------
    e_be, f_be, _, _ = study("backward_euler")
    print(f"backward_euler_errors={[f'{e:.3e}' for e in e_be]}")
    ratios = [e_bad[i] / e_be[i] for i in range(len(e_be))]
    print(f"dropped_over_backward_euler_ratios="
          f"{[f'{r:.4f}' for r in ratios]}")
    spread = max(ratios) - min(ratios)
    print(f"dropped_ratio_spread={spread:.4f}")
    print(f"dropped_factor_is_exactly_backward_euler="
          f"{max(abs(r - 1.0) for r in ratios) < 1e-6}")
    print(f"dropped_factor_tracks_backward_euler_at_a_fixed_ratio="
          f"{spread < 0.02}")
    be_factors = f_be
    print(f"backward_euler_factors={[f'{f:.3f}' for f in be_factors]}")
    print(f"both_are_first_order="
          f"{all(1.8 < f < 2.3 for f in f_bad + be_factors)}")
    if spread >= 0.02:
        print("FAIL: the degraded scheme did not track backward Euler at a "
              "fixed ratio, so it is not simply first order",
              file=sys.stderr)
        ok = False
    if not all(1.8 < f < 2.3 for f in f_bad + be_factors):
        print("FAIL: the degraded scheme and backward Euler were not both "
              "first order", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
