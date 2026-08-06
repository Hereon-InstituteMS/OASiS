"""Tier-2: for backward Euler dt is an accuracy choice, not a stability one.

Claim: skfem time_dependent#4 -- CFL is NOT needed for BE (the implicit scheme
is L-stable), but dt still affects ACCURACY.  "dt larger than the shortest
physical timescale of the problem gives the correct steady state but smears
any sharp transient -- comparing the simulated u(t_transient) against a
fine-dt reference shows L2 error ~ O(dt) instead of resolving the front.
Choose dt by accuracy, not stability, for implicit schemes."

Measured on skfem 12.0.1, heat equation on a 24x24 MeshTri.init_tensor with
ElementTriP1, cold interior, wall switched to 1 at t = 0, so there is a real
front to smear:

  * STABILITY: a dt a thousand times the explicit limit 2/lambda_max still
    produces a bounded, monotone, finite solution.  Nothing warns and nothing
    blows up, so an eye on stability alone tells you nothing.
  * STEADY STATE: run long enough, every dt lands on the SAME steady state
    (the discrete harmonic extension of the wall value) to machine precision,
    which is exactly the trap -- the answer at the end looks right.
  * TRANSIENT: at a fixed early time the error against a fine-dt reference
    falls at first order in dt, confirming the O(dt) part of the claim, and
    the coarse-dt profile is visibly smeared -- at T = 0.01 the front has
    barely left the wall, yet the coarse run already shows more than an
    order of magnitude too much heat at the domain CENTRE.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import scipy.linalg as sla
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, mass

NX = 24
T_TRANSIENT = 0.01
T_STEADY = 5.0


def build():
    m = MeshTri.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                            np.linspace(0.0, 1.0, NX + 1))
    ib = Basis(m, ElementTriP1())
    return m, ib, laplace.assemble(ib), mass.assemble(ib), ib.get_dofs().all()


def integrate(nsteps, T):
    m, ib, K, M, D = build()
    A = (M + (T / nsteps) * K).tocsr()
    xbc = ib.zeros()
    xbc[D] = 1.0
    u = ib.zeros()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(nsteps):
            u = solve(*condense(A, M @ u, x=xbc, D=D))
        msgs = sorted({str(c.message) for c in caught})
    return u, ib, M, D, msgs


def l2(M, a, b):
    d = a - b
    return float(np.sqrt(d @ (M @ d)))


def main() -> int:
    ok = True
    m, ib, K, M, D = build()
    free = np.setdiff1d(np.arange(ib.N), D)
    ev = sla.eigh(K[free][:, free].toarray(), M[free][:, free].toarray(),
                  eigvals_only=True)
    dt_explicit = 2.0 / float(ev.max())
    print(f"dofs_N={ib.N} lambda_max={float(ev.max()):.4e}")
    print(f"explicit_stability_limit_dt={dt_explicit:.4e}")

    # --- stability: a huge dt is still bounded ---------------------------
    huge = 1000.0 * dt_explicit
    nsteps_huge = max(1, int(round(T_STEADY / huge)))
    u_huge, _, Mh, Dh, msgs = integrate(nsteps_huge, T_STEADY)
    print(f"huge_dt_over_explicit_limit={huge / dt_explicit:.1f}")
    print(f"huge_dt_solution_finite={bool(np.isfinite(u_huge).all())}")
    print(f"huge_dt_max={float(u_huge.max()):.6f}")
    print(f"huge_dt_min={float(u_huge.min()):.6f}")
    print(f"huge_dt_stays_bounded="
          f"{bool(u_huge.max() <= 1.0 + 1e-9 and u_huge.min() >= -1e-9)}")
    print(f"huge_dt_warnings={msgs!r}")
    if not np.isfinite(u_huge).all() or u_huge.max() > 1.0 + 1e-9:
        print("FAIL: backward Euler was not unconditionally stable",
              file=sys.stderr)
        ok = False

    # --- the same steady state whatever dt -------------------------------
    u_fine, _, _, _, _ = integrate(400, T_STEADY)
    steady_gap = l2(M, u_huge, u_fine)
    print(f"steady_state_gap_between_coarse_and_fine_dt={steady_gap:.3e}")
    print(f"steady_state_is_dt_independent={steady_gap < 1e-9}")
    if steady_gap >= 1e-9:
        print("FAIL: the steady state depended on dt, so the trap the claim "
              "describes does not exist here", file=sys.stderr)
        ok = False

    # --- the transient does depend on dt ---------------------------------
    ref, _, _, _, _ = integrate(2048, T_TRANSIENT)
    counts = [8, 16, 32, 64]
    errs = [l2(M, integrate(n, T_TRANSIENT)[0], ref) for n in counts]
    rates = [float(np.log2(errs[i] / errs[i + 1]))
             for i in range(len(errs) - 1)]
    print(f"transient_errors={[f'{x:.3e}' for x in errs]}")
    print(f"transient_rates={[f'{x:.3f}' for x in rates]}")
    print(f"transient_error_is_first_order_in_dt="
          f"{all(0.8 < r < 1.25 for r in rates)}")
    if not all(0.8 < r < 1.25 for r in rates):
        print("FAIL: the transient error was not O(dt)", file=sys.stderr)
        ok = False

    # --- and the coarse-dt front really is smeared ------------------------
    u_coarse = integrate(2, T_TRANSIENT)[0]
    # the front has barely left the wall at T_TRANSIENT, so the value at the
    # DOMAIN CENTRE is the sharpest measure of how far the scheme has
    # artificially smeared it inward
    centre = int(np.argmin((ib.doflocs[0] - 0.5) ** 2
                           + (ib.doflocs[1] - 0.5) ** 2))
    centre_ref = float(ref[centre])
    centre_coarse = float(u_coarse[centre])
    print(f"transient_reference_centre_value={centre_ref:.6e}")
    print(f"transient_coarse_centre_value={centre_coarse:.6e}")
    print(f"coarse_dt_smears_the_front="
          f"{centre_coarse > 10.0 * centre_ref}")
    print(f"coarse_dt_transient_is_finite="
          f"{bool(np.isfinite(u_coarse).all())}")
    if not centre_coarse > 10.0 * centre_ref:
        print("FAIL: the coarse dt did not smear the front", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
