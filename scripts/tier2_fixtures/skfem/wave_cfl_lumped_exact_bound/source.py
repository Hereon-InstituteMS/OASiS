"""Tier-2: the exact CFL bound for lumped-mass central differences.

Claim: skfem wave#0 -- for explicit central differences with ROW-SUM-LUMPED
mass on a uniform Q1 tensor grid the exact stability limit is
dt_crit = 2/sqrt(c^2 * lambda_max(M_L^-1 K)) = h/c, because
lambda_max(M_L^-1 K) = 4/h^2 exactly; dt < h/(c*sqrt(2)) is SAFE but is not
the boundary.  Signal: dt/dt_crit = 0.636 / 0.707 / 0.849 / 0.990 stay
bounded while 1.061 / 1.131 / ... blow up; a short run hides this, so
integrate long enough.

Measured on skfem 12.0.1, MeshQuad.init_tensor uniform grids, ElementQuad1,
c = 1, standing-wave IC sin(pi x) sin(pi y), integrated to T = 20:

  * lambda_max(M_L^-1 K) = 4/h^2 to six decimal places at nx = 5, 9, 11 and
    21, so dt_crit = h/c EXACTLY, not approximately.
  * every ratio at or below 0.990 stays bounded; every ratio at or above
    1.061 grows geometrically past 1e44.  The transition is between them.
  * the "max|u| <= 1.0" wording in the entry is slightly optimistic: the
    bounded runs peak a little above 1 because the nodal projection of the
    initial mode is not exactly 1.  Test boundedness, not a unit ceiling.
  * the short-run warning is confirmed: at T = 0.5 an unstable ratio is still
    finite and unremarkable.
"""
from __future__ import annotations

import os

# Pin the BLAS thread count BEFORE numpy is imported: the dense eigensolves
# here are small, and thread oversubscription makes the run time depend on how
# busy the host is rather than on the code.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import warnings

import numpy as np
import scipy.sparse as sp
from skfem import Basis, ElementQuad1, MeshQuad
from skfem.models.poisson import laplace, mass

C = 1.0
SAFE = (0.636, 0.707, 0.849, 0.990)
UNSAFE = (1.061, 1.131, 1.202, 1.273, 1.414)


def build(nx):
    m = MeshQuad.init_tensor(np.linspace(0.0, 1.0, nx),
                             np.linspace(0.0, 1.0, nx))
    ib = Basis(m, ElementQuad1())
    K = laplace.assemble(ib).tocsr()
    M = mass.assemble(ib).tocsr()
    return m, ib, K, np.asarray(M.sum(axis=1)).ravel(), ib.get_dofs().all()


def lambda_max_lumped(nx):
    _, ib, K, ML, _ = build(nx)
    A = (sp.diags(1.0 / ML) @ K).toarray()
    return float(np.linalg.eigvals(A).real.max())


def march(nx, dt, T):
    _, ib, K, ML, D = build(nx)
    u0 = ib.project(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    u0[D] = 0.0

    def acc(u):
        return (-C ** 2 * (K @ u)) / ML

    um = u0 + 0.5 * dt * dt * acc(u0)
    u = u0.copy()
    peak = float(np.abs(u).max())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(int(round(T / dt))):
            un = 2.0 * u - um + dt * dt * acc(u)
            un[D] = 0.0
            um, u = u, un
            peak = max(peak, float(np.abs(u).max()))
            if not np.isfinite(peak):
                break
    return peak


def main() -> int:
    ok = True
    print(f"c={C}")

    # --- lambda_max is exactly 4/h^2 -----------------------------------
    exact = True
    for nx in (5, 9, 11, 21):
        h = 1.0 / (nx - 1)
        lam = lambda_max_lumped(nx)
        ratio = lam / (4.0 / h / h)
        print(f"nx{nx}_h={h:.4f} lambda_max={lam:.6f} four_over_h2="
              f"{4.0 / h / h:.6f} ratio={ratio:.6f}")
        print(f"nx{nx}_lambda_max_equals_4_over_h2="
              f"{abs(ratio - 1.0) < 1e-6}")
        print(f"nx{nx}_dt_crit={2.0 / np.sqrt(lam):.6f} h_over_c={h / C:.6f}")
        print(f"nx{nx}_dt_crit_equals_h_over_c="
              f"{abs(2.0 / np.sqrt(lam) - h / C) < 1e-9}")
        exact = exact and abs(ratio - 1.0) < 1e-6
    print(f"lambda_max_is_exactly_4_over_h2_at_every_nx={exact}")
    if not exact:
        print("FAIL: lambda_max(M_L^-1 K) was not 4/h^2", file=sys.stderr)
        ok = False

    # --- the sweep -----------------------------------------------------
    nx = 11
    h = 1.0 / (nx - 1)
    dt_crit = h / C
    print(f"sweep_nx={nx} dt_crit={dt_crit:.6f}")
    safe_peaks, unsafe_peaks = [], []
    for r in SAFE:
        p = march(nx, r * dt_crit, 20.0)
        safe_peaks.append(p)
        print(f"ratio_{r}_peak={p:.4e} bounded={p < 10.0}")
    for r in UNSAFE:
        p = march(nx, r * dt_crit, 20.0)
        unsafe_peaks.append(p)
        print(f"ratio_{r}_peak={p:.4e} bounded={p < 10.0}")
    print(f"all_ratios_up_to_0p990_bounded="
          f"{all(p < 10.0 for p in safe_peaks)}")
    print(f"all_ratios_from_1p061_blow_up="
          f"{all(p > 1e10 for p in unsafe_peaks)}")
    print(f"max_bounded_peak={max(safe_peaks):.4f}")
    print(f"bounded_peaks_stay_under_unity={max(safe_peaks) <= 1.0}")
    if not all(p < 10.0 for p in safe_peaks):
        print("FAIL: a ratio at or below 0.990 blew up", file=sys.stderr)
        ok = False
    if not all(p > 1e10 for p in unsafe_peaks):
        print("FAIL: a ratio at or above 1.061 stayed bounded",
              file=sys.stderr)
        ok = False

    # --- a short run hides it ------------------------------------------
    short = march(nx, 1.5 * dt_crit, 0.5)
    long_ = march(nx, 1.5 * dt_crit, 20.0)
    print(f"unstable_short_run_peak={short:.4e}")
    print(f"unstable_long_run_peak={long_:.4e}")
    print(f"short_run_looks_stable={short < 10.0}")
    print(f"long_run_reveals_instability={long_ > 1e10}")
    if not (short < 10.0 and long_ > 1e10):
        print("FAIL: the short run did not hide the instability",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
