"""Tier-2: the special first step for u_t(0) = 0 -- and its sign.

Claim: skfem wave#4 -- an initial condition with u_t(0) = 0 requires a special
first step, "u^{-1} = u^0 - 0.5 dt^2 M^{-1} (-c^2 K u^0)"; skipping it and
using u^{-1} = u^0 introduces a spurious initial velocity.  Signal: the
standing-wave amplitude at the centre "drifts linearly with t after t=0
instead of oscillating", and monotone drift appears in np.abs(u).max().

Measured on skfem 12.0.1, MeshQuad.init_tensor 11x11, ElementQuad1, c = 1,
row-sum-lumped mass, standing-wave IC sin(pi x) sin(pi y), integrated to
T = 1 and compared against the EXACT solution of the semi-discrete system
(spectral, from scipy.linalg.eigh of the same K and M_L), so the only error
present is the time-integration error:

  * THE SIGN IN THE QUOTED FORMULA IS WRONG.  Central differences with
    u_t(0) = 0 give u^{-1} = u^{1} = u^0 + 0.5 dt^2 a(u^0), where
    a = M^-1(-c^2 K u).  With the PLUS sign the scheme converges at second
    order in dt.  With the entry's MINUS sign it converges at FIRST order --
    and its error is about twice as large as simply setting u^{-1} = u^0,
    which also gives first order.  Following the entry as written is worse
    than doing nothing.
  * "DRIFTS LINEARLY WITH t INSTEAD OF OSCILLATING" IS NOT WHAT HAPPENS.
    All three starts oscillate; the difference is amplitude and phase error,
    not a monotone drift, and np.abs(u).max() does not grow monotonically in
    any of them.  The reliable diagnostic is the ORDER of the error in dt.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology
site -- the "entry_sign" start in march() flips from the entry's
um = u0 - 0.5*dt*dt*acc(u0) to the correct um = u0 + 0.5*dt*dt*acc(u0).
That start is then second order and no worse than doing nothing, so
'entry_minus_sign_gives_first_order=True' and
'entry_sign_worse_than_no_correction=True' disappear from the output.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

MUTATE = os.environ.get("T2_MUTATE") == "1"

import sys

import numpy as np
import scipy.linalg as sla
from skfem import Basis, ElementQuad1, MeshQuad
from skfem.models.poisson import laplace, mass

C = 1.0
NX = 11
T_END = 1.0


def setup():
    m = MeshQuad.init_tensor(np.linspace(0.0, 1.0, NX),
                             np.linspace(0.0, 1.0, NX))
    ib = Basis(m, ElementQuad1())
    K = laplace.assemble(ib).tocsr()
    M = mass.assemble(ib).tocsr()
    D = ib.get_dofs().all()
    ML = np.asarray(M.sum(axis=1)).ravel()
    u0 = ib.project(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    u0[D] = 0.0
    return ib, K, ML, D, u0


def semidiscrete(ib, K, ML, D, u0, t):
    """Exact solution of M_L u'' = -c^2 K u, u'(0) = 0, by eigendecomposition."""
    free = np.setdiff1d(np.arange(ib.N), D)
    Kf = K[free][:, free].toarray()
    Mf = np.diag(ML[free])
    lam, V = sla.eigh(Kf, Mf)
    coeff = V.T @ (Mf @ u0[free])
    out = ib.zeros()
    out[free] = V @ (coeff * np.cos(C * np.sqrt(lam) * t))
    return out


def march(ib, K, ML, D, u0, dt, T, start):
    def acc(u):
        return (-C ** 2 * (K @ u)) / ML

    if start == "correct":
        um = u0 + 0.5 * dt * dt * acc(u0)
    elif start == "entry_sign":
        # THE PATHOLOGY: the minus sign quoted by the entry.  Under
        # T2_MUTATE=1 the documented fix is applied here and this start uses
        # the correct plus sign.
        um = (u0 + 0.5 * dt * dt * acc(u0) if MUTATE
              else u0 - 0.5 * dt * dt * acc(u0))
    else:
        um = u0.copy()
    u = u0.copy()
    ctr = int(np.argmin((ib.doflocs[0] - 0.5) ** 2
                        + (ib.doflocs[1] - 0.5) ** 2))
    traj = [float(u0[ctr])]
    peaks = [float(np.abs(u).max())]
    for _ in range(int(round(T / dt))):
        un = 2.0 * u - um + dt * dt * acc(u)
        un[D] = 0.0
        um, u = u, un
        traj.append(float(u[ctr]))
        peaks.append(float(np.abs(u).max()))
    return u, np.array(traj), np.array(peaks)


def main() -> int:
    ok = True
    ib, K, ML, D, u0 = setup()
    h = 1.0 / (NX - 1)
    print(f"dofs_N={ib.N} h={h:.4f} dt_crit={h / C:.4f}")

    ref = semidiscrete(ib, K, ML, D, u0, T_END)
    orders = {}
    for start in ("correct", "entry_sign", "none"):
        errs = []
        for k in (1, 2, 4, 8):
            dt = 0.4 * h / k
            u, _, _ = march(ib, K, ML, D, u0, dt, T_END, start)
            errs.append(float(np.abs(u - ref).max()))
        rates = [float(np.log2(errs[i] / errs[i + 1]))
                 for i in range(len(errs) - 1)]
        orders[start] = (errs, rates)
        print(f"{start}_errors={[f'{e:.3e}' for e in errs]}")
        print(f"{start}_rates={[f'{r:.3f}' for r in rates]}")
        print(f"{start}_is_second_order={all(1.9 < r < 2.1 for r in rates)}")
        print(f"{start}_is_first_order={all(0.9 < r < 1.15 for r in rates)}")

    e_c, r_c = orders["correct"]
    e_e, r_e = orders["entry_sign"]
    e_n, r_n = orders["none"]
    print(f"plus_sign_gives_second_order="
          f"{all(1.9 < r < 2.1 for r in r_c)}")
    print(f"entry_minus_sign_gives_first_order="
          f"{all(0.9 < r < 1.15 for r in r_e)}")
    print(f"no_correction_gives_first_order="
          f"{all(0.9 < r < 1.15 for r in r_n)}")
    print(f"entry_sign_worse_than_no_correction={e_e[-1] > e_n[-1]}")
    print(f"entry_over_none_error_ratio={e_e[-1] / e_n[-1]:.3f}")
    if not all(1.9 < r < 2.1 for r in r_c):
        print("FAIL: the plus-sign start was not second order",
              file=sys.stderr)
        ok = False
    if all(1.9 < r < 2.1 for r in r_e):
        print("FAIL: the entry's minus-sign start WAS second order, so the "
              "sign in the entry is right after all", file=sys.stderr)
        ok = False
    if not e_e[-1] > e_n[-1]:
        print("FAIL: the entry's start was not worse than doing nothing",
              file=sys.stderr)
        ok = False

    # --- does anything "drift linearly" instead of oscillating? ---------
    dt = 0.4 * h
    for start in ("correct", "entry_sign", "none"):
        _, traj, peaks = march(ib, K, ML, D, u0, dt, 6.0, start)
        monotone = all(peaks[i + 1] >= peaks[i] - 1e-12
                       for i in range(len(peaks) - 1))
        sign_changes = int(np.sum(np.sign(traj[:-1]) * np.sign(traj[1:]) < 0))
        print(f"{start}_peak_is_monotone_growing={monotone}")
        print(f"{start}_centre_sign_changes={sign_changes}")
        print(f"{start}_oscillates={sign_changes >= 4}")
        if monotone:
            print(f"FAIL: {start} showed the monotone drift the entry "
                  f"describes", file=sys.stderr)
            ok = False
        if sign_changes < 4:
            print(f"FAIL: {start} did not oscillate", file=sys.stderr)
            ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
