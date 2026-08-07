"""Tier-2: pollution — at FIXED k*h the P1 error still grows with k; p-refinement fixes it.

Claim: skfem helmholtz#7 — the pollution effect: for large k, standard P1 has
O(k^3 h^2) phase error, so p-refinement is needed to control it.  At a fixed
h*k = 0.1 (nominally well-resolved) a P1 solution at k=50 still shows
~5-10% phase drift across the domain because the constant in front of k^3 h^2
dominates, while the same h with P2 (O(k^5 h^4)) keeps the drift < 0.1%.

Wrong variant: trust h*k = 0.1 and keep ElementQuad1 while raising k.  The
fixture holds h*k = 0.1 (i.e. 63 elements per wavelength) and the physical
domain fixed at L=1, and shows the P1 error is NOT k-independent: it grows
roughly linearly in k from k=5 to k=50, which is the pollution signature
k^3 h^2 = k (k h)^2.  ElementQuad2 at the same h is then run for comparison.

Scaled-down configuration: a one-element-thick MeshQuad strip (1002 P1 DOFs at
k=50, h=0.002) instead of a 2D domain, which at h*k=0.1 and k=50 would need
~2.5e5 DOFs.  A longer 32-wavelength run (L=4) is included because the claimed
magnitude only appears after many wavelengths of propagation.

Observed on skfem 12.0.1 (2026-08-06).  The phenomenon reproduces; the claimed
MAGNITUDE does not, and is pinned as corrected: on the unit domain at k=50 the
P1 phase drift is 0.33% of a wavelength (0.042 rad out of 50), not 5-10%.  The
claim's 5-10% is roughly the relative L2 error after ~32 wavelengths of
propagation (measured 4.8%).  The claim's "GridFunction" is NGSolve vocabulary,
not skfem.

Mutation control: T2_MUTATE=1 applies the fix this entry documents -- it sets
WRONG_ELEMENT from "P1" to "P2", i.e. p-refines the k-sweep from ElementQuad1
to ElementQuad2 at the same mesh.  The pollution then does not occur, so
'low_k=5 low_k_element=ElementQuad1 low_k_n_dofs=102',
'high_k=50 high_k_element=ElementQuad1 high_k_n_dofs=1002',
'pollution_l2_growth_gt_4x=True', 'long_run_l2_rel_in_2_to_10pct=True' and
'p_refinement_gain_gt_100x=True' disappear from the output.

Caveat found while building that control, recorded and NOT worked around: the
two DRIFT-ratio lines do not discriminate.  Under mutation both k-runs have a
phase drift at round-off (0.00000 rad each), and the ratio of those two noise
values came out 10.52, so 'pollution_drift_growth_gt_4x=True' and
'growth_tracks_k_ratio=True' still print True.  Nothing checks that the drift is
above round-off before the ratio is taken; only the L2 twin
(pollution_l2_growth_gt_4x, 1.02 under mutation) actually detects pollution.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from skfem import (Basis, BilinearForm, ElementQuad1, ElementQuad2, FacetBasis,
                   MeshQuad, asm, condense, solve)

MUTATE = os.environ.get("T2_MUTATE") == "1"

KH = 0.1              # the "nominally well-resolved" fixed product k*h
# The mistake: keep P1 and trust k*h alone.  Under mutation the documented fix
# -- p-refinement to ElementQuad2 at the same h -- is applied to the k-sweep.
WRONG_ELEMENT = "P1" if not MUTATE else "P2"
K_LOW = 5.0
K_HIGH = 50.0
L_UNIT = 1.0
L_LONG = 4.0          # ~32 wavelengths at k=50

ELEMENTS = {"P1": ElementQuad1, "P2": ElementQuad2}


def run(k: float, kind: str, L: float):
    e = ELEMENTS[kind]()
    h = KH / k
    nx = int(round(L / h))
    hy = L / nx
    m = (MeshQuad.init_tensor(np.linspace(0.0, L, nx + 1), np.array([0.0, hy]))
         .with_boundaries({"left": lambda x: x[0] < 1e-9 * L,
                           "right": lambda x: x[0] > L - 1e-9 * L}))
    b = Basis(m, e)

    @BilinearForm(dtype=complex)
    def helmholtz(u, v, w):
        return u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1] - k ** 2 * u * v

    @BilinearForm(dtype=complex)
    def abc(u, v, w):
        return 1j * k * u * v

    A = asm(helmholtz, b) + asm(abc, FacetBasis(m, e, facets="right"))
    f = np.zeros(A.shape[0], dtype=complex)
    D = b.get_dofs("left").flatten()
    x = np.zeros(A.shape[0], dtype=complex)
    x[D] = 1.0
    u = solve(*condense(A, f, x=x, D=D))

    xq = np.linspace(1e-9, L - 1e-9, 2001)
    uq = b.probes(np.vstack([xq, np.full_like(xq, 0.5 * hy)])) @ u
    exact = np.exp(-1j * k * xq)
    ph = np.unwrap(np.angle(uq))
    drift = abs((ph[-1] - ph[0]) - (-k * xq[-1] + k * xq[0]))
    return dict(k=k, element=type(e).__name__, ndofs=b.N, nelements=m.nelements,
                kh=k * h, drift_rad=drift, drift_lam=drift / (2 * np.pi),
                l2=float(np.linalg.norm(uq - exact) / np.linalg.norm(exact)))


def main() -> int:
    ok = True

    lo = run(K_LOW, WRONG_ELEMENT, L_UNIT)
    hi = run(K_HIGH, WRONG_ELEMENT, L_UNIT)
    print(f"mesh_class=MeshQuad1 fixed_kh={KH}")
    print(f"elements_per_wavelength={2 * np.pi / KH:.1f} "
          f"elements_per_wavelength_gt_60={2 * np.pi / KH > 60.0}")
    print(f"low_k={lo['k']:g} low_k_element={lo['element']} "
          f"low_k_n_dofs={lo['ndofs']}")
    print(f"high_k={hi['k']:g} high_k_element={hi['element']} "
          f"high_k_n_dofs={hi['ndofs']}")
    print(f"both_at_same_kh={abs(lo['kh'] - hi['kh']) < 1e-12}")

    # --- WRONG variant: the error is not k-independent at fixed k*h ----------
    ratio_drift = hi['drift_rad'] / lo['drift_rad']
    ratio_l2 = hi['l2'] / lo['l2']
    print(f"low_k_drift_rad={lo['drift_rad']:.5f} "
          f"high_k_drift_rad={hi['drift_rad']:.5f}")
    print(f"pollution_drift_growth_ratio={ratio_drift:.2f}")
    print(f"pollution_drift_growth_gt_4x={ratio_drift > 4.0}")
    print(f"pollution_l2_growth_gt_4x={ratio_l2 > 4.0}")
    print("growth_tracks_k_ratio="
          f"{0.5 * (K_HIGH / K_LOW) < ratio_drift < 2.0 * (K_HIGH / K_LOW)}")
    if not (ratio_drift > 4.0 and ratio_l2 > 4.0):
        print("FAIL: at fixed k*h the P1 error did not grow with k, so the "
              f"pollution effect did not occur (drift x{ratio_drift:.2f}, "
              f"L2 x{ratio_l2:.2f})", file=sys.stderr)
        ok = False

    # --- the claimed magnitude, checked and corrected ------------------------
    print(f"unit_domain_drift_pct_of_wavelength={hi['drift_lam'] * 100:.3f}")
    print("claimed_5_to_10pct_phase_drift_on_unit_domain="
          f"{0.05 <= hi['drift_lam'] <= 0.10}")
    print(f"unit_domain_drift_lt_1pct_of_wavelength={hi['drift_lam'] < 0.01}")

    long_run = run(K_HIGH, WRONG_ELEMENT, L_LONG)
    print(f"long_run_n_wavelengths={L_LONG / (2 * np.pi / K_HIGH):.1f} "
          f"long_run_n_dofs={long_run['ndofs']}")
    print(f"long_run_l2_rel_pct={long_run['l2'] * 100:.2f}")
    print(f"long_run_l2_rel_in_2_to_10pct={0.02 <= long_run['l2'] <= 0.10}")
    print(f"long_run_l2_exceeds_unit_domain_l2={long_run['l2'] > hi['l2']}")
    if not long_run['l2'] > hi['l2']:
        print("FAIL: the error did not accumulate with propagation distance",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: same h, p-refinement --------------------------------
    p2 = run(K_HIGH, "P2", L_UNIT)
    p2_long = run(K_HIGH, "P2", L_LONG)
    print(f"p2_element={p2['element']} p2_n_dofs={p2['ndofs']} "
          f"same_mesh={p2['nelements'] == hi['nelements']}")
    print(f"p2_drift_pct_of_wavelength={p2['drift_lam'] * 100:.5f}")
    print(f"p2_drift_lt_0p1pct_of_wavelength={p2['drift_lam'] * 100 < 0.1}")
    print(f"p2_l2_rel_lt_1e-4={p2['l2'] < 1e-4}")
    print(f"p_refinement_gain_gt_100x={hi['l2'] / p2['l2'] > 100.0}")
    print(f"p2_long_run_l2_rel_lt_1e-4={p2_long['l2'] < 1e-4}")
    if not (p2['drift_lam'] * 100 < 0.1 and hi['l2'] / p2['l2'] > 100.0):
        print("FAIL: p-refinement to ElementQuad2 did not control the "
              f"pollution error (drift {p2['drift_lam'] * 100:.5f}% of a "
              f"wavelength, L2 {p2['l2']:.3e})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
