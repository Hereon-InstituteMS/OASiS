"""Tier-2: the integral of the point-source solution, checked against theory.

Claim: skfem point_source#4 -- the total integral of u over Omega equals a
Green's-function flux balance; for -Laplace u = delta(x - x0) on the unit
square with Dirichlet BC and x0 = (0.5, 0.5) the value "should be near a known
Fourier-series sum", and the signal is that the computed integral differs from
that sum by orders of magnitude if the source magnitude, the BC orientation or
the sign in K is wrong.

The entry's own wording ("equals -1/lambda1 in the Laplacian spectrum") is not
the right identity.  The correct one is the double sine series: with
u = sum_{p,q} 4 sin(p pi x0) sin(q pi y0) sin(p pi x) sin(q pi y) /
(pi^2 (p^2 + q^2)), integrating over the unit square leaves
int u dx = sum over odd p, q of 16 (-1)^((p-1)/2) (-1)^((q-1)/2) /
(pi^4 p q (p^2 + q^2)).

Measured on skfem 12.0.1, ElementTriP1 on MeshTri().refined(r), r = 3..7:

  * the series (summed to p, q < 2001) and the FE integral agree, and the
    relative gap falls at second order under refinement -- so the integral IS
    a usable verification quantity even though the pointwise solution is not.
  * the three failure modes the entry names are all caught by it: doubling
    the source magnitude doubles the integral exactly, flipping the sign of K
    flips the integral exactly, and dropping the Dirichlet condition on one
    side inflates it by about a third.  None of the three raises or warns --
    the entry's "differs by orders of magnitude" overstates the boundary
    case, which shifts by tens of percent, not decades; the integral is
    still the quantity that notices, but the tolerance has to be tight.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, mass

SERIES_TERMS = 2001


def analytic_integral():
    tot = 0.0
    for p in range(1, SERIES_TERMS, 2):
        for q in range(1, SERIES_TERMS, 2):
            tot += (16.0 * ((-1) ** ((p - 1) // 2)) * ((-1) ** ((q - 1) // 2))
                    / (np.pi ** 4 * p * q * (p * p + q * q)))
    return tot


def run(refine, magnitude=1.0, sign=1.0, drop_one_side=False):
    m = MeshTri().refined(refine)
    ib = Basis(m, ElementTriP1())
    K = sign * laplace.assemble(ib)
    M = mass.assemble(ib)
    if drop_one_side:
        D = ib.get_dofs(lambda x: (x[0] < 1e-10) | (x[1] < 1e-10)
                        | (x[1] > 1.0 - 1e-10)).all()
    else:
        D = ib.get_dofs().all()
    f = ib.zeros()
    src = int(np.argmin((ib.doflocs[0] - 0.5) ** 2
                        + (ib.doflocs[1] - 0.5) ** 2))
    f[src] = magnitude
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        u = solve(*condense(K, f, D=D))
        msgs = sorted({str(c.message) for c in caught})
    return ib, float(np.ones(ib.N) @ (M @ u)), msgs


def main() -> int:
    ok = True
    exact = analytic_integral()
    print(f"series_terms_per_axis={(SERIES_TERMS - 1) // 2}")
    print(f"analytic_integral={exact:.8f}")

    gaps = []
    for r in (3, 4, 5, 6, 7):
        ib, val, _ = run(r)
        gap = abs(val - exact) / exact
        gaps.append(gap)
        print(f"r{r}_N={ib.N} integral={val:.8f} relative_gap={gap:.4e}")
    rates = [float(np.log2(gaps[i] / gaps[i + 1]))
             for i in range(len(gaps) - 1)]
    print(f"integral_gap_rates={[f'{x:.3f}' for x in rates]}")
    print(f"integral_converges_to_the_series={gaps[-1] < 1e-4}")
    print(f"integral_converges_at_second_order="
          f"{all(1.8 < x < 2.2 for x in rates)}")
    if gaps[-1] >= 1e-4:
        print("FAIL: the FE integral did not approach the series value",
              file=sys.stderr)
        ok = False
    if not all(1.8 < x < 2.2 for x in rates):
        print("FAIL: the integral did not converge at second order",
              file=sys.stderr)
        ok = False

    # --- the three failure modes the entry names -------------------------
    base_ib, base, _ = run(6)
    ib2, doubled, m2 = run(6, magnitude=2.0)
    ib3, flipped, m3 = run(6, sign=-1.0)
    ib4, partial, m4 = run(6, drop_one_side=True)
    print(f"base_integral={base:.8f}")
    print(f"double_magnitude_integral={doubled:.8f}")
    print(f"double_magnitude_ratio={doubled / base:.6f}")
    print(f"magnitude_error_is_caught={abs(doubled / base - 2.0) < 1e-9}")
    print(f"flipped_sign_integral={flipped:.8f}")
    print(f"flipped_sign_ratio={flipped / base:.6f}")
    print(f"sign_error_is_caught={flipped / base < -0.9}")
    print(f"partial_bc_integral={partial:.8f}")
    print(f"partial_bc_ratio={partial / base:.6f}")
    print(f"bc_error_is_caught={abs(partial / base - 1.0) > 0.2}")
    print(f"failure_modes_warnings={sorted(set(m2 + m3 + m4))!r}")
    print(f"all_three_failure_modes_are_silent={not (m2 or m3 or m4)}")
    if abs(doubled / base - 2.0) >= 1e-9:
        print("FAIL: doubling the source did not double the integral",
              file=sys.stderr)
        ok = False
    if flipped / base >= -0.9:
        print("FAIL: flipping the sign of K did not flip the integral",
              file=sys.stderr)
        ok = False
    if abs(partial / base - 1.0) <= 0.2:
        print("FAIL: dropping a Dirichlet side barely changed the integral",
              file=sys.stderr)
        ok = False
    if m2 or m3 or m4:
        print(f"FAIL: a failure mode warned {sorted(set(m2 + m3 + m4))!r}",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
