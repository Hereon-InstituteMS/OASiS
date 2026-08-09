"""Tier-2: a Dirac source does not converge in H^1, only in L^2 and at rate 1.

Claim: skfem point_source#0 -- a Dirac source is not smooth, so the FE solution
does NOT converge in the H^1 norm as h -> 0; the H^1 error stays O(1) while the
L^2 error converges only at rate O(h).  Signal: a refinement study shows the
H^1-norm error FLAT or non-monotone as h is halved while the L^2 error merely
halves.

Measured on skfem 12.0.1: unit source at the centre node, ElementTriP1 on
MeshTri().refined(r) for r = 3..6, errors taken against an r = 8 reference
projected onto each mesh.  Both halves of the claim reproduce -- the L^2 error
falls by roughly a factor two per halving (order about one, not two) while the
H^1 seminorm error is essentially flat across an eightfold refinement.

The fixture adds the structural fingerprint the entry does not mention: the
pointwise PEAK of the discrete solution GROWS without bound under refinement
(the Green's function has a logarithmic singularity), so a check on max(u) is
not a convergence check -- it is a divergence check.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology
site -- the Dirac right-hand side f[src] = 1.0 is replaced by the smooth
distributed load unit_load.assemble(ib) (the integral of f = 1 against the
test functions), in the reference run and in every refinement alike.  The
regularity is then restored (L2 rate 1.94/2.00/2.07, H1 rate 1.85/1.89/1.97,
peak growth factor 1.012) and 'l2_rate_is_below_two=True',
'h1_rate_is_below_one=True', 'h1_barely_moves_while_l2_falls=True' and
'peak_is_not_converging=True' disappear from the output.  Note that
'peak_grows_monotonically_under_refinement=True' SURVIVES the mutation: the
smooth-source peak also rises monotonically, it just converges, so monotone
growth on its own is not evidence of a divergent peak.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, mass, unit_load

MUTATE = os.environ.get("T2_MUTATE") == "1"


def solve_point(refine):
    m = MeshTri().refined(refine)
    ib = Basis(m, ElementTriP1())
    K = laplace.assemble(ib)
    D = ib.get_dofs().all()
    if MUTATE:
        # the documented fix: a smooth source instead of a point mass
        f = unit_load.assemble(ib)
    else:
        f = ib.zeros()
        src = int(np.argmin((ib.doflocs[0] - 0.5) ** 2
                            + (ib.doflocs[1] - 0.5) ** 2))
        f[src] = 1.0
    return m, ib, solve(*condense(K, f, D=D))


def main() -> int:
    ok = True
    m_ref, ib_ref, u_ref = solve_point(8)
    interp = ib_ref.interpolator(u_ref)
    print(f"reference_refine=8 reference_N={ib_ref.N}")

    l2, h1, peaks, dofs = [], [], [], []
    for r in (3, 4, 5, 6):
        m, ib, u = solve_point(r)
        K = laplace.assemble(ib)
        M = mass.assemble(ib)
        # project the reference onto this mesh by evaluating it at the DOFs
        u_ex = interp(ib.doflocs)
        d = u - u_ex
        l2.append(float(np.sqrt(d @ (M @ d))))
        h1.append(float(np.sqrt(abs(d @ (K @ d)))))
        peaks.append(float(u.max()))
        dofs.append(ib.N)
        print(f"r{r}_N={ib.N} l2={l2[-1]:.4e} h1={h1[-1]:.4e} "
              f"peak={peaks[-1]:.4f}")

    l2_rates = [float(np.log2(l2[i] / l2[i + 1])) for i in range(len(l2) - 1)]
    h1_rates = [float(np.log2(h1[i] / h1[i + 1])) for i in range(len(h1) - 1)]
    print(f"l2_rates={[f'{r:.3f}' for r in l2_rates]}")
    print(f"h1_rates={[f'{r:.3f}' for r in h1_rates]}")
    print(f"l2_rate_is_below_two={all(r < 1.8 for r in l2_rates)}")
    print(f"h1_rate_is_below_one={all(r < 0.9 for r in h1_rates)}")
    print(f"h1_rate_is_below_l2_rate="
          f"{all(h1_rates[i] < l2_rates[i] for i in range(len(h1_rates)))}")
    print(f"l2_total_reduction_over_8x_refinement={l2[0] / l2[-1]:.3f}")
    print(f"h1_total_reduction_over_8x_refinement={h1[0] / h1[-1]:.3f}")
    print(f"h1_barely_moves_while_l2_falls="
          f"{(l2[0] / l2[-1]) > 5.0 * (h1[0] / h1[-1])}")
    print(f"h1_error_is_exactly_flat={all(abs(r) < 0.1 for r in h1_rates)}")
    if not all(r < 1.8 for r in l2_rates):
        print("FAIL: the L2 error converged at the optimal order two, so the "
              "Dirac source is not spoiling the regularity", file=sys.stderr)
        ok = False
    if not all(r < 0.9 for r in h1_rates):
        print("FAIL: the H1 error converged at order one or better",
              file=sys.stderr)
        ok = False
    if not (l2[0] / l2[-1]) > 5.0 * (h1[0] / h1[-1]):
        print("FAIL: the H1 error kept pace with the L2 error",
              file=sys.stderr)
        ok = False

    # --- the peak diverges, so max(u) is not a convergence check --------
    print(f"peaks={[f'{p:.4f}' for p in peaks]}")
    growing = all(peaks[i + 1] > peaks[i] for i in range(len(peaks) - 1))
    print(f"peak_grows_monotonically_under_refinement={growing}")
    print(f"peak_growth_factor={peaks[-1] / peaks[0]:.3f}")
    print(f"peak_is_not_converging={growing and peaks[-1] > 1.3 * peaks[0]}")
    if not growing:
        print("FAIL: the pointwise peak did not grow under refinement",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
