"""Tier-2: row-sum lumping costs a constant factor, not a convergence order.

Claim: skfem wave#2 -- HRZ-style lumping via scipy.sparse.csr_matrix.sum(axis=1)
introduces a dispersion error ~O(h^2); "amplitude at T_end drifts from analytic
value cos(c*pi*sqrt(2)*T) by > 1% even with small dt; spsolve of the
consistent-mass system shows refinement convergence at O(h^2) while the lumped
form stalls at O(h^1.5)."

Measured on skfem 12.0.1 by taking the fundamental frequency straight out of
the generalised eigenproblem K phi = lambda M phi on uniform
MeshQuad.init_tensor grids at nx = 6, 11, 21, 41 with ElementQuad1, so the
result carries no time-stepping error at all:

  * "THE LUMPED FORM STALLS AT O(h^1.5)" IS FALSE.  Both mass matrices give a
    frequency error converging at rate 2.0 -- consistent 2.004/2.001/2.000,
    lumped 1.974/1.994/1.998.  Lumping does not change the ORDER.
  * WHAT IT DOES CHANGE IS THE CONSTANT AND THE SIGN.  The lumped frequency
    error is close to three times the consistent one at every refinement
    level, and it points the other way: the consistent Q1 mass OVERESTIMATES
    the fundamental frequency while row-sum lumping UNDERESTIMATES it.  A
    check that only looks at the magnitude of the frequency error cannot
    tell which mass matrix it is running on.
  * the ">1% amplitude drift" is a consequence of that frequency error and is
    therefore mesh-dependent: at the coarsest grid the lumped relative
    frequency error is several percent and at the finest it is well under a
    tenth of a percent, so the 1% figure is not a scheme property.
"""
from __future__ import annotations

import os

# Pin the BLAS thread count BEFORE numpy is imported.  The dense generalised
# eigensolves here are small enough that thread oversubscription costs far
# more than it saves, and a fixture must not depend on how busy the host is.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys

import numpy as np
import scipy.linalg as sla
from skfem import Basis, ElementQuad1, MeshQuad
from skfem.models.poisson import laplace, mass

EXACT_OMEGA = np.pi * np.sqrt(2.0)
GRIDS = (6, 11, 21, 41)


def frequency(nx, lumped):
    m = MeshQuad.init_tensor(np.linspace(0.0, 1.0, nx),
                             np.linspace(0.0, 1.0, nx))
    ib = Basis(m, ElementQuad1())
    K = laplace.assemble(ib).tocsr()
    M = mass.assemble(ib).tocsr()
    D = ib.get_dofs().all()
    free = np.setdiff1d(np.arange(ib.N), D)
    Kf = K[free][:, free].toarray()
    if lumped:
        Mf = np.diag(np.asarray(M.sum(axis=1)).ravel()[free])
    else:
        Mf = M[free][:, free].toarray()
    # only the fundamental is needed; asking for the whole spectrum on the
    # finest grid dominates the run time
    lam = sla.eigh(Kf, Mf, eigvals_only=True, subset_by_index=[0, 0])
    return float(np.sqrt(lam[0]))


def main() -> int:
    ok = True
    print(f"exact_omega={EXACT_OMEGA:.6f}")
    errs = {}
    over = {}
    for tag, lumped in (("consistent", False), ("lumped", True)):
        series = []
        signs = []
        for nx in GRIDS:
            w = frequency(nx, lumped)
            rel = abs(w - EXACT_OMEGA) / EXACT_OMEGA
            series.append(rel)
            signs.append(w < EXACT_OMEGA)
            print(f"{tag}_nx{nx}_omega={w:.6f} rel_error={rel:.4e} "
                  f"too_low={w < EXACT_OMEGA}")
        hs = [1.0 / (nx - 1) for nx in GRIDS]
        rates = [float(np.log(series[i] / series[i + 1])
                       / np.log(hs[i] / hs[i + 1]))
                 for i in range(len(series) - 1)]
        errs[tag] = (series, rates)
        over[tag] = all(signs) if lumped else not any(signs)
        print(f"{tag}_rates={[f'{r:.3f}' for r in rates]}")
        print(f"{tag}_order_is_two={all(1.9 < r < 2.1 for r in rates)}")

    cons, cons_r = errs["consistent"]
    lump, lump_r = errs["lumped"]
    print(f"both_converge_at_order_two="
          f"{all(1.9 < r < 2.1 for r in cons_r + lump_r)}")
    print(f"lumped_stalls_at_order_1p5="
          f"{all(1.35 < r < 1.65 for r in lump_r)}")
    if not all(1.9 < r < 2.1 for r in cons_r):
        print("FAIL: the consistent mass did not converge at order 2",
              file=sys.stderr)
        ok = False
    if not all(1.9 < r < 2.1 for r in lump_r):
        print("FAIL: the lumped mass did not converge at order 2 -- the "
              "entry's O(h^1.5) stall may be real after all", file=sys.stderr)
        ok = False

    print(f"consistent_overestimates_frequency={over['consistent']}")
    print(f"lumped_underestimates_frequency={over['lumped']}")
    print(f"lumping_flips_the_sign_of_the_error="
          f"{over['consistent'] and over['lumped']}")
    if not (over["consistent"] and over["lumped"]):
        print("FAIL: the two mass matrices did not err in opposite "
              "directions", file=sys.stderr)
        ok = False

    ratios = [lump[i] / cons[i] for i in range(len(GRIDS))]
    print(f"lumped_over_consistent_error_ratios="
          f"{[f'{r:.3f}' for r in ratios]}")
    print(f"lumping_costs_a_constant_factor_near_3="
          f"{all(2.5 < r < 3.5 for r in ratios)}")
    print(f"lumping_penalty_ratio_is_mesh_independent="
          f"{max(ratios) - min(ratios) < 0.2}")
    if not all(2.5 < r < 3.5 for r in ratios):
        print("FAIL: the lumping penalty was not a constant factor",
              file=sys.stderr)
        ok = False

    print(f"coarse_lumped_error_exceeds_one_percent={lump[0] > 0.01}")
    print(f"fine_lumped_error_below_one_percent={lump[-1] < 0.01}")
    print(f"one_percent_drift_is_mesh_dependent="
          f"{lump[0] > 0.01 and lump[-1] < 0.01}")
    if not (lump[0] > 0.01 and lump[-1] < 0.01):
        print("FAIL: the >1% drift did not depend on the mesh",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
