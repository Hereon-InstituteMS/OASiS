"""Tier-2: explicit Euler needs M inverted, and skfem's M is not diagonal.

Claim: skfem time_dependent#7 -- for explicit time stepping M du/dt = -K u + f
(avoid for stiff problems).  "Explicit Euler with dt > 2/lambda_max blows up
to NaN within a few steps; energy grows geometrically; warning: the standard
FEM mass matrix M (BilinearForm-asm'd ElementTriP1 mass) is NOT diagonal --
applying explicit Euler naively requires inverting M at each step (lumped mass
via condense + diagonal approximation is one workaround)."

Measured on skfem 12.0.1, u_t = Laplace u on a 12x12 MeshTri.init_tensor with
ElementTriP1, homogeneous Dirichlet:

  * "M IS NOT DIAGONAL" IS CONFIRMED STRUCTURALLY: the assembled mass matrix
    carries several times more non-zeros than its diagonal, and its largest
    off-diagonal entry is a substantial fraction of its smallest diagonal
    entry.  Row-sum lumping recovers a matrix with exactly N non-zeros, and
    both matrices have the same total mass (row sums add to the domain area).
  * "BLOWS UP TO NaN WITHIN A FEW STEPS" IS WRONG ABOUT THE TIMING.  At
    dt = 1.5 * (2/lambda_max) the amplitude is still BELOW its starting value
    after five steps -- a short run looks like ordinary decay.  NaN arrives
    only after roughly a thousand steps, once the unstable mode has grown
    through the whole float range.
  * WHAT IS EXACT AND CHECKABLE is the growth factor: once the stiffest mode
    dominates, the peak multiplies by |1 - dt*lambda_max| every step, which
    the fixture confirms against the analytic value to better than 1e-4 over
    fifteen consecutive steps.  Watch the ratio, not the step count.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from skfem import Basis, BilinearForm, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace

NX = 12


@BilinearForm
def mass_form(u, v, w):
    return u * v


def build():
    m = MeshTri.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                            np.linspace(0.0, 1.0, NX + 1))
    ib = Basis(m, ElementTriP1())
    return m, ib, laplace.assemble(ib), mass_form.assemble(ib), \
        ib.get_dofs().all()


def explicit(dt, nsteps, lumped=False):
    m, ib, K, M, D = build()
    if lumped:
        diag = np.asarray(M.sum(axis=1)).ravel()
        Muse = sp.diags(diag).tocsr()
    else:
        Muse = M.tocsr()
    u = ib.project(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    u[D] = 0.0
    peaks = [float(np.abs(u).max())]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(nsteps):
            u = solve(*condense(Muse, Muse @ u - dt * (K @ u), D=D))
            peaks.append(float(np.abs(u).max()))
            if not np.isfinite(peaks[-1]):
                break
        msgs = sorted({str(c.message) for c in caught})
    return u, peaks, msgs


def main() -> int:
    ok = True
    m, ib, K, M, D = build()
    print(f"dofs_N={ib.N} elements={m.t.shape[1]}")

    # --- M is not diagonal ----------------------------------------------
    Md = M.toarray()
    offdiag = Md - np.diag(np.diag(Md))
    print(f"mass_nnz={M.nnz} diagonal_entries={M.shape[0]}")
    print(f"mass_nnz_over_N={M.nnz / M.shape[0]:.2f}")
    print(f"mass_has_offdiagonal_nonzeros={int(np.count_nonzero(offdiag))}")
    print(f"mass_is_diagonal={bool(np.count_nonzero(offdiag) == 0)}")
    print(f"max_offdiag_over_min_diag="
          f"{float(np.abs(offdiag).max() / np.diag(Md).min()):.4f}")
    if np.count_nonzero(offdiag) == 0:
        print("FAIL: the assembled mass matrix WAS diagonal", file=sys.stderr)
        ok = False

    lump = np.asarray(M.sum(axis=1)).ravel()
    L = sp.diags(lump).tocsr()
    print(f"lumped_nnz={L.nnz} equals_N={L.nnz == M.shape[0]}")
    print(f"total_mass_consistent={float(M.sum()):.6f}")
    print(f"total_mass_lumped={float(L.sum()):.6f}")
    print(f"lumping_conserves_total_mass="
          f"{abs(float(M.sum()) - float(L.sum())) < 1e-12}")
    if L.nnz != M.shape[0]:
        print("FAIL: row-sum lumping did not give exactly N non-zeros",
              file=sys.stderr)
        ok = False

    # --- the explicit bound ----------------------------------------------
    free = np.setdiff1d(np.arange(ib.N), D)
    ev = sla.eigh(K[free][:, free].toarray(), M[free][:, free].toarray(),
                  eigvals_only=True)
    dt_crit = 2.0 / float(ev.max())
    print(f"lambda_max={float(ev.max()):.4e} dt_crit={dt_crit:.4e}")

    _, peaks_safe, _ = explicit(0.5 * dt_crit, 200)
    print(f"safe_dt_final_peak={peaks_safe[-1]:.4e}")
    print(f"safe_dt_decays={peaks_safe[-1] < peaks_safe[0]}")
    if peaks_safe[-1] >= peaks_safe[0]:
        print("FAIL: a dt below the bound did not decay", file=sys.stderr)
        ok = False

    u_bad, peaks_bad, msgs_bad = explicit(1.5 * dt_crit, 1200)
    finite = [p for p in peaks_bad if np.isfinite(p)]
    # the unstable eigenmode needs time to dominate, so the asymptotic growth
    # factor has to be read off a LATE window
    ratios = [finite[i + 1] / finite[i] for i in range(200, 215)
              if i + 1 < len(finite) and finite[i] > 0]
    analytic = abs(1.0 - 1.5 * dt_crit * float(ev.max()))
    print(f"unstable_steps_taken={len(peaks_bad) - 1}")
    print(f"unstable_initial_peak={peaks_bad[0]:.4e}")
    print(f"unstable_peak_after_5_steps={peaks_bad[5]:.4e}")
    print(f"unstable_still_below_initial_after_5_steps="
          f"{peaks_bad[5] < peaks_bad[0]}")
    print(f"unstable_growth_ratios={[f'{r:.6f}' for r in ratios]}")
    spread = (max(ratios) - min(ratios)) if ratios else 1.0
    print(f"unstable_growth_is_geometric={spread < 1e-4}")
    print(f"unstable_growth_factor={ratios[0]:.6f}")
    print(f"analytic_amplification_abs_1_minus_dt_lambda={analytic:.6f}")
    print(f"growth_factor_matches_analytic="
          f"{abs(ratios[0] - analytic) < 1e-4}")
    print(f"steps_until_non_finite={len(finite)}")
    print(f"nan_within_a_few_steps={len(finite) <= 20}")
    print(f"non_finite_value_is_nan={bool(np.isnan(peaks_bad[-1]))}")
    print(f"last_finite_peak={finite[-1]:.4e}")
    print(f"unstable_warnings={msgs_bad!r}")
    if spread >= 1e-4:
        print("FAIL: the unstable growth was not geometric", file=sys.stderr)
        ok = False
    if abs(ratios[0] - analytic) >= 1e-4:
        print("FAIL: the growth factor did not match |1 - dt*lambda_max|",
              file=sys.stderr)
        ok = False
    if len(finite) <= 20:
        print(f"FAIL: the run went non-finite after only {len(finite)} steps, "
              f"so the entry's 'within a few steps' holds after all",
              file=sys.stderr)
        ok = False
    if not np.isnan(peaks_bad[-1]):
        print("FAIL: the run never reached NaN", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
