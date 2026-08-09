"""Tier-2: eigsh on the non-Hermitian Helmholtz-with-ABC system is SILENTLY wrong.

Claim: skfem helmholtz#5 — a Helmholtz system with an absorbing BC is
non-Hermitian, and ``scipy.sparse.linalg.eigsh`` assumes Hermitian: it gives
wrong numbers WITHOUT raising.  On ``MeshTri().refined(2)`` / ``ElementTriP1``,
taking the P1 stiffness matrix, casting to complex and adding ``5j`` to a single
off-diagonal entry, ``eigsh(A, k=3, M=M, sigma=0)`` returns a corrupted lowest
eigenvalue with no warning; there is no ``ArpackNoConvergence`` and no
"non-Hermitian" diagnostic.  Use ``eigs`` (general non-Hermitian ARPACK) or a
direct/GMRES forward solve instead.

Wrong variant: this fixture calls ``eigsh`` on exactly that matrix inside
``warnings.catch_warnings(record=True)`` and counts what came back.  The truth
is established twice independently — sparse ``eigs`` and dense
``scipy.linalg.eig`` — and the fixture shows eigsh's lowest eigenvalue is the
bare real part of a genuinely complex eigenvalue, i.e. an imaginary part of
several units is discarded silently.  ``eigsh`` on the untouched Hermitian
stiffness matrix is also run and confirmed accurate, so a change in either
direction is caught.  Measured floats are not pinned; only orderings,
thresholds, dtypes and exact combinatorial counts are asserted.

Observed on skfem 12.0.1 / scipy 1.15.3 (2026-08-06).

Mutation control: with T2_MUTATE=1 the wrong variant's solver call switches
from ``sla.eigsh`` to ``sla.eigs`` on the identical non-Hermitian pencil — the
documented fix (use the general non-Hermitian ARPACK driver) applied at the
pathology site.  The returned spectrum is then complex and correct, so
'eigsh_evals_dtype=float64', 'eigsh_returns_real_dtype=True',
'eigsh_lowest_error_gt_1=True' and 'eigsh_lowest_is_bare_real_part=True'
disappear and the fixture goes red.  The RIGHT variant further down (eigsh on
the untouched Hermitian matrix) is not the pathology and is left alone.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import scipy.linalg as dla
import scipy.sparse.linalg as sla
from skfem import Basis, ElementTriP1, MeshTri
from skfem.models.poisson import laplace, mass

MUTATE = os.environ.get("T2_MUTATE") == "1"
# the mistake: the Hermitian driver on a non-Hermitian pencil; the documented
# fix is the general driver sla.eigs
HERMITIAN_DRIVER = sla.eigsh if not MUTATE else sla.eigs

PERTURB = 5j  # breaks Hermitian symmetry of entry (0, 1)


def main() -> int:
    ok = True

    m = MeshTri().refined(2)
    b = Basis(m, ElementTriP1())
    K = laplace.assemble(b)
    M = mass.assemble(b)
    print(f"mesh_class={type(m).__name__} element_class=ElementTriP1")
    print(f"n_dofs={b.N} stiffness_nnz={K.nnz}")

    A = K.astype(complex).tolil()
    A[0, 1] += PERTURB          # single off-diagonal entry -> non-Hermitian
    A = A.tocsr()
    herm_gap = float(abs(A - A.getH()).max())
    print(f"hermitian_gap={herm_gap:.1f}")
    print(f"matrix_is_non_hermitian={herm_gap > 0.0}")
    if herm_gap <= 0.0:
        print("FAIL: the perturbed matrix is still Hermitian, so the wrong "
              "code path was never entered", file=sys.stderr)
        ok = False

    # --- WRONG variant: eigsh on a non-Hermitian matrix ---------------------
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            ev_sh = np.sort(HERMITIAN_DRIVER(A, k=3, M=M, sigma=0.0)[0])
            sh_exc = ""
        except Exception as exc:                 # noqa: BLE001 - want the type
            ev_sh, sh_exc = np.zeros(3), f"{type(exc).__name__}: {exc}"
        warn_texts = [f"{w.category.__name__}: {w.message}" for w in rec]
    print(f"eigsh_raised={bool(sh_exc)}")
    print(f"eigsh_exception_text={sh_exc}")
    print(f"eigsh_n_warnings={len(warn_texts)}")
    print(f"eigsh_warnings={warn_texts}")
    print(f"eigsh_evals_dtype={ev_sh.dtype}")
    print(f"eigsh_returns_real_dtype={ev_sh.dtype == np.float64}")
    arpack_text = (sh_exc + " ".join(warn_texts)).lower()
    print("arpack_non_hermitian_diagnostic_absent="
          f"{'arpacknoconvergence' not in arpack_text and 'non-hermitian' not in arpack_text}")
    if sh_exc or warn_texts:
        print("FAIL: eigsh was loud about the non-Hermitian matrix; the claim "
              "says the failure is silent", file=sys.stderr)
        ok = False

    # --- Truth, established twice ------------------------------------------
    ev_g = sla.eigs(A, k=3, M=M, sigma=0.0)[0]
    ev_g = ev_g[np.argsort(np.abs(ev_g))]
    w_dense = dla.eig(A.toarray(), M.toarray(), right=False)
    w_dense = w_dense[np.argsort(np.abs(w_dense))][:3]
    print(f"eigs_evals_dtype={ev_g.dtype}")
    agree = float(np.abs(np.sort_complex(ev_g)
                         - np.sort_complex(w_dense)).max())
    print(f"eigs_matches_dense_reference={agree < 1e-6}")
    if not agree < 1e-6:
        print("FAIL: sparse eigs and dense eig disagree, so the reference "
              "spectrum is not trustworthy", file=sys.stderr)
        ok = False

    lowest_true = ev_g[np.argmin(np.abs(ev_g))]
    print(f"true_lowest_imag_abs_gt_1={abs(lowest_true.imag) > 1.0}")
    if not abs(lowest_true.imag) > 1.0:
        print("FAIL: the true lowest eigenvalue is essentially real, so "
              "eigsh has nothing to discard", file=sys.stderr)
        ok = False

    lowest_sh = ev_sh[np.argmin(np.abs(ev_sh))]
    err = abs(complex(lowest_sh) - lowest_true)
    print(f"eigsh_lowest_error_gt_1={err > 1.0}")
    print(f"eigsh_lowest_is_bare_real_part={abs(lowest_sh - lowest_true.real) < 1e-6}")
    if not err > 1.0:
        print("FAIL: eigsh agreed with the true spectrum, so the silent "
              "corruption did not occur", file=sys.stderr)
        ok = False

    # the two upper eigenvalues happen to survive -> the damage is selective,
    # which is exactly why it is easy to miss.
    upper_sh = np.sort(ev_sh[np.argsort(np.abs(ev_sh))][1:])
    upper_true = np.sort(np.real(ev_g[np.argsort(np.abs(ev_g))][1:]))
    print("eigsh_upper_pair_still_agrees="
          f"{float(np.abs(upper_sh - upper_true).max()) < 1e-6}")

    # --- RIGHT variant: eigsh IS accurate on the Hermitian matrix ----------
    ev_ref = np.sort(sla.eigsh(K, k=3, M=M, sigma=0.0)[0])
    dense_ref = np.sort(dla.eigh(K.toarray(), M.toarray(),
                                 eigvals_only=True)[:3])
    ref_ok = float(np.abs(ev_ref - dense_ref).max()) < 1e-8
    print(f"eigsh_accurate_on_hermitian={ref_ok}")
    print(f"hermitian_lowest_is_near_zero={abs(ev_ref[0]) < 1e-10}")
    if not ref_ok:
        print("FAIL: eigsh is inaccurate even on the Hermitian matrix, so the "
              "comparison above proves nothing", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
