"""Tier-2: analytic Dirichlet-Laplacian spectrum on [0,1]^2, and how P1 misses it.

Claim: skfem eigenvalue#2 -- the analytic eigenvalues are pi^2*(m^2+n^2); a
MeshTri refined(4) P1 mesh recovers the first five within ~1-4%, and the error
GROWS up the spectrum, so a 5% bar on the fifth eigenvalue is much tighter than
on the first.

Wrong variant: judging the discretisation on a coarse mesh, or assuming a single
tolerance applies uniformly across the computed modes. This fixture measures the
per-mode relative error on refined(2) and refined(4) and asserts the qualitative
structure -- one-sided (discrete above analytic), monotone growth up the
spectrum, degenerate second/third pair, coarse worse than fine. No measured float
is pinned; only orderings and a 5% bar.

Mutation control: ``T2_MUTATE=1 python source.py`` applies the documented fix at
the pathology site -- the coarse comparison spectrum is computed on
``spectrum(4)`` instead of ``spectrum(2)``, i.e. the under-resolved mesh is
refined away.  The coarse/fine error gap then no longer exists and the fixture
goes red.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.sparse.linalg import eigsh
from skfem import Basis, ElementTriP1, MeshTri
from skfem.models.poisson import laplace, mass

MUTATE = os.environ.get("T2_MUTATE") == "1"


def spectrum(refine: int, k: int = 5) -> np.ndarray:
    basis = Basis(MeshTri().refined(refine), ElementTriP1())
    K = laplace.assemble(basis)
    M = mass.assemble(basis)
    interior = basis.complement_dofs(basis.get_dofs())
    return np.sort(eigsh(K[interior][:, interior], M=M[interior][:, interior],
                         k=k, sigma=0, which="LM", return_eigenvectors=False))


def main() -> int:
    ok = True
    analytic = np.sort(np.array(
        [np.pi ** 2 * (mm ** 2 + nn ** 2)
         for mm in range(1, 5) for nn in range(1, 5)]))[:5]

    fine = spectrum(4)
    # The under-resolved mesh.  Under T2_MUTATE the documented fix is applied
    # at this site: refine it to the same level as `fine`.
    coarse = spectrum(2) if not MUTATE else spectrum(4)
    rel_fine = np.abs(fine - analytic) / analytic
    rel_coarse = np.abs(coarse - analytic) / analytic
    for i, r in enumerate(rel_fine):
        print(f"refined4_mode{i}_relerr_below_5pct={r < 0.05}")

    within = bool((rel_fine < 0.05).all())
    print(f"all_within_5pct={within}")
    if not within:
        print(f"FAIL: refined(4) P1 relative errors {rel_fine} exceed the 5% "
              f"bar", file=sys.stderr)
        ok = False

    above = bool((fine > analytic).all())
    print(f"all_discrete_above_analytic={above}")
    if not above:
        print(f"FAIL: a discrete eigenvalue fell below its analytic value: "
              f"{fine} vs {analytic}", file=sys.stderr)
        ok = False

    grows = bool(rel_fine[-1] > rel_fine[0])
    print(f"error_grows_up_the_spectrum={grows}")
    if not grows:
        print(f"FAIL: relative error does not grow up the spectrum: {rel_fine}",
              file=sys.stderr)
        ok = False

    degenerate = bool(abs(analytic[1] - analytic[2]) < 1e-12)
    print(f"second_third_degenerate={degenerate}")
    # The discrete pair splits the exact degeneracy -- that is the P1 artefact.
    split = abs(fine[2] - fine[1]) / analytic[1]
    print(f"discrete_degeneracy_split_nonzero={split > 1e-6}")
    if not degenerate:
        print("FAIL: analytic second and third eigenvalue are not degenerate",
              file=sys.stderr)
        ok = False

    # --- WRONG variant: coarse mesh -------------------------------------
    worse = bool((rel_coarse > rel_fine).all())
    print(f"coarse_mesh_error_larger={worse}")
    print(f"coarse_exceeds_5pct_somewhere={(rel_coarse > 0.05).any()}")
    if not worse:
        print(f"FAIL: refined(2) is not uniformly worse than refined(4): "
              f"{rel_coarse} vs {rel_fine}", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
