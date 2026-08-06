"""Tier-2: eigsh(sigma=0) targets the smallest eigenvalues; the default does not.

Claim: skfem eigenvalue#1 -- eigsh(K_I, M=M_I, k=5, sigma=0, which='LM') returns
the 5 SMALLEST eigenvalues via shift-and-invert; which='SM' without sigma gets
the same values more slowly.

Wrong variant: calling eigsh with neither sigma nor which. ARPACK then returns
the LARGEST magnitude eigenvalues of the pencil -- for a stiffness/mass pencil
that is the mesh-resolution end of the spectrum, unrelated to any physical
low-frequency mode, and nothing warns.

Assertions are orderings and agreements, never pinned measured floats.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.sparse.linalg import eigsh
from skfem import Basis, ElementTriP1, MeshTri
from skfem.models.poisson import laplace, mass


def main() -> int:
    ok = True
    m = MeshTri().refined(4)
    basis = Basis(m, ElementTriP1())
    K = laplace.assemble(basis)
    M = mass.assemble(basis)
    boundary = basis.get_dofs()
    interior = basis.complement_dofs(boundary)
    KI = K[interior][:, interior]
    MI = M[interior][:, interior]
    print(f"basis_N={basis.N}")
    print(f"n_interior={len(interior)}")

    shifted = np.sort(eigsh(KI, M=MI, k=5, sigma=0, which="LM",
                            return_eigenvectors=False))
    smallest = np.sort(eigsh(KI, M=MI, k=5, which="SM",
                             return_eigenvectors=False))
    # --- WRONG variant: no sigma, no which ------------------------------
    default = np.sort(eigsh(KI, M=MI, k=5, return_eigenvectors=False))

    agree = bool(np.allclose(shifted, smallest, rtol=1e-8))
    print(f"sigma0_and_SM_agree={agree}")
    if not agree:
        print(f"FAIL: sigma=0 gave {shifted} but which='SM' gave {smallest}",
              file=sys.stderr)
        ok = False

    top = bool(default.min() > shifted.max())
    print(f"default_returns_top_of_spectrum={top}")
    print(f"default_min_exceeds_sigma0_max={top}")
    print(f"default_min_over_sigma0_max_gt_10={default.min() > 10 * shifted.max()}")
    if not top:
        print(f"FAIL: the default eigsh call did not return the top of the "
              f"spectrum: default {default}, sigma=0 {shifted}",
              file=sys.stderr)
        ok = False

    # The lowest shift-and-invert eigenvalue must be the fundamental mode:
    # bracket it against the analytic 2*pi^2 for the unit square, loosely,
    # so nothing measured is pinned.
    lowest = float(shifted.min())
    analytic = 2.0 * np.pi ** 2
    bracketed = analytic < lowest < 1.10 * analytic
    print(f"sigma0_smallest_is_lowest_mode={bracketed}")
    if not bracketed:
        print(f"FAIL: lowest eigenvalue {lowest!r} not in the P1 bracket "
              f"({analytic!r}, {1.10 * analytic!r})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
