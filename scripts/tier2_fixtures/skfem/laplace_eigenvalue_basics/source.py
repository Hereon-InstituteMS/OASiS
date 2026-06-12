"""Tier-2: skfem Dirichlet Laplace eigenvalues + complement_dofs split.

Pitfalls (skfem eigenvalue#0 + #2):
  - basis.complement_dofs(D) returns interior dofs; cardinality
    of D + I equals basis.N exactly.
  - scipy.sparse.linalg.eigsh on the interior-restricted block
    gives the first Dirichlet-Laplace eigenvalues within a few
    percent of the analytic pi^2*(m^2+n^2) sequence on a
    MeshTri.refined(4) P1 mesh.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import scipy.sparse.linalg
import skfem
from skfem.models.poisson import laplace, mass


def main() -> int:
    mesh = skfem.MeshTri().refined(4)
    basis = skfem.Basis(mesh, skfem.ElementTriP1())
    D = basis.get_dofs()
    I = basis.complement_dofs(D)
    split_ok = len(D.nodal["u"]) + len(I) == basis.N
    print(f"D_plus_I_equals_N={split_ok}")

    K = laplace.assemble(basis)
    M = mass.assemble(basis)
    K_I = K[I][:, I]
    M_I = M[I][:, I]
    eigs, _ = scipy.sparse.linalg.eigsh(
        K_I, M=M_I, k=3, sigma=0, which="LM")
    eigs = np.sort(np.real(eigs))
    analytic = sorted(
        math.pi ** 2 * (m * m + n * n)
        for m in range(1, 4) for n in range(1, 4)
    )[:3]
    rel_err_0 = abs(eigs[0] - analytic[0]) / analytic[0]
    print(f"eig0={eigs[0]:.4f}, analytic={analytic[0]:.4f}, "
          f"rel_err={rel_err_0:.4f}")
    print(f"eig0_rel_err_lt_0.05={rel_err_0 < 0.05}")

    if split_ok and rel_err_0 < 0.05:
        return 0
    print("ERROR: eigenvalue checks failed", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
