"""Tier-2: K x = omega^2 M x needs M; dropping it changes the problem silently.

Claim: skfem eigenvalue#3 (previously "claim inherited -- not yet empirically
separated") -- eigenvalues of the structural pencil are squared angular
frequencies, so omega = sqrt(lam). Passing both K and M solves the generalised
problem; passing only K solves the standard problem against the identity and
gives wrong frequencies.

Wrong variant: eigsh(K_I, k=5, sigma=0) with no M= argument. It returns without
raising or warning, and the numbers are not frequencies of anything -- they are
the eigenvalues of the stiffness matrix in the (mesh-dependent) nodal basis.

Mutation control: ``T2_MUTATE=1 python source.py`` applies the documented fix at
the pathology site -- the wrong-variant call gets its ``M=MI`` argument back, so
it solves the generalised pencil too.  The two answers then coincide and the
"dropping M= changes the answer" contrast disappears, so the fixture goes red.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.sparse.linalg import eigsh
from skfem import Basis, ElementTriP1, MeshTri
from skfem.models.poisson import laplace, mass

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    ok = True
    basis = Basis(MeshTri().refined(4), ElementTriP1())
    K = laplace.assemble(basis)
    M = mass.assemble(basis)
    interior = basis.complement_dofs(basis.get_dofs())
    KI = K[interior][:, interior]
    MI = M[interior][:, interior]

    generalised = np.sort(eigsh(KI, M=MI, k=5, sigma=0, which="LM",
                                return_eigenvectors=False))

    # --- WRONG variant: no M= -> standard pencil against the identity ---
    # Under T2_MUTATE the documented fix is applied right here: M=MI is
    # supplied, so this call solves the generalised pencil as well.
    kw = {} if not MUTATE else {"M": MI}
    raised = ""
    try:
        standard = np.sort(eigsh(KI, k=5, sigma=0, which="LM",
                                 return_eigenvectors=False, **kw))
    except Exception as exc:              # pragma: no cover - must not happen
        raised = f"{type(exc).__name__}: {exc}"
        standard = np.array([np.nan] * 5)
    print(f"standard_pencil_raised_nothing={not raised}")
    if raised:
        print(f"FAIL: the standard-pencil call raised {raised}", file=sys.stderr)
        ok = False

    analytic_lowest = 2.0 * np.pi ** 2
    lowest = float(generalised.min())
    bracketed = analytic_lowest < lowest < 1.10 * analytic_lowest
    print(f"generalised_lowest_in_analytic_bracket={bracketed}")
    if not bracketed:
        print(f"FAIL: generalised lowest eigenvalue {lowest!r} outside the P1 "
              f"bracket around {analytic_lowest!r}", file=sys.stderr)
        ok = False

    ratio = lowest / float(standard.min())
    far = ratio > 10.0
    print(f"standard_pencil_differs_by_over_10x={far}")
    print(f"standard_pencil_differs_by_over_100x={ratio > 100.0}")
    if not far:
        print(f"FAIL: dropping M= changed the answer by only a factor "
              f"{ratio:.3g}", file=sys.stderr)
        ok = False

    omega = np.sqrt(generalised)
    consistent = bool(np.allclose(omega ** 2, generalised, rtol=1e-12))
    print(f"omega_is_sqrt_of_eigenvalue={consistent}")
    if not consistent:
        print("FAIL: omega**2 does not reproduce the eigenvalue",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
