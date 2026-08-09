"""Tier-2: elasticity vibration modes need eigsh(..., sigma=0).

Claim: skfem linear_elasticity#3 -- for vibration problems use
eigsh(K, M=M, k=n, sigma=0); omitting sigma returns the LARGEST eigenvalues,
unrelated to physical vibration modes, and it does so silently.

This fixture builds a genuine vector elasticity pencil -- a cantilever beam,
ElementVector(ElementQuad1()) stiffness from
skfem.models.elasticity.linear_elasticity plus a consistent vector mass matrix --
rather than reusing the scalar Laplacian, so the claim is tested on the physics
it is written about.

Wrong variant: eigsh(K_free, M=M_free, k=4) with no sigma and no which.

Note: skfem.models.poisson.mass does NOT work on a vector basis (it broadcasts
to the wrong shape), so the vector mass form is written out explicitly here.

Mutation control: ``T2_MUTATE=1 python source.py`` applies the documented fix at
the pathology site -- the no-sigma call gets ``sigma=0, which="LM"`` back, which
is exactly the edit the pitfall prescribes.  Both calls then return the same low
vibration modes, the top-of-spectrum contrast disappears and the fixture goes
red.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.sparse.linalg import eigsh
from skfem import (
    Basis,
    BilinearForm,
    ElementQuad1,
    ElementVector,
    MeshQuad,
    asm,
)
from skfem.helpers import dot
from skfem.models.elasticity import lame_parameters, linear_elasticity

MUTATE = os.environ.get("T2_MUTATE") == "1"


@BilinearForm
def vector_mass(u, v, w):
    return dot(u, v)


def main() -> int:
    ok = True
    m = MeshQuad.init_tensor(np.linspace(0.0, 4.0, 17),
                             np.linspace(0.0, 1.0, 5)).with_boundaries(
        {"clamped": lambda x: x[0] < 1e-10})
    basis = Basis(m, ElementVector(ElementQuad1()))
    lam, mu = lame_parameters(1.0, 0.3)
    K = asm(linear_elasticity(lam, mu), basis)
    M = asm(vector_mass, basis)
    free = basis.complement_dofs(basis.get_dofs("clamped"))
    KI = K[free][:, free]
    MI = M[free][:, free]
    print(f"basis_N={basis.N}")
    print(f"n_free_dofs={len(free)}")

    low = np.sort(eigsh(KI, M=MI, k=4, sigma=0, which="LM",
                        return_eigenvectors=False))

    # --- WRONG variant: no sigma ----------------------------------------
    # Under T2_MUTATE the documented fix is applied right here: sigma=0 and
    # which='LM' are supplied, so this call shift-and-inverts too.
    kw = {} if not MUTATE else {"sigma": 0, "which": "LM"}
    raised = ""
    try:
        high = np.sort(eigsh(KI, M=MI, k=4, return_eigenvectors=False, **kw))
    except Exception as exc:              # pragma: no cover
        raised = f"{type(exc).__name__}: {exc}"
        high = np.array([np.nan] * 4)
    print(f"no_sigma_raised_nothing={not raised}")
    if raised:
        print(f"FAIL: the no-sigma call raised {raised}", file=sys.stderr)
        ok = False

    top = bool(high.min() > low.max())
    ratio = float(high.min() / low.max())
    print(f"no_sigma_returns_top_of_spectrum={top}")
    print(f"no_sigma_min_over_sigma0_max_gt_100={ratio > 100.0}")
    if not (top and ratio > 100.0):
        print(f"FAIL: no-sigma eigenvalues {high} are not far above the "
              f"sigma=0 ones {low} (ratio {ratio:.3g})", file=sys.stderr)
        ok = False

    ascending = bool((np.diff(low) > 0).all()) and bool((low > 0).all())
    print(f"sigma0_modes_ascending={ascending}")
    print(f"sigma0_omega_all_positive={bool((np.sqrt(low) > 0).all())}")
    if not ascending:
        print(f"FAIL: sigma=0 modes are not a positive ascending sequence: "
              f"{low}", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
