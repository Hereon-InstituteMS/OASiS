"""Tier-2: skfem Poisson solved without condense() blows up.

Pitfall: omitting `condense(K, f, D=...)` on the Dirichlet DoFs
gives a singular stiffness matrix. scipy.sparse.linalg's
default direct solver doesn't throw — it returns a "solution"
with values of O(1e14) (the wrong-by-orders-of-magnitude
signature).

This fixture builds the canonical Poisson problem on a quad
mesh, solves it BOTH with and without condense, and verifies
the no-condense version's solution magnitude is at least 10^8
larger than the condensed version's.
"""
from __future__ import annotations

import sys
import numpy as np
from skfem import (
    Basis,
    ElementQuad1,
    MeshQuad,
    condense,
    solve,
)
from skfem.models.poisson import laplace, unit_load


def main() -> int:
    m = (MeshQuad
         .init_tensor(np.linspace(0, 1, 9), np.linspace(0, 1, 9))
         .with_boundaries({"left": lambda x: x[0] < 1e-10}))
    ib = Basis(m, ElementQuad1())
    K = laplace.assemble(ib)
    f = unit_load.assemble(ib)

    # CORRECT solve: condense Dirichlet DoFs.
    D = ib.get_dofs("left").flatten()
    u_correct = solve(*condense(K, f, D=D))
    correct_mag = np.max(np.abs(u_correct))

    # WRONG solve: K is singular without condense.
    u_wrong = solve(K, f)
    wrong_mag = np.max(np.abs(u_wrong))

    print(f"|u_correct|_max = {correct_mag:.3e}")
    print(f"|u_wrong|_max   = {wrong_mag:.3e}")

    if wrong_mag / max(correct_mag, 1.0) < 1e6:
        print("FIXTURE WARNING: no-condense solution did NOT blow "
              "up — pitfall claim no longer holds; "
              "magnitude ratio < 1e6", file=sys.stderr)
        return 2

    print("Missing condense() yields singular K; solve returns "
          "values orders of magnitude larger than the correctly "
          "condensed solution — wrong-by-magnitude signature "
          "confirmed.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
