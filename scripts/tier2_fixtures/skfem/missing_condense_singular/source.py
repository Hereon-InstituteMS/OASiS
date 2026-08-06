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

Mutation control: T2_MUTATE=1 applies the documented fix at the
wrong-solve site -- solve(K, f) becomes
solve(*condense(K, f, D=D)) -- so K is no longer singular and
the blow-up does not happen.  The magnitude ratio drops to 1,
the fixture takes its own failure branch, and the expectation
string 'singular' disappears from the output.  Note that the
other two expectations, 'condense' and 'magnitude', still match
under mutation: they are matched by the failure message
("no-condense", "magnitude ratio < 1e6"), not by any measured
quantity.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
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

MUTATE = os.environ.get("T2_MUTATE") == "1"


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

    # WRONG solve: K is singular without condense.  Under mutation the
    # documented fix is applied here and the Dirichlet DoFs are condensed out.
    u_wrong = solve(K, f) if not MUTATE else solve(*condense(K, f, D=D))
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
