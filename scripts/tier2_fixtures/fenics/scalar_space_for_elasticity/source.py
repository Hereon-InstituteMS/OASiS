"""Tier-2: dolfinx scalar FunctionSpace for elasticity weak form.

Pitfall (fenics linear_elasticity#0): Vector function space for
elasticity must be created with ('Lagrange', 1, (gdim,)) — the
trailing shape tuple marks it vector-valued. Passing the plain
('Lagrange', 1) gives a SCALAR space. The elasticity weak form
inner(sigma(u), epsilon(v)) fails at FORM CONSTRUCTION time
when ufl.sym(ufl.grad(u)) is invoked on the scalar trial
function:

  ValueError: Symmetric part of tensor with rank != 2 is
  undefined.
"""
from __future__ import annotations

import sys
import traceback

import dolfinx
import ufl
from mpi4py import MPI


def main() -> int:
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    # SCALAR space — bug.
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    try:
        # Elasticity strain = sym(grad(u)) requires vector u.
        ufl.sym(ufl.grad(u))
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: ufl.sym accepted grad of scalar Function "
          "(catalog claim wrong)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
