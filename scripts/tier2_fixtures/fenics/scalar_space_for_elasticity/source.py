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

Mutation control: T2_MUTATE=1 builds the space the catalog actually
recommends, ('Lagrange', 1, (gdim,)). sym(grad(u)) is then a legal
rank-2 expression, nothing raises, the fixture prints the strain
shape instead of a traceback, and both expected strings ('ValueError',
'Symmetric part of tensor with rank != 2') disappear.
"""
from __future__ import annotations

import os
import sys
import traceback

import dolfinx
import ufl
from mpi4py import MPI

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    gdim = mesh.geometry.dim
    # SCALAR space — bug.  MUTATE: the vector space it should have been.
    spec = ("Lagrange", 1, (gdim,)) if MUTATE else ("Lagrange", 1)
    V = dolfinx.fem.functionspace(mesh, spec)
    u = ufl.TrialFunction(V)
    print(f"space_spec={spec} trial_shape={u.ufl_shape}")
    try:
        # Elasticity strain = sym(grad(u)) requires vector u.
        eps = ufl.sym(ufl.grad(u))
    except Exception:
        traceback.print_exc()
        return 1
    print(f"sym_grad_accepted=True strain_shape={eps.ufl_shape}")
    if MUTATE:
        return 0
    print("ERROR: ufl.sym accepted grad of scalar Function "
          "(catalog claim wrong)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
