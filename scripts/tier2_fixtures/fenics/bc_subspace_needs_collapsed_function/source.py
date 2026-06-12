"""Tier-2: dolfinx dirichletbc on a sub-space needs collapsed Function.

Pitfall (fenics navier_stokes#3): For a Taylor-Hood
mixed_element([P2-vector, P1-scalar]) FunctionSpace, applying
a Dirichlet BC to the velocity sub-space requires:

  - boundary DOFs from locate_dofs_geometrical((V_sub,
    V_collapsed), is_boundary)
  - the value as a Function defined on V_collapsed (the
    collapsed sub-space), NOT a raw numpy array

Passing a numpy constant raises:

  TypeError: __init__(): incompatible function arguments. The
  following argument types are supported: ...
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import basix.ufl
import dolfinx
from mpi4py import MPI


def main() -> int:
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    P2 = basix.ufl.element("Lagrange", mesh.basix_cell(), 2,
                            shape=(2,))
    P1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
    TH = basix.ufl.mixed_element([P2, P1])
    W = dolfinx.fem.functionspace(mesh, TH)
    V_sub = W.sub(0)
    V_collapsed, _ = V_sub.collapse()

    def is_boundary(x):
        return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
                | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))

    boundary_dofs = dolfinx.fem.locate_dofs_geometrical(
        (V_sub, V_collapsed), is_boundary)
    try:
        # Bug: raw numpy constant on a sub-space.
        dolfinx.fem.dirichletbc(
            np.array([0.0, 0.0],
                      dtype=dolfinx.default_scalar_type),
            boundary_dofs, V_sub)
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: dirichletbc accepted raw constant on sub-space "
          "(catalog claim wrong)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
