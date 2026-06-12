"""Tier-2: ufl.variable + ufl.diff produce automatic stress derivative.

Pitfall (fenics hyperelasticity#3): For hyperelasticity, the
recommended pattern is to wrap the deformation gradient F in
ufl.variable, define the stored energy W(F_var) as a ufl
expression, and obtain the 1st Piola-Kirchhoff stress as
P = ufl.diff(W, F_var). ufl returns a VariableDerivative
expression directly usable inside ufl.inner(P, grad(v))*dx.
"""
from __future__ import annotations

import sys

import dolfinx
import ufl
from mpi4py import MPI


def main() -> int:
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    u = dolfinx.fem.Function(V)
    F = ufl.Identity(2) + ufl.grad(u)
    F_var = ufl.variable(F)
    print(f"F_var_type={type(F_var).__name__}")
    # Simple Neo-Hookean energy
    mu = 1.0
    W = (mu / 2) * (ufl.tr(F_var.T * F_var) - 2)
    P = ufl.diff(W, F_var)
    print(f"P_type={type(P).__name__}")
    if type(F_var).__name__ == "Variable" and type(P).__name__ == "VariableDerivative":
        return 0
    print("ERROR: unexpected types from ufl.variable / ufl.diff",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
