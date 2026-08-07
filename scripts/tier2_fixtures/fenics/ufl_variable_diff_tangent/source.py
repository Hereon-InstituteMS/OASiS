"""Tier-2: ufl.variable + ufl.diff produce automatic stress derivative.

Pitfall (fenics hyperelasticity#3): For hyperelasticity, the
recommended pattern is to wrap the deformation gradient F in
ufl.variable, define the stored energy W(F_var) as a ufl
expression, and obtain the 1st Piola-Kirchhoff stress as
P = ufl.diff(W, F_var). ufl returns a VariableDerivative
expression directly usable inside ufl.inner(P, grad(v))*dx.

Mutation control: T2_MUTATE=1 skips the ufl.variable() wrapper — the
exact step the pitfall says you must not forget — and differentiates
with respect to the bare F expression. Both measurements then change:
F_var_type is measured as Sum (Identity + Grad, which is what F is
without the wrapper) and ufl.diff refuses the target, so P_type is
reported as the ValueError text rather than VariableDerivative. The
diff call is caught and printed, not allowed to abort the script, so
the kill is the flipped measurement and not a traceback.
"""
from __future__ import annotations

import os
import sys

import dolfinx
import ufl
from mpi4py import MPI

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (2,)))
    u = dolfinx.fem.Function(V)
    F = ufl.Identity(2) + ufl.grad(u)
    # MUTATE: forget the ufl.variable() wrapper.
    F_var = F if MUTATE else ufl.variable(F)
    print(f"F_var_type={type(F_var).__name__}")
    # Simple Neo-Hookean energy
    mu = 1.0
    W = (mu / 2) * (ufl.tr(F_var.T * F_var) - 2)
    try:
        P = ufl.diff(W, F_var)
        print(f"P_type={type(P).__name__}")
    except Exception as exc:
        P = None
        print(f"P_type=refused_{type(exc).__name__} {exc}")
    if type(F_var).__name__ == "Variable" and type(P).__name__ == "VariableDerivative":
        return 0
    print("ERROR: unexpected types from ufl.variable / ufl.diff",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
