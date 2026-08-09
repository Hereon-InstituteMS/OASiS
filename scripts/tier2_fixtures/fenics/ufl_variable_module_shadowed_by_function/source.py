"""Tier-2 for fenics hyperelasticity#4: ufl.variable() + ufl.diff() give the
first Piola-Kirchhoff stress from a stored energy, but the class Variable is NOT
reachable as ufl.variable.Variable.

The attribute ufl.variable is the FUNCTION variable(), which shadows the
submodule of the same name, so repr(type(ufl.variable(F))) prints
"<class 'ufl.variable.Variable'>" while writing ufl.variable.Variable raises
AttributeError: 'function' object has no attribute 'Variable'. The spelling that
works is ufl.classes.Variable.

The fixture also checks the useful half of the claim: type(ufl.variable(F)) is
Variable, type(ufl.diff(W, F_var)).__name__ == "VariableDerivative", and the
resulting P is usable inside inner(P, grad(v))*dx -- it is assembled into a real
residual vector, so "usable" is measured, not asserted.

Mutation control: T2_MUTATE=1 looks the class up as ufl.classes.Variable, which
does not raise.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    d = msh.geometry.dim
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (d,)))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.vstack([0.05 * x[0], 0.0 * x[1]]))
    v = ufl.TestFunction(V)

    f_var = ufl.variable(ufl.Identity(d) + ufl.grad(u))
    mu, lmbda = 1.0, 10.0
    j = ufl.det(f_var)
    psi = (mu / 2) * (ufl.tr(f_var.T * f_var) - d) - mu * ufl.ln(j) \
        + (lmbda / 2) * ufl.ln(j) ** 2
    piola = ufl.diff(psi, f_var)

    print(f"ufl_variable_attribute_is_a={type(ufl.variable).__name__}")
    print(f"repr_of_type_of_F_var={type(f_var)!r}")
    print(f"F_var_type={type(f_var).__name__}")
    print(f"P_type={type(piola).__name__}")
    print(f"F_var_is_ufl_classes_Variable="
          f"{isinstance(f_var, ufl.classes.Variable)}")

    raised = ""
    if MUTATE:
        cls = ufl.classes.Variable
        print(f"lookup=ufl.classes.Variable raised=False resolved={cls.__name__}")
    else:
        try:
            cls = ufl.variable.Variable
            print(f"lookup=ufl.variable.Variable raised=False "
                  f"resolved={cls.__name__}")
        except AttributeError as exc:
            raised = f"{type(exc).__name__}: {exc}"
            print(f"lookup=ufl.variable.Variable raised=True {raised}")

    residual = dolfinx.fem.form(ufl.inner(piola, ufl.grad(v)) * ufl.dx)
    b = dolfinx.fem.assemble_vector(residual)
    nrm = float(np.linalg.norm(b.array))
    print(f"residual_norm_from_diff_stress={nrm:.6e}")
    usable = np.isfinite(nrm) and nrm > 0.0

    shadowed = ("'function' object has no attribute 'Variable'" in raised)
    print(f"submodule_shadowed_by_function={shadowed}")
    print(f"P_assembles_into_a_residual={usable}")
    if (shadowed and usable and type(f_var).__name__ == "Variable"
            and type(piola).__name__ == "VariableDerivative"):
        print("VERDICT=use_ufl_classes_Variable_not_ufl_variable_Variable")
        return 0
    print("VERDICT=variable_spelling_is_harmless")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
