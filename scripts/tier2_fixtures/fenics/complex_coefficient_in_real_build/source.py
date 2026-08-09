"""Tier-2 for fenics poisson#1: where a complex value actually fails in a REAL
PETSc build.

The claim: use dolfinx.default_scalar_type so dtypes match the PETSc build, and
the failure from getting it wrong surfaces at form compilation / array
assignment, NOT at fem.Constant construction.

The fixture checks all three statements on the real (float64) build:
  * fem.Constant(msh, 1+2j) constructs without raising;
  * fem.form of a form carrying that constant raises;
  * writing a complex number into a real Function array raises.

Mutation control: T2_MUTATE=1 uses default_scalar_type for both, which is the
documented correct pattern; nothing raises and the verdict changes.
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
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    print(f"default_scalar_type={np.dtype(dolfinx.default_scalar_type).name}")

    value = dolfinx.default_scalar_type(2.0) if MUTATE else (1.0 + 2.0j)

    # Statement 1: Constant construction itself.
    try:
        c = dolfinx.fem.Constant(msh, value)
        print("constant_construction_raised=False")
    except Exception as exc:  # pragma: no cover - would falsify the claim
        print(f"constant_construction_raised=True {type(exc).__name__}: {exc}")
        return 1

    # Statement 2: form compilation.
    v = ufl.TestFunction(V)
    form_err = ""
    try:
        dolfinx.fem.form(c * v * ufl.dx)
        print("form_raised=False")
    except Exception as exc:
        form_err = f"{type(exc).__name__}: {exc}"
        print(f"form_raised=True {form_err}")

    # Statement 3: array assignment.
    f = dolfinx.fem.Function(V)
    arr_err = ""
    try:
        f.x.array[0] = value
        print("array_assign_raised=False")
    except Exception as exc:
        arr_err = f"{type(exc).__name__}: {exc}"
        print(f"array_assign_raised=True {arr_err}")

    if form_err and arr_err:
        print("VERDICT=complex_rejected_at_form_and_array_not_at_constant")
        return 0
    print("VERDICT=real_build_accepted_complex")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
