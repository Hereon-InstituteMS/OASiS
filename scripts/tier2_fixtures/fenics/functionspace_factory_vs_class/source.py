"""Tier-2: dolfinx fem.functionspace (factory) vs fem.FunctionSpace (class).

In dolfinx 0.10 the canonical way to build a function space is
the lowercase factory:

    V = fem.functionspace(mesh, ("Lagrange", 1))

The capital-F name `fem.FunctionSpace` ALSO exists — it is the
underlying ABC class. Calling it directly fails:

    fem.FunctionSpace(mesh, ("Lagrange", 1))
      → TypeError: FunctionSpace.__init__() missing 1 required
        positional argument: 'cppV'

This is a real LLM-trap migration drift because:
  * Old dolfin (legacy, pre-x) used FunctionSpace as the
    canonical name.
  * Many tutorials and Stack Overflow answers from 2020-2023
    still show `FunctionSpace(mesh, ...)`.
  * An LLM agent trained on that corpus that pastes
    `fem.FunctionSpace(mesh, ...)` into a dolfinx 0.10 script
    crashes with TypeError on a non-obvious 'cppV' arg.

This fixture asserts both ends:
  * fem.functionspace returns a FunctionSpace-typed object.
  * fem.FunctionSpace(mesh, ...) raises TypeError mentioning
    'cppV'.
"""
from __future__ import annotations

import sys

from dolfinx import fem, mesh
from mpi4py import MPI


def main() -> int:
    m = mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)

    V = fem.functionspace(m, ("Lagrange", 1))
    ok_factory = type(V).__name__ == "FunctionSpace"
    print(f"factory_returns_FunctionSpace={ok_factory}")
    print(f"factory_callable={callable(fem.functionspace)}")
    print(f"class_exists={hasattr(fem, 'FunctionSpace')}")

    # Old-style direct call must raise TypeError mentioning cppV
    raised = False
    err = ""
    try:
        fem.FunctionSpace(m, ("Lagrange", 1))
    except TypeError as exc:
        err = str(exc)
        raised = "cppV" in err
    print(f"class_direct_call_raises_typeerror={raised}")
    print(f"class_diag_has_cppv={('cppV' in err)}")

    ok = ok_factory and raised
    if ok:
        return 0
    print("FAIL: factory/class invariant not held",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
