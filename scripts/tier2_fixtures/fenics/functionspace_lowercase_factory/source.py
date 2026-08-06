"""Tier-2 for fenics heat#12: the space constructor is fem.functionspace with a
lower-case s. fem.FunctionSpace exists but is the internal class, and
fem.VectorFunctionSpace is gone entirely.

Mutation control: T2_MUTATE=1 uses the factory; nothing raises.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"



def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)

    cls_err = ""
    if MUTATE:
        V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
        print(f"factory_ndofs={V.dofmap.index_map.size_global}")
        print("class_call_raised=False")
    else:
        try:
            dolfinx.fem.FunctionSpace(msh, ("Lagrange", 1))
            print("class_call_raised=False")
        except TypeError as exc:
            cls_err = str(exc)
            print(f"class_call_raised=True TypeError: {cls_err}")

    has_vector = hasattr(dolfinx.fem, "VectorFunctionSpace")
    print(f"VectorFunctionSpace_exists={has_vector}")
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (2,)))
    print(f"factory_vector_ndofs={V.dofmap.index_map.size_global * 2}")
    print(f"factory_vector_ok={V.dofmap.index_map.size_global > 0}")
    if cls_err and not has_vector:
        print("VERDICT=only_the_lowercase_factory_works")
        return 0
    print("VERDICT=class_form_usable")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
