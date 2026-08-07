"""Tier-2 for fenics maxwell#1: the exact text a REAL build produces for a
complex Maxwell form, and the denial of two strings that do not appear.

Mutation control: T2_MUTATE=1 uses a real coefficient; the form builds.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix  # noqa: E402
import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"



def main() -> int:
    print(f"scalar_type={np.dtype(dolfinx.default_scalar_type).name}")
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    el = basix.ufl.element("N1curl", msh.basix_cell(), 1)
    V = dolfinx.fem.functionspace(msh, el)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    omega2 = dolfinx.fem.Constant(msh, 1.0 if MUTATE else 1.0j)
    a = (ufl.inner(ufl.curl(u), ufl.curl(v))
         - omega2 * ufl.inner(u, v)) * ufl.dx

    # TWO SPELLINGS, and they fail differently — the claim quotes only one.
    # A complex fem.Constant is rejected by the float64 Form binding with a
    # pybind11 TypeError; the documented ValueError appears only when the
    # imaginary value is a UFL literal inside the expression.
    msg = ""
    try:
        dolfinx.fem.form(a)
        print("constant_spelling_raised=False")
    except Exception as exc:
        print(f"constant_spelling_raised=True {type(exc).__name__}: "
              f"{str(exc).splitlines()[0]}")
    a_lit = (ufl.inner(ufl.curl(u), ufl.curl(v))
             - ufl.as_ufl(1.0 if MUTATE else 1.0j) * ufl.inner(u, v)) * ufl.dx
    try:
        dolfinx.fem.form(a_lit)
        print("literal_spelling_raised=False")
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        print(f"literal_spelling_raised=True {msg}")
    print(f"form_raised={bool(msg)}")

    f = dolfinx.fem.Function(V)
    arr_msg = ""
    try:
        f.x.array[0] = 1.0 if MUTATE else 1.0j
        print("array_assign_raised=False")
    except Exception as exc:
        arr_msg = f"{type(exc).__name__}: {exc}"
        print(f"array_assign_raised=True {arr_msg}")

    both = msg + " " + arr_msg
    print(f"old_string_cannot_convert_present="
          f"{'cannot convert complex to float' in both}")
    print(f"old_string_imaginary_discarded_present="
          f"{'imaginary part discarded' in both}")
    if msg and arr_msg:
        print("VERDICT=real_build_rejects_complex_at_form_and_array")
        return 0
    print("VERDICT=complex_accepted")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
