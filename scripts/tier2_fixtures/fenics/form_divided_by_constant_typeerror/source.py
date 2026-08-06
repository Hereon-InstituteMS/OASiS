"""Tier-2 for fenics heat#5: a UFL Form cannot be divided by a fem.Constant.

`u * v * ufl.dx / dt` divides the FORM, not the argument. The fixture triggers
it and prints the TypeError verbatim, then shows the two documented spellings
that work.

Mutation control: T2_MUTATE=1 divides the argument instead of the form.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt = dolfinx.fem.Constant(msh, 0.1)

    msg = ""
    try:
        if MUTATE:
            (u / dt) * v * ufl.dx
        else:
            u * v * ufl.dx / dt
        print("division_raised=False")
    except TypeError as exc:
        msg = str(exc)
        print(f"division_raised=True TypeError: {msg}")

    # Both documented cures.
    (u / dt) * v * ufl.dx
    (1.0 / dt) * u * v * ufl.dx
    print("both_documented_spellings_build=True")
    if msg:
        print("VERDICT=form_divided_by_constant_is_a_typeerror")
        return 0
    print("VERDICT=form_division_accepted")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
