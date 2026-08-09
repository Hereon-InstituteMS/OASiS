"""Tier-2: FEniCSx UFL curl() on scalar space raises.

MEASURED CORRECTION (2026-08-07, dolfinx 0.10.0 / ufl 2025.2): UFL does
NOT reject curl() on a scalar space. `ufl.curl(u)` for a scalar 2D `u`
returns a legal rank-1 object (the rotated gradient, ufl_shape (2,)) —
the printed `trial_shape=() curl_shape=(2,)` line below is that
measurement. The ValueError this fixture greps for is raised one level
later, by the SCALAR product `curl(u) * curl(v)`: rank-1 * rank-1 is
"Invalid ranks 1 and 1 in product". Written with ufl.inner() the very
same scalar space compiles silently, which is what the mutation does.

Mutation control: T2_MUTATE=1 contracts the two curls with ufl.inner()
instead of `*`. The form then builds on the SAME scalar space, nothing
raises, and both expected strings ('ValueError', 'Invalid ranks')
disappear.
"""
from __future__ import annotations

import os
import sys
import traceback

import ufl
from mpi4py import MPI

from dolfinx.fem import form, functionspace
from dolfinx.mesh import CellType, create_unit_square

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = create_unit_square(MPI.COMM_WORLD, 4, 4, CellType.triangle)
    V = functionspace(mesh, ("Lagrange", 1))  # scalar
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    # curl() ACCEPTS the scalar argument; the result is rank 1.
    print(f"trial_shape={u.ufl_shape} curl_shape={ufl.curl(u).ufl_shape}")
    try:
        if MUTATE:
            # Legal spelling: contract the two rank-1 curls.
            f = form(ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx)
        else:
            # rank-1 * rank-1 → UFL rank check rejects the PRODUCT.
            f = form(ufl.curl(u) * ufl.curl(v) * ufl.dx)
    except Exception:
        traceback.print_exc()
        return 1
    print(f"form_built_on_scalar_space=True kind={type(f).__name__}")
    if MUTATE:
        return 0
    print("ERROR: UFL accepted curl() on scalar space",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
