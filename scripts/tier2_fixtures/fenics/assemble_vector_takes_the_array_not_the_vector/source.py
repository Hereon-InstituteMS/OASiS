"""Tier-2 for fenics matrix_free_poisson#1: the in-place assembly entry point is
`fem.assemble_vector(b.array, M_fem)` -- the first argument is the numpy ARRAY,
not the la.Vector. The one-argument call `b = fem.assemble_vector(L_fem)` is also
valid and returns a fresh dolfinx.la.Vector, which is how b is created in the
first place.

Wrong variant: fem.assemble_vector(b, M_fem) with the la.Vector as first
argument. Right variant: fem.assemble_vector(b.array, M_fem).

Observed on dolfinx 0.10.0: passing the la.Vector gives
"AttributeError: 'Vector' object has no attribute 'function_spaces'" -- the
two-argument overload treats its first positional argument as the OUTPUT buffer
only when the second is a compiled Form, and with a single argument it treats it
as the form itself, which is why the error is about function_spaces. Neither of
the previously quoted messages reproduces: no "TypeError: assemble_vector() takes
positional argument" and no "expected ndarray, got Vector" appears anywhere in
the raised text. The one-argument call returns dolfinx.la.Vector, and the array
form assembles in place; both are checked against each other here.

Mutation control: T2_MUTATE=1 passes b.array, the documented first argument, so
the AttributeError text and the failure tokens never appear.
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

from dolfinx import fem, la, mesh  # noqa: E402

DTYPE = dolfinx.default_scalar_type
N = 16


def main() -> int:
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 10.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_fem = fem.form(ufl.inner(f, v) * ufl.dx, dtype=DTYPE)
    ui = fem.Function(V, dtype=DTYPE)
    ui.interpolate(lambda X: 1.0 + X[0] + 2.0 * X[1])
    M_fem = fem.form(ufl.action(a, ui), dtype=DTYPE)

    # The one-argument call is how b is created; it returns a la.Vector.
    b = fem.assemble_vector(L_fem)
    tname = f"{type(b).__module__}.{type(b).__name__}"
    print(f"one_argument_call_return_type={tname}")
    returns_la_vector = tname == "dolfinx.la.Vector"

    raised = ""
    if MUTATE:
        y = la.vector(V.dofmap.index_map, 1, DTYPE)
        y.array[:] = 0.0
        fem.assemble_vector(y.array, M_fem)
        print("mutation=first_argument_is_the_numpy_array")
    else:
        y = la.vector(V.dofmap.index_map, 1, DTYPE)
        y.array[:] = 0.0
        try:
            fem.assemble_vector(y, M_fem)  # the wrong variant: la.Vector
            print("la_vector_as_first_argument_succeeded=True")
        except Exception as exc:  # noqa: BLE001 - the text is the evidence
            raised = f"{type(exc).__name__}: {exc}"
            print(f"fem.assemble_vector(la_vector, M_fem) -> {raised}")
        fem.assemble_vector(y.array, M_fem)

    y.scatter_reverse(la.InsertMode.add)
    filled = float(np.linalg.norm(y.array)) > 0.0
    print(f"array_form_assembled_a_nonzero_vector={filled}")

    # cross-check: the in-place assembly of the action equals A*ui
    A = dolfinx.fem.petsc.assemble_matrix(fem.form(a))
    A.assemble()
    xv, ref = A.createVecRight(), A.createVecLeft()
    xv.array[:] = ui.x.array[: xv.local_size]
    A.mult(xv, ref)
    nref = float(np.linalg.norm(ref.array))
    rel = float(np.linalg.norm(y.array[: ref.local_size] - ref.array)) / max(nref, 1.0)
    print(f"inplace_action_vs_matvec_rel_diff={rel:.3e}")

    old1 = "assemble_vector() takes positional argument"
    old2 = "expected ndarray, got Vector"
    print(f"one_argument_call_returns_la_vector={returns_la_vector}")
    print(f"la_vector_as_first_argument_raises_attributeerror={bool(raised)}")
    print(f"previously_quoted_typeerror_text_absent={old1 not in raised}")
    print(f"previously_quoted_ndarray_text_absent={old2 not in raised}")
    print(f"array_form_matches_matvec={rel < 1e-10}")
    if returns_la_vector and raised and filled and rel < 1e-10 \
            and old1 not in raised and old2 not in raised:
        print("VERDICT=assemble_vector_needs_the_array_not_the_la_vector")
        return 0
    print("VERDICT=assemble_vector_accepted_the_la_vector")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
