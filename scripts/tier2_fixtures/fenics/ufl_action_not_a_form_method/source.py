"""Tier-2 for fenics matrix_free_poisson#0: build the operator action with
`M = ufl.action(a, ui)`; `a.action(ui)` is not a UFL Form method. Then compile
it once with `M_fem = fem.form(M, dtype=dtype)` and reuse the compiled form in
every CG iteration.

Wrong variant: `a.action(ui)`, and handing the raw UFL form to
fem.assemble_vector without fem.form. Right variant: ufl.action(a, ui) compiled
once with fem.form and assembled in place.

Observed on dolfinx 0.10.0: `a.action(ui)` raises
"AttributeError: 'Form' object has no attribute 'action'" at the line that
builds M -- ufl.form.Form simply has no such method. The claim's second half is
call-form dependent: the in-place two-argument call
fem.assemble_vector(y.array, M) with an UNCOMPILED form raises
"AttributeError: 'Form' object has no attribute '_cpp_object'" exactly as
claimed, while the one-argument fem.assemble_vector(M) raises
"AttributeError: 'Form' object has no attribute 'function_spaces'" instead.
Neither is the previously quoted "RuntimeError: cannot assemble: form has not
been compiled", which appears nowhere. The compiled form assembles and its
result equals a plain matrix-vector product with the assembled stiffness matrix.

The recompile-in-the-loop cost is printed as context only: with a WARM FFCx
cache the ratio is a factor of a few, not orders of magnitude, so it is not
asserted here.

Mutation control: T2_MUTATE=1 uses ufl.action(a, ui) and fem.form(...), so
neither AttributeError text nor the failure tokens appear.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import time  # noqa: E402

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dolfinx import fem, la, mesh  # noqa: E402

DTYPE = dolfinx.default_scalar_type
N = 16
DEGREE = 2


def main() -> int:
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, N, N)
    V = fem.functionspace(msh, ("Lagrange", DEGREE))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    ui = fem.Function(V, dtype=DTYPE)
    ui.interpolate(lambda x: 1.0 + x[0] + 2.0 * x[1])

    print(f"type_of_bilinear_form={type(a).__module__}.{type(a).__name__}")

    # --- 1) building the action -------------------------------------------
    method_err = ""
    if MUTATE:
        M = ufl.action(a, ui)
        print("mutation=ufl_action_used_for_the_operator_action")
    else:
        try:
            M = a.action(ui)  # the wrong variant
            print("form_dot_action_returned_something=True")
        except AttributeError as exc:
            method_err = f"{type(exc).__name__}: {exc}"
            print(f"a.action(ui) -> {method_err}")
            M = ufl.action(a, ui)

    # --- 2) assembling an UNCOMPILED form ---------------------------------
    y = la.vector(V.dofmap.index_map, 1, DTYPE)
    inplace_err = ""
    oneshot_err = ""
    if MUTATE:
        print("mutation=form_compiled_with_fem_form_before_assembly")
    else:
        try:
            fem.assemble_vector(y.array, M)  # M is raw UFL, not compiled
            print("uncompiled_inplace_assembly_succeeded=True")
        except Exception as exc:  # noqa: BLE001 - the text is the evidence
            inplace_err = f"{type(exc).__name__}: {exc}"
            print(f"fem.assemble_vector(y.array, M_uncompiled) -> {inplace_err}")
        try:
            fem.assemble_vector(M)
            print("uncompiled_oneshot_assembly_succeeded=True")
        except Exception as exc:  # noqa: BLE001
            oneshot_err = f"{type(exc).__name__}: {exc}"
            print(f"fem.assemble_vector(M_uncompiled) -> {oneshot_err}")

    # --- 3) the correct route, and a check that it IS the operator --------
    M_fem = fem.form(M, dtype=DTYPE)
    y.array[:] = 0.0
    fem.assemble_vector(y.array, M_fem)
    y.scatter_reverse(la.InsertMode.add)

    A = dolfinx.fem.petsc.assemble_matrix(fem.form(a))
    A.assemble()
    xv = A.createVecRight()
    xv.array[:] = ui.x.array[: xv.local_size]
    ref = A.createVecLeft()
    A.mult(xv, ref)
    nref = float(np.linalg.norm(ref.array))
    diff = float(np.linalg.norm(y.array[: ref.local_size] - ref.array))
    action_matches = diff <= 1e-10 * max(nref, 1.0)
    print(f"action_vs_matvec_rel_diff={diff / max(nref, 1.0):.3e}")

    # --- 4) recompile cost, context only ---------------------------------
    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        y.array[:] = 0.0
        fem.assemble_vector(y.array, M_fem)
    t_once = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(reps):
        y.array[:] = 0.0
        fem.assemble_vector(y.array, fem.form(ufl.action(a, ui), dtype=DTYPE))
    t_loop = time.perf_counter() - t0
    print(f"compiled_once_s={t_once:.4f} recompiled_each_iteration_s={t_loop:.4f} "
          f"ratio={t_loop / max(t_once, 1e-9):.1f} (context only, warm ffcx cache)")

    quoted = "cannot assemble: form has not been compiled"
    seen = quoted in (inplace_err + oneshot_err)
    print(f"form_dot_action_raises_attributeerror={bool(method_err)}")
    print(f"uncompiled_inplace_assembly_raises_attributeerror={bool(inplace_err)}")
    print(f"uncompiled_oneshot_message_differs_from_inplace_one="
          f"{bool(oneshot_err) and oneshot_err != inplace_err}")
    print(f"previously_quoted_runtime_error_text_absent={not seen}")
    print(f"ufl_action_compiled_once_equals_matvec={action_matches}")
    if method_err and inplace_err and action_matches and not seen:
        print("VERDICT=form_has_no_action_method_and_uncompiled_forms_do_not_assemble")
        return 0
    print("VERDICT=form_action_and_uncompiled_assembly_worked")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
