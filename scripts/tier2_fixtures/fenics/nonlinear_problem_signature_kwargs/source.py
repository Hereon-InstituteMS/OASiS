"""Tier-2: dolfinx NonlinearProblem & LinearProblem keyword-only signature.

Catalog claim under audit (deep_knowledge.py:540-542):
  api_0_10:
    problem = dolfinx.fem.petsc.NonlinearProblem(F, u, bcs,
                                                 petsc_options={...})
    problem.solve()

The catalog form is WRONG in two ways for dolfinx 0.10:

  (1) bcs is keyword-only (after the * in the signature), so
      NonlinearProblem(F, u, bcs) raises
        TypeError: __init__() takes 3 positional arguments but
                   4 were given
  (2) petsc_options_prefix is a REQUIRED keyword arg, so
      NonlinearProblem(F, u, bcs=bcs, petsc_options={...})
      raises
        TypeError: __init__() missing 1 required keyword-only
                   argument: 'petsc_options_prefix'

LinearProblem has the same kwarg-only / required-prefix shape.

Empirically verified 2026-06-01 against dolfinx 0.10.0. The
catalog's _version_info section already noted the prefix
requirement for the maxwell physics entry, but the
solver_catalog API stub at line 542 still showed the broken
form — exactly the snippet an LLM agent would paste verbatim.

This fixture asserts:
  * NonlinearProblem(F, u, bcs)           → TypeError
  * NonlinearProblem(F, u, bcs=bcs, ...)  → TypeError missing
                                            petsc_options_prefix
  * NonlinearProblem(F, u, bcs=bcs,
                     petsc_options_prefix='probe_',
                     petsc_options={...})  → OK
  * Same pattern for LinearProblem.
"""
from __future__ import annotations

import sys

import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from mpi4py import MPI


def main() -> int:
    m = dmesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = fem.functionspace(m, ("Lagrange", 1))
    u_nl = fem.Function(V)
    v_nl = ufl.TestFunction(V)
    F = u_nl * v_nl * ufl.dx
    u_lin = ufl.TrialFunction(V)
    a_lin = u_lin * v_nl * ufl.dx
    L_lin = 1.0 * v_nl * ufl.dx

    # (1) Positional bcs → TypeError
    pos_raises = False
    msg_pos = ""
    try:
        NonlinearProblem(F, u_nl, [])
    except TypeError as exc:
        msg_pos = str(exc)
        pos_raises = (
            "takes 3 positional arguments" in msg_pos
            or "positional argument" in msg_pos)
    print(f"positional_bcs_raises={pos_raises}")

    # (2) Without petsc_options_prefix → TypeError
    no_prefix_raises = False
    msg_np = ""
    try:
        NonlinearProblem(F, u_nl, bcs=[],
                         petsc_options={"ksp_type": "preonly"})
    except TypeError as exc:
        msg_np = str(exc)
        no_prefix_raises = (
            "petsc_options_prefix" in msg_np
            and "missing 1 required keyword-only" in msg_np)
    print(f"no_prefix_raises={no_prefix_raises}")
    print(f"no_prefix_diag_has_prefix_name="
          f"{'petsc_options_prefix' in msg_np}")

    # (3) Correct form → OK
    ok = False
    try:
        NonlinearProblem(
            F, u_nl, bcs=[],
            petsc_options_prefix="probe_nlp_",
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        ok = True
    except TypeError as exc:
        print(f"correct_form_unexpectedly_failed={exc}",
              file=sys.stderr)
    print(f"correct_form_ok={ok}")

    # (4) Same pattern for LinearProblem
    lp_no_prefix_raises = False
    try:
        LinearProblem(a_lin, L_lin, bcs=[],
                      petsc_options={"ksp_type": "preonly"})
    except TypeError as exc:
        lp_no_prefix_raises = (
            "petsc_options_prefix" in str(exc))
    print(f"linear_problem_no_prefix_raises={lp_no_prefix_raises}")

    if (pos_raises and no_prefix_raises and ok
            and lp_no_prefix_raises):
        return 0
    print("FAIL: signature invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
