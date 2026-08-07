"""Tier-2 for fenics nonlinear_pde#5: on this install SNES converged reason 2 is
CONVERGED_FNORM_ABS, not CONVERGED_FNORM_RELATIVE. A wrapper whose table is
shifted by one reports a plausible-looking name for every solve, so any logic
built on the NAME is unreliable. Test only the SIGN.

The fixture reads the value->name table out of PETSc.SNES.ConvergedReason,
checks the whole list the knowledge text quotes, then runs three real solves on
the Bratu problem (unit square 16x16, P1, homogeneous Dirichlet): one with a
deliberately loose snes_atol so that SNES stops at iteration 0 with reason 2,
one at lambda = 1 which converges normally, and one at lambda = 20 which
diverges. The wrong variant -- a naive off-by-one name table -- is applied to
each and compared with PETSc's own name.

Observed on dolfinx 0.10.0 / PETSc 3.24.5: 2 = CONVERGED_FNORM_ABS,
3 = CONVERGED_FNORM_RELATIVE, 4 = CONVERGED_SNORM_RELATIVE, 5 = CONVERGED_ITS,
-1 DIVERGED_FUNCTION_DOMAIN, -3 DIVERGED_LINEAR_SOLVE, -4 DIVERGED_FNORM_NAN,
-5 DIVERGED_MAX_IT, -6 DIVERGED_LINE_SEARCH, -8 DIVERGED_LOCAL_MIN. The naive
table mislabels the real solves while the sign test is right every time.

Mutation control: T2_MUTATE=1 gives the wrapper the correct table, so it stops
mislabelling.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N = 16
CLAIMED = {2: "CONVERGED_FNORM_ABS", 3: "CONVERGED_FNORM_RELATIVE",
           4: "CONVERGED_SNORM_RELATIVE", 5: "CONVERGED_ITS",
           -1: "DIVERGED_FUNCTION_DOMAIN", -3: "DIVERGED_LINEAR_SOLVE",
           -4: "DIVERGED_FNORM_NAN", -5: "DIVERGED_MAX_IT",
           -6: "DIVERGED_LINE_SEARCH", -8: "DIVERGED_LOCAL_MIN"}
# The off-by-one table a wrapper produces when it reads the positive reasons in
# the wrong order.
NAIVE = {2: "CONVERGED_FNORM_RELATIVE", 3: "CONVERGED_SNORM_RELATIVE",
         4: "CONVERGED_ITS", 5: "CONVERGED_FNORM_ABS"}


def petsc_names() -> dict[int, str]:
    R = PETSc.SNES.ConvergedReason
    out: dict[int, str] = {}
    for name in dir(R):
        if name.startswith("_") or name in ("ITERATING",):
            continue
        val = getattr(R, name)
        if isinstance(val, int):
            out.setdefault(int(val), name)
    return out


def bratu(tag: str, lmbda: float, opts: dict):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    msh.topology.create_connectivity(1, 2)
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, 1, facets)
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(msh, PETSc.ScalarType(0.0)), dofs, V)
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    lam = dolfinx.fem.Constant(msh, PETSc.ScalarType(lmbda))
    F = (ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - lam * ufl.exp(u) * v * ufl.dx)
    base = {"ksp_type": "preonly", "pc_type": "lu"}
    base.update(opts)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, u, bcs=[bc], petsc_options_prefix=f"t2_np5_{tag}_",
        petsc_options=base)
    prob.solve()
    return (prob.solver.getConvergedReason(),
            prob.solver.getIterationNumber())


def main() -> int:
    names = petsc_names()
    two = names.get(2)
    three = names.get(3)
    print(f"petsc_reason_2_name={two}")
    print(f"petsc_reason_3_name={three}")
    print(f"reason_2_is_converged_fnorm_abs={two == 'CONVERGED_FNORM_ABS'}")
    print("reason_3_is_converged_fnorm_relative="
          f"{three == 'CONVERGED_FNORM_RELATIVE'}")
    table_ok = all(names.get(k) == v for k, v in CLAIMED.items())
    print(f"table={ {k: names.get(k) for k in sorted(CLAIMED)} }")
    print(f"claimed_reason_table_matches_this_install={table_ok}")

    wrapper = dict(CLAIMED) if MUTATE else dict(NAIVE)
    cases = [
        ("loose_atol", 1.0, {"snes_atol": 1.0, "snes_max_it": 30}),
        ("lambda_1", 1.0, {"snes_max_it": 30}),
        ("lambda_20", 20.0, {"snes_max_it": 30,
                             "snes_linesearch_type": "basic"}),
    ]
    mislabelled, sign_ok = 0, True
    for tag, lam, opts in cases:
        reason, its = bratu(tag, lam, opts)
        truth = names.get(reason, "UNKNOWN")
        guess = wrapper.get(reason, truth)
        agree = guess == truth
        mislabelled += 0 if agree else 1
        converged_by_sign = reason > 0
        converged_by_truth = truth.startswith("CONVERGED")
        sign_ok = sign_ok and (converged_by_sign == converged_by_truth)
        print(f"{tag}: reason={reason} iterations={its} petsc_name={truth} "
              f"wrapper_name={guess} wrapper_agrees={agree} "
              f"sign_says_converged={converged_by_sign}")
    print(f"wrapper_mislabelled_this_many_real_solves={mislabelled}")
    print(f"naive_wrapper_mislabelled_a_real_solve={mislabelled > 0}")
    print(f"sign_test_agreed_with_petsc_in_every_case={sign_ok}")

    if (two == "CONVERGED_FNORM_ABS" and three == "CONVERGED_FNORM_RELATIVE"
            and table_ok and mislabelled > 0 and sign_ok):
        print("VERDICT=test_only_the_sign_the_names_are_easy_to_shift")
        return 0
    print("VERDICT=the_name_table_was_trustworthy")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
