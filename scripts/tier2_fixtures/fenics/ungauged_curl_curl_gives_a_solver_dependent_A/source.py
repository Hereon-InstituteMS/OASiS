"""Tier-2 for fenics magnetostatics#3: the 3D curl-curl operator without a
gauge is singular -- every gradient is in its kernel -- and this does NOT show
up as a solver failure. Every solver reports success, no zero pivot is
reported, and each one returns a DIFFERENT vector potential A while returning
the SAME B = curl A.

Wrong variant: assemble only inner(curl(A), curl(w))*dx on N1curl and solve it
with three different solvers (MUMPS-LU, CG/Jacobi, GMRES/ILU) on a 6x6x6 unit
cube with a divergence-free source, then post-process A itself.

Mutation control: T2_MUTATE=1 adds the regularising mass term
+ inner(A, w)*dx, which makes the operator nonsingular; A is then the same for
all three solvers and the solver-dependence signal disappears.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import basix.ufl  # noqa: E402
import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N = 6
SOLVERS = (
    ("mumps_lu", {"ksp_type": "preonly", "pc_type": "lu",
                  "pc_factor_mat_solver_type": "mumps"}),
    ("cg_jacobi", {"ksp_type": "cg", "pc_type": "jacobi",
                   "ksp_rtol": 1e-10, "ksp_max_it": 3000}),
    ("gmres_ilu", {"ksp_type": "gmres", "pc_type": "ilu",
                   "ksp_rtol": 1e-10, "ksp_max_it": 3000}),
)


def solve(name: str, opts: dict) -> tuple[int, float, float, float]:
    msh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N)
    V = dolfinx.fem.functionspace(
        msh, basix.ufl.element("N1curl", msh.basix_cell(), 1))
    A, w = ufl.TrialFunction(V), ufl.TestFunction(V)
    # RHS built as curl of a given field, so b lies exactly in the range of the
    # singular operator: the system is consistent and the solvers have no
    # reason to complain -- only the kernel component of A is undetermined.
    psi = dolfinx.fem.Function(V)
    psi.interpolate(lambda x: np.vstack((np.sin(np.pi * x[1]) * x[2],
                                         np.cos(np.pi * x[0]) * x[2],
                                         np.sin(np.pi * x[0]) * x[1])))
    a = ufl.inner(ufl.curl(A), ufl.curl(w)) * ufl.dx
    if MUTATE:
        a = a + ufl.inner(A, w) * ufl.dx
    L = ufl.inner(ufl.curl(psi), ufl.curl(w)) * ufl.dx
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[], petsc_options_prefix=f"t2_ms3_{name}_",
        petsc_options=opts)
    Ah = prob.solve()
    if isinstance(Ah, tuple):
        Ah = Ah[0]
    reason = prob.solver.getConvergedReason()
    res = float(prob.solver.getResidualNorm())
    na = float(np.sqrt(abs(dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        ufl.inner(Ah, Ah) * ufl.dx)))))
    nb = float(np.sqrt(abs(dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        ufl.inner(ufl.curl(Ah), ufl.curl(Ah)) * ufl.dx)))))
    return reason, res, na, nb


def main() -> int:
    print(f"gauge_term_present={MUTATE}")
    reasons, na, nb = [], [], []
    for name, opts in SOLVERS:
        r, res, a_norm, b_norm = solve(name, opts)
        reasons.append(r)
        na.append(a_norm)
        nb.append(b_norm)
        print(f"solver={name} converged_reason={r} residual={res:.3e} "
              f"L2_norm_A={a_norm:.14f} L2_norm_curl_A={b_norm:.14f}")

    all_conv = all(r > 0 for r in reasons)
    spread_a = max(na) / min(na)
    spread_b = max(nb) / min(nb)
    print(f"A_norm_spread_max_over_min={spread_a:.4f}")
    print(f"curl_A_norm_spread_max_over_min={spread_b:.12f}")
    print(f"every_solver_reported_converged={all_conv}")
    print(f"A_differs_between_solvers_by_more_than_2x={spread_a > 2.0}")
    print(f"curl_A_agrees_across_solvers_to_9_digits={spread_b - 1.0 < 1e-9}")

    if all_conv and spread_a > 2.0 and (spread_b - 1.0) < 1e-9:
        print("VERDICT=ungauged_curl_curl_is_silently_solver_dependent_in_A")
        return 0
    print("VERDICT=A_was_solver_independent")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
