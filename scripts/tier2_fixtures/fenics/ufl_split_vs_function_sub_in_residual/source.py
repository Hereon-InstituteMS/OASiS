"""Tier-2 for fenics reaction_diffusion#1: take the species out of a mixed
Function with ufl.split(w) when building the residual. w.sub(i) and w.split()
also COMPILE -- that is what makes them dangerous -- but the automatic Jacobian
is then empty on those rows and the linear solve inside SNES dies.

Two-species 2A <-> B on a 16x16 unit square, P1 x P1, one backward-Euler step,
solved with dolfinx.fem.petsc.NonlinearProblem.

Observed on dolfinx 0.10.0:
  a, b = ufl.split(w)  -> SNES converges in 3 iterations (residual norms
      1.624290508696e-01, 1.640274608701e-03, 1.914942356727e-07,
      2.933399431099e-15)
  a, b = w.sub(0), w.sub(1) (and a, b = w.split()) -> the residual form and the
      Jacobian form both compile, the Jacobian assembles with every row empty,
      and problem.solve() raises "Error: error code 73" with the stack ending in
      MatLUFactorSymbolic_SeqAIJ() / "Object is in wrong state" /
      "Matrix is missing diagonal entry 0".

Mutation control: T2_MUTATE=1 uses ufl.split in the slot under test.
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

N, DT, D, K1, K2 = 16, 0.05, 0.01, 1.0, 1.0


def build(how: str):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
    w = dolfinx.fem.Function(W)
    w_n = dolfinx.fem.Function(W)
    w_n.sub(0).interpolate(lambda x: 1.0 + 0.5 * np.sin(2 * np.pi * x[0]))
    w_n.sub(1).interpolate(lambda x: np.full_like(x[0], 0.2))
    w_n.x.scatter_forward()
    w.x.array[:] = w_n.x.array
    if how == "split":
        A, B = ufl.split(w)
        An, Bn = ufl.split(w_n)
    else:
        A, B = w.sub(0), w.sub(1)
        An, Bn = w_n.sub(0), w_n.sub(1)
    va, vb = ufl.TestFunctions(W)
    r = K1 * A * A - K2 * B
    F = (((A - An) / DT) * va * ufl.dx
         + D * ufl.dot(ufl.grad(A), ufl.grad(va)) * ufl.dx
         + 2 * r * va * ufl.dx
         + ((B - Bn) / DT) * vb * ufl.dx
         + D * ufl.dot(ufl.grad(B), ufl.grad(vb)) * ufl.dx
         - r * vb * ufl.dx)
    return msh, W, w, F


def empty_jacobian_rows(F, w) -> tuple[int, int]:
    J = dolfinx.fem.petsc.assemble_matrix(
        dolfinx.fem.form(ufl.derivative(F, w)))
    J.assemble()
    dense = J.copy().convert("dense").getDenseArray()
    return int(np.sum(np.all(dense == 0.0, axis=1))), dense.shape[0]


def run(how: str):
    msh, W, w, F = build(how)
    compiled = dolfinx.fem.form(F) is not None
    zero_rows, n = empty_jacobian_rows(F, w)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, w, petsc_options_prefix=f"t2_rd1_{how}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    try:
        prob.solve()
    except Exception as exc:  # noqa: BLE001 - the message is the evidence
        return compiled, zero_rows, n, None, None, str(exc)
    return (compiled, zero_rows, n, prob.solver.getConvergedReason(),
            prob.solver.getIterationNumber(), "")


def main() -> int:
    how = "split" if MUTATE else "sub"
    comp_t, zr_t, n_t, reason_t, its_t, msg_t = run(how)
    comp_r, zr_r, n_r, reason_r, its_r, msg_r = run("split")

    print(f"reference_ufl_split: form_compiled={comp_r} "
          f"empty_jacobian_rows={zr_r}/{n_r} reason={reason_r} "
          f"newton_iterations={its_r}")
    print(f"under_test_{how}: form_compiled={comp_t} "
          f"empty_jacobian_rows={zr_t}/{n_t} reason={reason_t} "
          f"newton_iterations={its_t}")
    print(f"ufl_split_converges={reason_r is not None and reason_r > 0}")
    print(f"ufl_split_jacobian_has_no_empty_rows={zr_r == 0}")
    print(f"function_sub_residual_still_compiles={comp_t}")
    print(f"function_sub_jacobian_is_entirely_empty={zr_t == n_t}")
    print(f"function_sub_solve_raised={bool(msg_t)}")
    if msg_t:
        print("--- PETSc error from the w.sub(i) residual ---")
        print(msg_t)
        print("--- end PETSc error ---")
    if (reason_r is not None and reason_r > 0 and zr_r == 0 and comp_t
            and zr_t == n_t and msg_t):
        print("VERDICT=only_ufl_split_gives_a_usable_jacobian")
        return 0
    print("VERDICT=function_sub_in_the_residual_worked")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
