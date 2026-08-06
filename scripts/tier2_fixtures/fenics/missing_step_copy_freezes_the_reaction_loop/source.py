"""Tier-2 for fenics reaction_diffusion#11: the time loop of a mixed
reaction-diffusion system needs w_n.x.array[:] = w.x.array at the END of every
step. Leave it out and every step re-solves the first one: nothing is raised,
the SNES reports converged throughout, and the only visible tell is the Newton
count collapsing.

Initial condition per species with w_n.sub(i).interpolate(...), then
w_n.x.scatter_forward(), then the Newton guess w.x.array[:] = w_n.x.array.

Observed on dolfinx 0.10.0 (16x16 unit square, P1 x P1, 2A <-> B, 10
backward-Euler steps):
  with the copy    min(w) walks 0.202966, 0.206283, 0.209852, ... and the Newton
                   count is 3 (2 on the last two steps), reason 3 throughout
  without the copy min(w) is 0.202966 at all ten steps, the Newton count is
                   3, 1, 1, 1, 1, 1, 1, 1, 1, 1 and the converged reason
                   switches from 3 (FNORM_RELATIVE) to 4 (SNORM_RELATIVE) from
                   the second step on -- "converged because the update was
                   tiny", which is exactly what re-solving a solved step does
NOTE the claim's exact tell was a count of 0 from the third step on; here the
floor is 1 with reason 4, so the collapse is real but not literally to zero.
Nothing is raised and the reason stays positive in both runs.

Mutation control: T2_MUTATE=1 puts the end-of-step copy back.
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

N, D, KF, KR, DT, NSTEP = 16, 0.01, 1.0, 1.0, 0.05, 10


def run(copy_back: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    P1 = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([P1, P1]))
    w = dolfinx.fem.Function(W)
    w_n = dolfinx.fem.Function(W)
    w_n.sub(0).interpolate(lambda x: 1.0 + 0.5 * np.sin(2 * np.pi * x[0]))
    w_n.sub(1).interpolate(lambda x: np.full_like(x[0], 0.2))
    w_n.x.scatter_forward()
    w.x.array[:] = w_n.x.array
    A, B = ufl.split(w)
    An, Bn = ufl.split(w_n)
    va, vb = ufl.TestFunctions(W)
    r = KF * A * A - KR * B
    F = (((A - An) / DT) * va * ufl.dx
         + D * ufl.dot(ufl.grad(A), ufl.grad(va)) * ufl.dx
         + 2 * r * va * ufl.dx
         + ((B - Bn) / DT) * vb * ufl.dx
         + D * ufl.dot(ufl.grad(B), ufl.grad(vb)) * ufl.dx
         - r * vb * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, w, petsc_options_prefix=f"t2_rd11_{int(copy_back)}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    mins, its, reasons = [], [], []
    for _ in range(NSTEP):
        prob.solve()
        w.x.scatter_forward()
        its.append(int(prob.solver.getIterationNumber()))
        reasons.append(int(prob.solver.getConvergedReason()))
        mins.append(float(w.x.array.min()))
        if copy_back:
            w_n.x.array[:] = w.x.array
    return mins, its, reasons


def main() -> int:
    mins_t, its_t, rea_t = run(copy_back=MUTATE)
    mins_r, its_r, rea_r = run(copy_back=True)

    print("with_copy_min=" + " ".join(f"{m:.6f}" for m in mins_r))
    print(f"with_copy_newton_counts={its_r} reasons={rea_r}")
    print("under_test_min=" + " ".join(f"{m:.6f}" for m in mins_t))
    print(f"under_test_newton_counts={its_t} reasons={rea_t}")

    walks = len(set(f"{m:.9f}" for m in mins_r)) == NSTEP
    frozen = len(set(f"{m:.9f}" for m in mins_t)) == 1
    steady_its = all(i >= 2 for i in its_r)
    collapsed = its_t[0] > 1 and max(its_t[1:]) <= 1
    snorm = rea_t[0] == 3 and all(r == 4 for r in rea_t[1:])
    converged = all(r > 0 for r in rea_t)
    print(f"with_copy_solution_advances_every_step={walks}")
    print(f"with_copy_newton_count_is_steady={steady_its}")
    print(f"without_copy_solution_is_frozen={frozen}")
    print(f"without_copy_newton_count_collapses_to_one_or_zero={collapsed}")
    print(f"without_copy_reason_switches_to_snorm_relative={snorm}")
    print(f"snes_reported_converged_throughout={converged}")

    if walks and steady_its and frozen and collapsed and snorm and converged:
        print("VERDICT=missing_step_copy_re_solves_the_same_step")
        return 0
    print("VERDICT=the_loop_advanced_without_the_copy")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
