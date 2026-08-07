"""Tier-2 for fenics heat#9: the theta-method LHS is time-independent, and
re-solving through LinearProblem every step re-assembles and re-factorises it.
The answer is identical, so only the clock shows the mistake.

Three loops over the same transient problem: assemble+factorise once and reuse
the KSP; keep one LinearProblem and call .solve() each step; construct a fresh
LinearProblem each step. The fixture asserts the ORDERING of the wall-times and
that all three end at the same integral — no absolute timing is pinned, because
that would only be true on this machine.

Mutation control: T2_MUTATE=1 makes the "reuse" loop rebuild a LinearProblem
every step as well, so the speed advantage disappears.
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

import time  # noqa: E402

NSTEP, DT, N = 50, 0.01, 64


def build(msh):
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    T_n = dolfinx.fem.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt = dolfinx.fem.Constant(msh, DT)
    f = dolfinx.fem.Constant(msh, 1.0)
    a = (u / dt) * v * ufl.dx + ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (T_n / dt) * v * ufl.dx + f * v * ufl.dx
    return V, T_n, a, L


def loop_reuse(msh, rebuild_each_step: bool):
    from petsc4py import PETSc
    V, T_n, a, L = build(msh)
    a_f, L_f = dolfinx.fem.form(a), dolfinx.fem.form(L)
    t0 = time.perf_counter()
    if rebuild_each_step:
        for _ in range(NSTEP):
            prob = dolfinx.fem.petsc.LinearProblem(
                a, L, bcs=[], petsc_options_prefix="t2_p9a_",
                petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
            T_h = prob.solve()
            if isinstance(T_h, tuple):
                T_h = T_h[0]
            T_n.x.array[:] = T_h.x.array
    else:
        A = dolfinx.fem.petsc.assemble_matrix(a_f)
        A.assemble()
        ksp = PETSc.KSP().create(msh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        b = dolfinx.fem.petsc.create_vector(V)
        T_h = dolfinx.fem.Function(V)
        for _ in range(NSTEP):
            with b.localForm() as loc:
                loc.set(0.0)
            dolfinx.fem.petsc.assemble_vector(b, L_f)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)
            ksp.solve(b, T_h.x.petsc_vec)
            T_h.x.scatter_forward()
            T_n.x.array[:] = T_h.x.array
    dtime = time.perf_counter() - t0
    total = float(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(T_n * ufl.dx)))
    return dtime, total


def loop_one_problem(msh):
    V, T_n, a, L = build(msh)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[], petsc_options_prefix="t2_p9b_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    t0 = time.perf_counter()
    for _ in range(NSTEP):
        T_h = prob.solve()
        if isinstance(T_h, tuple):
            T_h = T_h[0]
        T_n.x.array[:] = T_h.x.array
    dtime = time.perf_counter() - t0
    total = float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(T_n * ufl.dx)))
    return dtime, total


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    t_reuse, i_reuse = loop_reuse(msh, rebuild_each_step=MUTATE)
    t_one, i_one = loop_one_problem(msh)
    t_fresh, i_fresh = loop_reuse(msh, rebuild_each_step=True)
    print(f"reuse_seconds={t_reuse:.3f} one_problem_seconds={t_one:.3f} "
          f"fresh_problem_seconds={t_fresh:.3f}")
    print(f"reuse_integral={i_reuse:.10f} one_problem_integral={i_one:.10f} "
          f"fresh_integral={i_fresh:.10f}")
    same = (abs(i_reuse - i_one) < 1e-9 and abs(i_reuse - i_fresh) < 1e-9)
    faster = t_reuse * 2.0 < t_one and t_one < t_fresh
    print(f"all_three_give_the_same_answer={same}")
    print(f"reuse_at_least_2x_faster_and_fresh_is_slowest={faster}")
    if same and faster:
        print("VERDICT=only_wall_time_reveals_the_reassembly")
        return 0
    print("VERDICT=no_timing_separation")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
