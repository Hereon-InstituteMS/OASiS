"""Tier-2 for fenics heat#2: forget `T_n.x.array[:] = T_h.x.array` and the time
loop recomputes step 1 for ever, silently.

32x32 unit square, backward Euler, 30 steps of dt = 0.01, unit source. With the
copy, int(T dx) grows monotonically. Without it, every step returns the SAME
vector — bit-identical — and the KSP reports converged each time.

Mutation control: T2_MUTATE=1 restores the copy; the per-step values start
changing.
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
NSTEP, DT = 30, 0.01


def run(copy_back: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    T_n = dolfinx.fem.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt = dolfinx.fem.Constant(msh, DT)
    f = dolfinx.fem.Constant(msh, 1.0)
    a = (u / dt) * v * ufl.dx + ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (T_n / dt) * v * ufl.dx + f * v * ufl.dx
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[], petsc_options_prefix="t2_cp_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    integral = dolfinx.fem.form(
        (T_n if False else dolfinx.fem.Function(V)) * ufl.dx)  # placeholder
    vals, arrays = [], []
    for _ in range(NSTEP):
        T_h = prob.solve()
        if isinstance(T_h, tuple):
            T_h = T_h[0]
        vals.append(float(dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(T_h * ufl.dx))))
        arrays.append(T_h.x.array.copy())
        if copy_back:
            T_n.x.array[:] = T_h.x.array
    del integral
    return vals, arrays


def main() -> int:
    vals, arrays = run(copy_back=MUTATE)
    ref_vals, _ = run(copy_back=True)
    print(f"without_copy_step1={vals[0]:.6f} without_copy_step30={vals[-1]:.6f}")
    print(f"with_copy_step1={ref_vals[0]:.6f} "
          f"with_copy_step30={ref_vals[-1]:.6f}")
    frozen = all(np.array_equal(arrays[0], a) for a in arrays[1:])
    grows = ref_vals[-1] > 3.0 * ref_vals[0]
    print(f"every_step_bit_identical={frozen}")
    print(f"with_copy_integral_grows={grows}")
    if frozen and grows:
        print("VERDICT=missing_copy_freezes_the_solution_silently")
        return 0
    print("VERDICT=solution_advanced")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
