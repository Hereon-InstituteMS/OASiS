"""Tier-2 for fenics heat#1: a time-dependent Dirichlet value must be written
INTO the fem.Constant the bc holds; rebinding the Python name is a no-op.

Transient heat, T0 = 0, left wall g(t) = 1 + 2t, other walls insulated, backward
Euler, 50 steps of dt = 0.01. Three variants: update `g.value`, rebind the name
to a fresh Constant, and do nothing at all. The claim's sharpest observable is
that rebinding is BIT-IDENTICAL to doing nothing, and that neither raises.

Mutation control: T2_MUTATE=1 makes the "rebind" variant write g.value instead,
so it stops matching the do-nothing run.
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
NSTEP, DT = 50, 0.01


def run(mode: str) -> np.ndarray:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 16, 16)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    T_n = dolfinx.fem.Function(V)
    T_n.x.array[:] = 0.0
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt = dolfinx.fem.Constant(msh, DT)
    a = (u / dt) * v * ufl.dx + ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (T_n / dt) * v * ufl.dx

    left = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs = dolfinx.fem.locate_dofs_topological(V, fdim, left)
    g = dolfinx.fem.Constant(msh, 1.0)
    bc = dolfinx.fem.dirichletbc(g, dofs, V)

    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix="t2_gd_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    for step in range(1, NSTEP + 1):
        t = step * DT
        target = 1.0 + 2.0 * t
        if mode == "update":
            g.value = target
        elif mode == "rebind":
            g = dolfinx.fem.Constant(msh, target)  # noqa: F841 - the mistake
        T_h = prob.solve()
        if isinstance(T_h, tuple):
            T_h = T_h[0]
        T_n.x.array[:] = T_h.x.array
    return T_n.x.array.copy()


def main() -> int:
    a_update = run("update")
    a_rebind = run("update" if MUTATE else "rebind")
    a_none = run("none")
    print(f"update_max={np.max(a_update):.6f} "
          f"rebind_max={np.max(a_rebind):.6f} "
          f"none_max={np.max(a_none):.6f}")
    identical = bool(np.array_equal(a_rebind, a_none))
    evolved = abs(np.max(a_update) - 2.0) < 1e-9
    stuck = abs(np.max(a_none) - 1.0) < 1e-9
    print(f"rebind_bit_identical_to_no_update={identical}")
    print(f"update_reaches_g_at_final_time={evolved}")
    print(f"no_update_stays_at_first_step_value={stuck}")
    if identical and evolved and stuck:
        print("VERDICT=rebinding_the_name_is_a_silent_noop")
        return 0
    print("VERDICT=rebinding_had_an_effect")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
