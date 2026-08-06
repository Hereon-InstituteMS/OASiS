"""Tier-2 for fenics time_dependent_heat#4: omitting `set_bc(b, bcs)` after the
ghost update leaves the constrained rows carrying the assembled load instead of
the prescribed values. Unlike the missing-apply_lifting case, this one IS visible
in a min/max print.

Unit square 32x32, T = 1 on the left wall, T = 0 on the right wall, T = 0
initially, no source, dt = 0.01, 20 backward-Euler steps, apply_lifting present
throughout. Wrong variant: the loop never calls set_bc. Right variant: it does.

Observed: with set_bc the range is exactly 'T in [0.0000, 1.0000]'; without it
the temperature lands in [-1.0543, +0.9360] and the hot wall itself ends at
-0.5169 - a negative temperature and a wall that never reaches its prescribed
value, both impossible under the discrete maximum principle for a source-free
problem with data in [0, 1]. Every KSP still reports converged reason 4 and the
field stays finite.

Mutation control: T2_MUTATE=1 calls set_bc in the checked variant; the range
violation disappears.
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

from petsc4py import PETSc  # noqa: E402

N, DT, NSTEP = 32, 0.01, 20
HOT, COLD = 1.0, 0.0


def run(use_set_bc: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    T_n, T_h = dolfinx.fem.Function(V), dolfinx.fem.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt = dolfinx.fem.Constant(msh, DT)
    a = (u / dt) * v * ufl.dx + ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (T_n / dt) * v * ufl.dx

    left = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    right = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 1.0))
    dl = dolfinx.fem.locate_dofs_topological(V, fdim, left)
    dr = dolfinx.fem.locate_dofs_topological(V, fdim, right)
    bcs = [dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, HOT), dl, V),
           dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, COLD), dr, V)]

    a_f, L_f = dolfinx.fem.form(a), dolfinx.fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(a_f, bcs=bcs)
    A.assemble()
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    b = dolfinx.fem.petsc.create_vector(V)

    lo, hi, reasons = HOT, COLD, []
    for _ in range(NSTEP):
        with b.localForm() as loc:
            loc.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, L_f)
        dolfinx.fem.petsc.apply_lifting(b, [a_f], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        if use_set_bc:
            dolfinx.fem.petsc.set_bc(b, bcs)
        ksp.solve(b, T_h.x.petsc_vec)
        T_h.x.scatter_forward()
        reasons.append(ksp.getConvergedReason())
        lo = min(lo, float(T_h.x.array.min()))
        hi = max(hi, float(T_h.x.array.max()))
        T_n.x.array[:] = T_h.x.array
    wall = float(np.max(T_h.x.array[dl]))
    return lo, hi, reasons, wall, bool(np.all(np.isfinite(T_h.x.array)))


def main() -> int:
    glo, ghi, greason, gwall, gfin = run(use_set_bc=True)
    lo, hi, reasons, wall, fin = run(use_set_bc=MUTATE)
    print(f"with_set_bc_range_line=T in [{glo:.4f}, {ghi:.4f}]")
    print(f"checked_range_line=T in [{lo:.4f}, {hi:.4f}]")
    print(f"checked_hot_wall_value={wall:.4f} checked_field_is_finite={fin}")
    print(f"checked_ksp_reason_is_4_every_step={set(reasons) == {4}}")
    print(f"with_set_bc_range_is_exactly_the_data_range="
          f"{abs(glo - COLD) < 1e-12 and abs(ghi - HOT) < 1e-12}")
    print(f"checked_min_is_negative={lo < -1e-3}")
    print(f"checked_hot_wall_never_reaches_its_prescribed_value={wall < HOT - 1e-3}")
    print(f"checked_violates_the_discrete_maximum_principle="
          f"{lo < COLD - 1e-3 or hi > HOT + 1e-3}")

    if (set(reasons) == {4} and fin and lo < -1e-3 and wall < HOT - 1e-3
            and abs(glo - COLD) < 1e-12 and abs(ghi - HOT) < 1e-12):
        print("VERDICT=missing_set_bc_leaves_the_prescribed_rows_wrong")
        return 0
    print("VERDICT=set_bc_made_no_difference")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
