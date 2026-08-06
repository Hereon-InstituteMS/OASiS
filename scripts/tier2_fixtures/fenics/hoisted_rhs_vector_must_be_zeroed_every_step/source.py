"""Tier-2 for fenics time_dependent_heat#1: if you hoist the right-hand-side
vector out of the time loop to avoid reallocating it, you MUST zero it at the top
of every step, because `dolfinx.fem.petsc.assemble_vector(b, L)` ADDS into b.

Unit square 16x16, T = 1 on the left wall, T = 0 on the right, T = 0 initially,
dt = 0.01, 100 backward-Euler steps. Wrong variant: b is created once and never
zeroed. Right variant: `with b.localForm() as bl: bl.set(0.0)` at the top of
every step.

Observed: the unzeroed run reaches max|T| = 7.69e+26 after 100 steps while every
single KSP reports converged reason 4 - the field is finite, so a NaN check would
not fire either. The opposite mistake is measured in the same run and is NOT a
problem: allocating a fresh vector with `b = assemble_vector(L_form)` inside the
loop leaves the resident set flat (0.0 MB growth over 200 steps and 0.0 MB over
1000 steps).

Mutation control: T2_MUTATE=1 zeroes the hoisted vector in the checked variant;
the blow-up disappears and T stays inside [0, 1].
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

N, DT, NSTEP = 16, 0.01, 100


def rss_mb() -> float:
    with open("/proc/self/statm", "r") as fh:
        pages = int(fh.read().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE") / 1e6


def build():
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
    bcs = [dolfinx.fem.dirichletbc(
               dolfinx.fem.Constant(msh, 1.0),
               dolfinx.fem.locate_dofs_topological(V, fdim, left), V),
           dolfinx.fem.dirichletbc(
               dolfinx.fem.Constant(msh, 0.0),
               dolfinx.fem.locate_dofs_topological(V, fdim, right), V)]
    a_f, L_f = dolfinx.fem.form(a), dolfinx.fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(a_f, bcs=bcs)
    A.assemble()
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    return V, T_n, T_h, a_f, L_f, bcs, ksp


def march(zero_it: bool, nstep: int = NSTEP):
    V, T_n, T_h, a_f, L_f, bcs, ksp = build()
    b = dolfinx.fem.petsc.create_vector(V)
    reasons, peak = [], 0.0
    for _ in range(nstep):
        if zero_it:
            with b.localForm() as loc:
                loc.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, L_f)
        dolfinx.fem.petsc.apply_lifting(b, [a_f], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, bcs)
        ksp.solve(b, T_h.x.petsc_vec)
        T_h.x.scatter_forward()
        reasons.append(ksp.getConvergedReason())
        peak = max(peak, float(np.max(np.abs(T_h.x.array))))
        T_n.x.array[:] = T_h.x.array
    return peak, reasons, bool(np.all(np.isfinite(T_h.x.array)))


def fresh_vector_rss(nstep: int) -> float:
    V, T_n, T_h, a_f, L_f, bcs, ksp = build()
    before = rss_mb()
    for _ in range(nstep):
        b = dolfinx.fem.petsc.assemble_vector(L_f)   # fresh every step
        dolfinx.fem.petsc.apply_lifting(b, [a_f], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, bcs)
        ksp.solve(b, T_h.x.petsc_vec)
        T_h.x.scatter_forward()
        T_n.x.array[:] = T_h.x.array
    return rss_mb() - before


def main() -> int:
    gpeak, greason, _ = march(zero_it=True)
    peak, reasons, finite = march(zero_it=MUTATE)
    print(f"zeroed_peak_abs_T={gpeak:.4e} "
          f"zeroed_reason_is_4_every_step={set(greason) == {4}}")
    print(f"checked_peak_abs_T_after_{NSTEP}_steps={peak:.2e} "
          f"checked_field_is_finite={finite}")
    print(f"checked_ksp_reason_is_4_every_step={set(reasons) == {4}}")
    print(f"zeroed_stays_inside_the_data_range={gpeak <= 1.0 + 1e-12}")
    print(f"checked_grew_past_1e10={peak > 1e10}")

    g200 = fresh_vector_rss(200)
    g1000 = fresh_vector_rss(1000)
    print(f"fresh_vector_rss_growth_200_steps_mb={g200:.1f} "
          f"fresh_vector_rss_growth_1000_steps_mb={g1000:.1f}")
    flat = g1000 < max(3.0 * g200, 0.0) + 20.0
    print(f"fresh_vector_per_step_does_not_leak={flat}")

    if (peak > 1e10 and finite and set(reasons) == {4}
            and gpeak <= 1.0 + 1e-12 and set(greason) == {4} and flat):
        print("VERDICT=unzeroed_hoisted_rhs_blows_up_while_every_ksp_reports_converged")
        return 0
    print("VERDICT=zeroing_the_rhs_made_no_difference")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
