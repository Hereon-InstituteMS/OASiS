"""Tier-2 for fenics time_dependent_heat#3: changing dt (or k, rho, cp) mid-run
without reassembling A is a silent failure - the matrix was built once outside the
loop, so only the right-hand side sees the new value.

32x32 unit square, T = 1 on the left wall, T = 0 on the right, insulated top and
bottom, backward Euler. 25 steps of dt = 0.01, then dt is multiplied by 10 for
another 25 steps. With the stale matrix the min/max print stays 'T in [0.0000,
1.0000]' and every KSP still reports converged reason 4, while the mid-height
profile collapses towards zero instead of holding the straight ramp: the RHS mass
term is now 1/(10 dt) while the matrix still carries 1/dt, so each step multiplies
the interior by about a tenth. The remedy from the claim - A.zeroEntries();
assemble_matrix(A, a_form, bcs=bcs); A.assemble() - is what the mutation runs.

Mutation control: T2_MUTATE=1 reassembles A when dt changes; the profile then
matches the reference and the collapse is gone.
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

N, NSTEP, SWITCH, DT0, FACTOR = 32, 50, 25, 0.01, 10.0
PROBE_X = np.linspace(0.05, 0.95, 10)


def run(reassemble: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    T_n, T_h = dolfinx.fem.Function(V), dolfinx.fem.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt_c = dolfinx.fem.Constant(msh, DT0)
    a = (u / dt_c) * v * ufl.dx + ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (T_n / dt_c) * v * ufl.dx
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
    b = dolfinx.fem.petsc.create_vector(V)

    lines, reasons = [], []
    for step in range(NSTEP):
        if step == SWITCH:
            dt_c.value = DT0 * FACTOR
            if reassemble:
                A.zeroEntries()
                dolfinx.fem.petsc.assemble_matrix(A, a_f, bcs=bcs)
                A.assemble()
                ksp.setOperators(A)
        with b.localForm() as loc:
            loc.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, L_f)
        dolfinx.fem.petsc.apply_lifting(b, [a_f], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, bcs)
        ksp.solve(b, T_h.x.petsc_vec)
        T_h.x.scatter_forward()
        reasons.append(ksp.getConvergedReason())
        lines.append(f"T in [{T_h.x.array.min():.4f}, {T_h.x.array.max():.4f}]")
        T_n.x.array[:] = T_h.x.array

    coords = V.tabulate_dof_coordinates()
    mid = np.where(np.isclose(coords[:, 1], 0.5))[0]
    profile = [float(T_h.x.array[mid[np.argmin(np.abs(coords[mid, 0] - xq))]])
               for xq in PROBE_X]
    return lines, reasons, np.array(profile)


def main() -> int:
    lines, reasons, prof = run(reassemble=MUTATE)
    ok_lines, ok_reasons, ok_prof = run(reassemble=True)
    ramp = 1.0 - PROBE_X

    print(f"selected_run_reassembles={MUTATE}")
    print(f"selected_last_step_line={lines[-1]}")
    healthy_line = all(ln == "T in [0.0000, 1.0000]" for ln in lines)
    print(f"selected_min_max_print_looks_healthy_every_step={healthy_line}")
    pos = all(r > 0 for r in reasons) and all(r > 0 for r in ok_reasons)
    print(f"every_ksp_reason_positive={pos} reasons={sorted(set(reasons))}")
    print(f"selected_profile={[f'{p:.3e}' for p in prof]}")
    print(f"reassembled_profile={[f'{p:.3e}' for p in ok_prof]}")
    gap = float(np.max(np.abs(prof - ok_prof)))
    ramp_err = float(np.max(np.abs(ok_prof - ramp)))
    print(f"max_profile_gap={gap:.4f} reassembled_ramp_error={ramp_err:.4f}")
    print(f"stale_matrix_profile_differs_from_the_reassembled_one={gap > 0.3}")
    print(f"reassembled_profile_follows_the_ramp={ramp_err < 0.05}")
    if healthy_line and pos and gap > 0.3 and ramp_err < 0.05:
        print("VERDICT=stale_matrix_is_a_converged_wrong_answer")
        return 0
    print("VERDICT=stale_matrix_made_no_difference")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
