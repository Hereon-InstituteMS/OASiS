"""Tier-2 for fenics time_dependent_heat#0: omitting `apply_lifting` in an
assemble-once time loop is a SILENT failure. `set_bc` still overwrites the
Dirichlet rows, so every printed min/max lands exactly on the prescribed values
and every KSP reports converged, while the interior solution is wrong by an
order of magnitude.

Unit square 32x32, T = 1 on the left wall, T = 0 on the right wall, T = 0
initially, no source, dt = 0.01, 20 backward-Euler steps. Wrong variant: the
loop assembles b, ghost-updates and calls set_bc but never calls apply_lifting.
Right variant: the same loop with apply_lifting.

Observed: both variants print the identical line 'T in [0.0000, 1.0000]' at
every single step and both report KSP converged reason 4 at every step, yet the
run without lifting stores 22x less heat (int T dx = 2.201e-02 against
4.895e-01) and its mid-height profile is O(1e-2) where the correct one ramps
from ~0.94 down to ~0.03. A min/max print cannot see this; the assembled
Galerkin residual over the UNCONSTRAINED dofs can - it is at machine zero with
lifting and O(1) without.

Mutation control: T2_MUTATE=1 calls apply_lifting in the checked variant; the
residual drops to machine zero and the stored-heat deficit disappears.
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


def run(lift: bool):
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
    bcs = [dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 1.0), dl, V),
           dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), dr, V)]

    a_f, L_f = dolfinx.fem.form(a), dolfinx.fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(a_f, bcs=bcs)
    A.assemble()
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    b = dolfinx.fem.petsc.create_vector(V)
    heat = dolfinx.fem.form(T_h * ufl.dx)

    lines, reasons, prev = [], [], None
    for _ in range(NSTEP):
        with b.localForm() as loc:
            loc.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, L_f)
        if lift:
            dolfinx.fem.petsc.apply_lifting(b, [a_f], bcs=[bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, bcs)
        ksp.solve(b, T_h.x.petsc_vec)
        T_h.x.scatter_forward()
        reasons.append(ksp.getConvergedReason())
        lines.append(f"T in [{float(T_h.x.array.min()):.4f}, "
                     f"{float(T_h.x.array.max()):.4f}]")
        prev = T_n.x.array.copy()
        T_n.x.array[:] = T_h.x.array

    stored = float(dolfinx.fem.assemble_scalar(heat))
    # Galerkin residual of the LAST step: T_n must hold what that step used.
    T_n.x.array[:] = prev
    r = dolfinx.fem.petsc.assemble_vector(
        dolfinx.fem.form(ufl.replace(a - L, {u: T_h})))
    r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    res = r.array.copy()
    free = np.setdiff1d(np.arange(res.size), np.concatenate([dl, dr]))
    xy = V.tabulate_dof_coordinates()
    row = np.where(np.abs(xy[:, 1] - 0.5) < 1e-9)[0]
    row = row[np.argsort(xy[row, 0])]
    prof = T_h.x.array[row][::4]
    return {"lines": lines, "reasons": reasons, "stored": stored,
            "resid": float(np.max(np.abs(res[free]))),
            "profile": prof}


def main() -> int:
    good = run(lift=True)
    bad = run(lift=MUTATE)
    same_line = set(bad["lines"]) == set(good["lines"]) == {good["lines"][0]}
    print(f"with_lifting_range_line_every_step={good['lines'][0]}")
    print(f"checked_range_line_every_step={bad['lines'][0]}")
    print(f"checked_prints_the_same_reassuring_range_line_at_every_step={same_line}")
    print(f"checked_ksp_reasons={sorted(set(bad['reasons']))}")
    print(f"checked_ksp_reason_is_4_every_step={set(bad['reasons']) == {4}}")
    print(f"with_lifting_stored_heat={good['stored']:.4e} "
          f"checked_stored_heat={bad['stored']:.4e}")
    print(f"with_lifting_midheight_profile="
          f"{np.array2string(good['profile'], precision=4)}")
    print(f"checked_midheight_profile="
          f"{np.array2string(bad['profile'], precision=4)}")
    print(f"with_lifting_free_dof_residual={good['resid']:.3e} "
          f"checked_free_dof_residual={bad['resid']:.3e}")
    ratio = good["stored"] / bad["stored"]
    print(f"stored_heat_ratio={ratio:.2f}")
    print(f"with_lifting_residual_at_machine_zero={good['resid'] < 1e-10}")
    print(f"checked_stored_heat_is_more_than_ten_times_too_small={ratio > 10.0}")
    print(f"checked_free_dof_residual_is_order_one={bad['resid'] > 1e-2}")

    if (same_line and set(bad["reasons"]) == {4} and good["resid"] < 1e-10
            and ratio > 10.0 and bad["resid"] > 1e-2):
        print("VERDICT=missing_apply_lifting_is_invisible_to_a_min_max_print")
        return 0
    print("VERDICT=lifting_made_no_difference")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
