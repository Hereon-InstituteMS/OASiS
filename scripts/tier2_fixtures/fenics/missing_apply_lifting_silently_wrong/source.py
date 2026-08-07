"""Tier-2 for fenics heat#10: in a manual assemble-once loop, omitting
apply_lifting leaves the Dirichlet columns unaccounted for. The KSP still
reports converged, the field is finite, and the answer is wrong.

The detector the claim recommends is the one used here: assemble the step
residual with the computed solution substituted and look at the NON-Dirichlet
dofs. With apply_lifting they are at machine zero; without, they are O(1).

Mutation control: T2_MUTATE=1 restores apply_lifting.
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

NSTEP, DT, N = 5, 0.01, 16


def run(lift: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    T_n = dolfinx.fem.Function(V)
    T_n.x.array[:] = 1.0
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt = dolfinx.fem.Constant(msh, DT)
    a = (u / dt) * v * ufl.dx + ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (T_n / dt) * v * ufl.dx
    left = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    dofs = dolfinx.fem.locate_dofs_topological(V, fdim, left)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 2.0), dofs, V)

    a_f, L_f = dolfinx.fem.form(a), dolfinx.fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(a_f, bcs=[bc])
    A.assemble()
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    b = dolfinx.fem.petsc.create_vector(V)
    T_h = dolfinx.fem.Function(V)
    reason = 0
    prev = T_n.x.array.copy()
    for _ in range(NSTEP):
        with b.localForm() as loc:
            loc.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b, L_f)
        if lift:
            dolfinx.fem.petsc.apply_lifting(b, [a_f], bcs=[[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, [bc])
        ksp.solve(b, T_h.x.petsc_vec)
        T_h.x.scatter_forward()
        reason = ksp.getConvergedReason()
        prev = T_n.x.array.copy()
        T_n.x.array[:] = T_h.x.array

    # Residual of the LAST step, so T_n must hold the value that step actually
    # used — not the solution that overwrote it at the end of the loop. Getting
    # this wrong made the correct variant report a residual of 2e-2 and the
    # fixture concluded lifting made no difference.
    T_n.x.array[:] = prev
    F = ufl.replace(a - L, {u: T_h})
    r = dolfinx.fem.petsc.assemble_vector(dolfinx.fem.form(F))
    r.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    res = r.array.copy()
    free = np.setdiff1d(np.arange(res.size), dofs)
    return (reason, float(np.max(np.abs(res[free]))),
            float(np.min(T_h.x.array)), float(np.max(T_h.x.array)),
            bool(np.all(np.isfinite(T_h.x.array))))


def main() -> int:
    reason, resid, lo, hi, finite = run(lift=MUTATE)
    r_ok, res_ok, lo_ok, hi_ok, _ = run(lift=True)
    print(f"with_lifting_free_dof_residual={res_ok:.3e} "
          f"with_lifting_range=[{lo_ok:.6f}, {hi_ok:.6f}]")
    print(f"no_lifting_converged_reason={reason} "
          f"no_lifting_free_dof_residual={resid:.3e} "
          f"no_lifting_range=[{lo:.6f}, {hi:.6f}]")
    print(f"with_lifting_residual_at_machine_zero={res_ok < 1e-10}")
    print(f"no_lifting_converged_anyway={reason > 0}")
    print(f"no_lifting_field_is_finite={finite}")
    print(f"no_lifting_residual_is_order_one={resid > 1e-2}")
    if res_ok < 1e-10 and reason > 0 and finite and resid > 1e-2:
        print("VERDICT=missing_apply_lifting_is_a_converged_wrong_answer")
        return 0
    print("VERDICT=lifting_made_no_difference")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
