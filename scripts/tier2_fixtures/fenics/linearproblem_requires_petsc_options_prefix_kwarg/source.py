"""Tier-2 for fenics magnetostatics#8: `dolfinx.fem.petsc.LinearProblem`
requires the keyword-only argument `petsc_options_prefix`, and the claim quotes
the TypeError verbatim. The same claim asserts that this install has 32-bit
PETSc indices and a working MUMPS, which is what makes
'pc_factor_mat_solver_type': 'mumps' a legal choice here.

The fixture constructs the magnetostatics Az problem without the prefix and
prints the exact exception text, then reports PETSc.IntType and solves the same
problem through MUMPS.

Mutation control: T2_MUTATE=1 passes petsc_options_prefix, and no TypeError is
raised.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

MU0 = 4.0e-7 * np.pi


def main() -> int:
    msh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([-0.5, -0.5]), np.array([0.5, 0.5])],
        [16, 16])
    tdim = msh.topology.dim
    ncells = msh.topology.index_map(tdim).size_local
    mid = dolfinx.mesh.compute_midpoints(
        msh, tdim, np.arange(ncells, dtype=np.int32)).T
    DG0 = dolfinx.fem.functionspace(msh, ("DG", 0))
    Jz = dolfinx.fem.Function(DG0)
    Jz.x.array[:] = 0.0
    Jz.x.array[(mid[0] ** 2 + mid[1] ** 2) < 0.04] = 1.0e6
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (1.0 / MU0) * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = Jz * v * ufl.dx
    msh.topology.create_connectivity(tdim - 1, tdim)
    bdofs = dolfinx.fem.locate_dofs_topological(
        V, tdim - 1, dolfinx.mesh.exterior_facet_indices(msh.topology))
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), bdofs, V)
    opts = {"ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps"}

    kwargs = {"petsc_options_prefix": "t2_ms8_"} if MUTATE else {}
    print(f"petsc_options_prefix_supplied={bool(kwargs)}")
    msg = ""
    try:
        prob = dolfinx.fem.petsc.LinearProblem(
            a, L, bcs=[bc], petsc_options=opts, **kwargs)
        print("constructor_returned=True")
    except TypeError as exc:
        msg = str(exc)
        prob = None
        print(f"TypeError: {msg}")
    print(f"typeerror_raised={bool(msg)}")

    prob2 = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix="t2_ms8_ok_", petsc_options=opts)
    prob2.solve()
    mumps_reason = prob2.solver.getConvergedReason()
    print(f"petsc_int_type={np.dtype(PETSc.IntType).name}")
    print(f"mumps_reason={mumps_reason}")
    print(f"mumps_solve_reason_is_positive={mumps_reason > 0}")
    if msg and mumps_reason > 0 and np.dtype(PETSc.IntType).name == "int32":
        print("VERDICT=petsc_options_prefix_is_required_and_mumps_is_available")
        return 0
    print("VERDICT=prefix_is_optional")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
