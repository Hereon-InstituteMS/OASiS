"""Tier-2 for fenics magnetostatics#8: `dolfinx.fem.petsc.LinearProblem`
requires the keyword-only argument `petsc_options_prefix`, and the MUMPS direct
solve the magnetostatics recipe asks for is usable on this install because PETSc
was built with 32-bit indices.

Wrong variant: construct LinearProblem the 0.9 way, without the prefix.
Observed: TypeError: LinearProblem.__init__() missing 1 required keyword-only
argument: 'petsc_options_prefix'. The second half of the claim is checked by
measurement: PETSc.IntType is int32 here, and
'pc_factor_mat_solver_type': 'mumps' solves the coil problem with converged
reason 4; superlu_dist, the replacement the claim names for 64-bit builds, is
also present.

"Traceback" is deliberately NOT in this fixture's forbid_in_output: the
TypeError leaves a half-constructed LinearProblem whose __del__ then raises
AttributeError on self._solver, and Python prints that as an ignored-exception
traceback on stderr. That noise is dolfinx's, not the fixture's.

Mutation control: T2_MUTATE=1 passes petsc_options_prefix; construction
succeeds and the TypeError signal disappears.
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
J0 = 1.0e6
R_COIL = 0.2


def coil_problem(n: int = 16):
    msh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([-0.5, -0.5]), np.array([0.5, 0.5])], [n, n])
    tdim = msh.topology.dim
    ncells = msh.topology.index_map(tdim).size_local
    mid = dolfinx.mesh.compute_midpoints(
        msh, tdim, np.arange(ncells, dtype=np.int32)).T
    DG0 = dolfinx.fem.functionspace(msh, ("DG", 0))
    Jz = dolfinx.fem.Function(DG0)
    Jz.x.array[:] = 0.0
    Jz.x.array[(mid[0] ** 2 + mid[1] ** 2) < R_COIL ** 2] = J0
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (1.0 / MU0) * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = Jz * v * ufl.dx
    msh.topology.create_connectivity(tdim - 1, tdim)
    bdofs = dolfinx.fem.locate_dofs_topological(
        V, tdim - 1, dolfinx.mesh.exterior_facet_indices(msh.topology))
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), bdofs, V)
    return a, L, [bc]


def main() -> int:
    a, L, bcs = coil_problem()

    err = ""
    try:
        if MUTATE:
            dolfinx.fem.petsc.LinearProblem(
                a, L, bcs=bcs, petsc_options_prefix="t2_ms8_ok_")
        else:
            dolfinx.fem.petsc.LinearProblem(a, L, bcs=bcs)
        print("prefix_omitted_construction_raised=False")
    except TypeError as exc:
        err = str(exc)
        print(f"prefix_omitted_construction_raised=True TypeError: {err}")
    print(f"error_names_the_missing_kwarg="
          f"{'petsc_options_prefix' in err}")

    itype = np.dtype(PETSc.IntType).name
    print(f"petsc_index_type={itype}")
    print(f"petsc_int_is_32_bit={itype == 'int32'}")
    print(f"petsc_scalar_type={np.dtype(PETSc.ScalarType).name}")

    reasons = {}
    for solver in ("mumps", "superlu_dist"):
        a2, L2, bcs2 = coil_problem()
        try:
            prob = dolfinx.fem.petsc.LinearProblem(
                a2, L2, bcs=bcs2, petsc_options_prefix=f"t2_ms8_{solver}_",
                petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                               "pc_factor_mat_solver_type": solver})
            az = prob.solve()
            if isinstance(az, tuple):
                az = az[0]
            reasons[solver] = prob.solver.getConvergedReason()
            print(f"direct_solver={solver} converged_reason={reasons[solver]} "
                  f"max_Az={float(np.abs(az.x.array).max()):.6e}")
        except Exception as exc:
            reasons[solver] = 0
            print(f"direct_solver={solver} failed "
                  f"{type(exc).__name__}: {str(exc).splitlines()[0]}")
    print(f"mumps_solve_converged={reasons.get('mumps', 0) > 0}")
    print(f"superlu_dist_also_available="
          f"{reasons.get('superlu_dist', 0) > 0}")

    if err and itype == "int32" and reasons.get("mumps", 0) > 0:
        print("VERDICT=prefix_is_required_and_mumps_works_on_32bit_indices")
        return 0
    print("VERDICT=prefix_was_optional")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
