"""Tier-2 for fenics magnetostatics#6: `LinearProblem.solve()` never raises when
the KSP fails. The dolfinx 0.10 docstring says so outright, and the script
happily carries on writing output files afterwards. The guard is
`problem.solver.getConvergedReason() > 0`, or
`"ksp_error_if_not_converged": True`.

On the coil-in-iron Az problem, 'preonly'+LU returns reason 4 and CG+Jacobi
returns reason 2; unpreconditioned GMRES capped at 2 iterations returns reason
-3 (DIVERGED_MAX_IT), solve() returns the (wrong) Function without raising, and
an XDMF file of that field is written straight afterwards.

Mutation control: T2_MUTATE=1 adds "ksp_error_if_not_converged": True to the
same failing configuration, which turns the silent failure into a raised
petsc4py error.
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

MU0 = 4.0e-7 * np.pi


def problem(tag: str, opts: dict):
    msh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([-0.5, -0.5]), np.array([0.5, 0.5])],
        [16, 16])
    tdim = msh.topology.dim
    ncells = msh.topology.index_map(tdim).size_local
    mid = dolfinx.mesh.compute_midpoints(
        msh, tdim, np.arange(ncells, dtype=np.int32)).T
    DG0 = dolfinx.fem.functionspace(msh, ("DG", 0))
    nu = dolfinx.fem.Function(DG0)
    nu.x.array[:] = 1.0 / MU0
    m = np.maximum(np.abs(mid[0]), np.abs(mid[1]))
    nu.x.array[(m > 0.25) & (m < 0.40)] = 1.0 / (MU0 * 1000.0)
    Jz = dolfinx.fem.Function(DG0)
    Jz.x.array[:] = 0.0
    Jz.x.array[(mid[0] ** 2 + mid[1] ** 2) < 0.04] = 1.0e6
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = nu * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = Jz * v * ufl.dx
    msh.topology.create_connectivity(tdim - 1, tdim)
    bdofs = dolfinx.fem.locate_dofs_topological(
        V, tdim - 1, dolfinx.mesh.exterior_facet_indices(msh.topology))
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), bdofs, V)
    return msh, dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix=tag, petsc_options=opts)


def main() -> int:
    doc = " ".join((dolfinx.fem.petsc.LinearProblem.solve.__doc__ or "").split())
    quoted = "user is responsible for asserting convergence of the KSP solver"
    print(f"docstring_states_user_must_assert_convergence={quoted in doc}")

    _, p = problem("t2_ms6_lu_", {"ksp_type": "preonly", "pc_type": "lu"})
    p.solve()
    r_lu = p.solver.getConvergedReason()
    _, p = problem("t2_ms6_cg_", {"ksp_type": "cg", "pc_type": "jacobi",
                                  "ksp_max_it": 5000})
    p.solve()
    r_cg = p.solver.getConvergedReason()
    print(f"preonly_lu_reason={r_lu}")
    print(f"cg_jacobi_reason={r_cg}")

    bad = {"ksp_type": "gmres", "pc_type": "none", "ksp_max_it": 2,
           "ksp_rtol": 1e-14}
    if MUTATE:
        bad["ksp_error_if_not_converged"] = True
    print(f"ksp_error_if_not_converged_requested={MUTATE}")
    raised, reason, wrote = "", None, False
    msh, p = problem("t2_ms6_bad_", bad)
    try:
        Az = p.solve()
        reason = p.solver.getConvergedReason()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "az.xdmf")
            with dolfinx.io.XDMFFile(msh.comm, path, "w") as xf:
                xf.write_mesh(msh)
                xf.write_function(Az)
            wrote = os.path.getsize(path) > 0
    except Exception as exc:  # noqa: BLE001
        raised = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    print(f"failed_solve_raised={bool(raised)} {raised}".rstrip())
    print(f"failed_solve_reason={reason}")
    print(f"output_written_after_failed_solve={wrote}")
    healthy = r_lu > 0 and r_cg > 0
    print(f"healthy_configurations_report_positive_reasons={healthy}")
    print(f"failed_reason_is_negative={reason is not None and reason < 0}")
    if healthy and not raised and reason is not None and reason < 0 and wrote:
        print("VERDICT=solve_returned_normally_after_a_diverged_ksp")
        return 0
    print("VERDICT=solve_reported_the_failure_itself")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
