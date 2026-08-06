"""Tier-2 for fenics magnetostatics#6: `LinearProblem.solve()` never raises
when the KSP fails. The dolfinx 0.10 docstring says so outright, the failing
solve returns a Function full of whatever the Krylov method had reached, the
script goes on to WRITE ITS OUTPUT FILE and would exit 0. The only way to
notice is `problem.solver.getConvergedReason() > 0` (or passing
`"ksp_error_if_not_converged": True`).

Wrong variant: solve the 2D coil magnetostatics problem with CG capped at 2
iterations and trust the returned field. Observed: converged reason -3
(DIVERGED_MAX_IT), no exception, and the XDMF file is written anyway. Healthy
runs on the same problem report 4 (CONVERGED_ITS) for a direct solve and 2
(CONVERGED_RTOL) for CG.

Mutation control: T2_MUTATE=1 passes "ksp_error_if_not_converged": True, and
the same capped solve raises PETSc.Error instead of returning silently.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import inspect  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

MU0 = 4.0e-7 * np.pi
J0 = 1.0e6
R_COIL = 0.2


def build(n: int):
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
    return msh, a, L, [bc]


def run(prefix: str, opts: dict):
    msh, a, L, bcs = build(16)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix=prefix, petsc_options=opts)
    raised = ""
    az = None
    try:
        az = prob.solve()
        if isinstance(az, tuple):
            az = az[0]
    except Exception as exc:
        raised = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    return msh, prob, az, raised


def main() -> int:
    doc = inspect.getdoc(dolfinx.fem.petsc.LinearProblem.solve) or ""
    print("--- LinearProblem.solve docstring ---")
    print(doc)
    print("--- end docstring ---")
    flat = re.sub(r"\s+", " ", doc)
    doc_ok = ("user is responsible for asserting convergence of the KSP solver"
              in flat)
    print(f"docstring_says_user_must_assert_convergence={doc_ok}")

    failing = {"ksp_type": "cg", "pc_type": "none", "ksp_max_it": 2,
               "ksp_rtol": 1e-12}
    if MUTATE:
        failing["ksp_error_if_not_converged"] = True
    msh, prob, az, raised = run("t2_ms6_fail_", failing)
    reason = prob.solver.getConvergedReason()
    print(f"capped_cg_converged_reason={reason} "
          f"iterations={prob.solver.getIterationNumber()}")
    print(f"failing_solve_raised={bool(raised)}"
          + (f" {raised}" if raised else ""))
    print(f"failing_reason_is_negative={reason < 0}")
    print(f"failing_reason_is_diverged_max_it={reason == -3}")

    wrote = False
    if az is not None:
        out = os.path.join(tempfile.mkdtemp(prefix="t2_ms6_out_"), "az.xdmf")
        with dolfinx.io.XDMFFile(msh.comm, out, "w") as xf:
            xf.write_mesh(msh)
            xf.write_function(az)
        wrote = os.path.exists(out)
        print(f"max_Az_from_the_failed_solve={float(np.abs(az.x.array).max()):.6e}")
    print(f"wrote_output_after_a_failed_solve={wrote}")

    _, prob_lu, _, _ = run("t2_ms6_lu_",
                           {"ksp_type": "preonly", "pc_type": "lu"})
    r_lu = prob_lu.solver.getConvergedReason()
    _, prob_cg, _, _ = run("t2_ms6_cg_",
                           {"ksp_type": "cg", "pc_type": "jacobi",
                            "ksp_rtol": 1e-10, "ksp_max_it": 5000})
    r_cg = prob_cg.solver.getConvergedReason()
    print(f"healthy_reasons preonly={r_lu} cg={r_cg}")
    print(f"healthy_preonly_reason_is_4_converged_its={r_lu == 4}")
    print(f"healthy_cg_reason_is_2_converged_rtol={r_cg == 2}")

    if (doc_ok and not raised and reason < 0 and wrote
            and r_lu == 4 and r_cg == 2):
        print("VERDICT=ksp_failure_is_silent_only_getconvergedreason_shows_it")
        return 0
    print("VERDICT=ksp_failure_was_reported_by_solve")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
