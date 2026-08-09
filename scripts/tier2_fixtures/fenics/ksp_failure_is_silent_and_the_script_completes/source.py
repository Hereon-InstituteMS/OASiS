"""Tier-2 for fenics time_dependent_heat#2: `ksp.solve()` and
`LinearProblem.solve()` never raise on solver failure - the dolfinx 0.10
docstring says outright that the user is responsible for asserting convergence.
Assert `getConvergedReason() > 0` every step, or pass
`"ksp_error_if_not_converged": True`.

One backward Euler heat step on a 16x16 unit square (T = 1 on the left wall).
The crippled solve is Richardson with ksp_max_it = 1 and rtol = 1e-14: it returns
normally, the returned Function is a Function, the converged reason is negative
(DIVERGED_MAX_IT), and the script goes on to write an XDMF output file and would
exit 0. For reference the same step is solved with 'preonly' (reason
CONVERGED_ITS) and with CG (reason CONVERGED_RTOL), both positive. The remedy is
checked too: with "ksp_error_if_not_converged": True the same crippled solve
raises petsc4py.PETSc.Error carrying "KSPSolve has not converged".

Mutation control: T2_MUTATE=1 selects a properly converging CG for the checked
solve; the negative reason disappears.
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
import dolfinx.io  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from petsc4py import PETSc  # noqa: E402

N, DT = 16, 0.01
REASONS = {v: k for k, v in PETSc.KSP.ConvergedReason.__dict__.items()
           if isinstance(v, int)}
CRIPPLED = {"ksp_type": "richardson", "pc_type": "none",
            "ksp_max_it": 1, "ksp_rtol": 1e-14}
HEALTHY_CG = {"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12,
              "ksp_max_it": 500}
DIRECT = {"ksp_type": "preonly", "pc_type": "lu"}


def one_step(options: dict, prefix: str):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    T_n = dolfinx.fem.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt_c = dolfinx.fem.Constant(msh, DT)
    a = (u / dt_c) * v * ufl.dx + ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (T_n / dt_c) * v * ufl.dx
    left = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    bcs = [dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(msh, 1.0),
        dolfinx.fem.locate_dofs_topological(V, fdim, left), V)]
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix=prefix, petsc_options=options)
    raised = "none"
    out = None
    try:
        out = prob.solve()
    except Exception as exc:  # noqa: BLE001
        raised = " ".join(f"{type(exc).__name__}: {exc}".split())
    if isinstance(out, tuple):
        out = out[0]
    reason = prob.solver.getConvergedReason()
    return msh, out, reason, raised


def main() -> int:
    doc = " ".join((dolfinx.fem.petsc.LinearProblem.solve.__doc__ or "").split())
    print(f"docstring_says: {doc}")

    sel = HEALTHY_CG if MUTATE else CRIPPLED
    msh, T_h, reason, raised = one_step(sel, "t2_tdh2_sel_")
    name = REASONS.get(reason)
    print(f"selected_solver={sel['ksp_type']} converged_reason={reason} "
          f"reason_name={name} raised={raised}")
    is_function = isinstance(T_h, dolfinx.fem.Function)
    print(f"solve_returned_a_function={is_function}")
    print(f"solve_raised_nothing={raised == 'none'}")
    print(f"selected_reason_is_negative={reason < 0}")

    # ... and the script carries on and writes its output file.
    path = os.path.join(tempfile.mkdtemp(prefix="t2_tdh2_out_"), "T.xdmf")
    with dolfinx.io.XDMFFile(msh.comm, path, "w") as xf:
        xf.write_mesh(msh)
        xf.write_function(T_h)
    wrote = os.path.exists(path) and os.path.getsize(path) > 0
    print(f"output_file_written_after_the_failed_solve={wrote}")

    _, _, r_direct, _ = one_step(DIRECT, "t2_tdh2_lu_")
    _, _, r_cg, _ = one_step(HEALTHY_CG, "t2_tdh2_cg_")
    print(f"preonly_reason_name={REASONS.get(r_direct)} "
          f"cg_reason_name={REASONS.get(r_cg)}")
    healthy = r_direct > 0 and r_cg > 0
    print(f"healthy_solvers_report_positive_reasons={healthy}")

    guarded = dict(sel)
    guarded["ksp_error_if_not_converged"] = True
    _, _, _, raised_guarded = one_step(guarded, "t2_tdh2_err_")
    print(f"with_ksp_error_if_not_converged: {raised_guarded}")
    remedy = raised_guarded != "none"
    print(f"remedy_raises={remedy}")

    if (is_function and raised == "none" and reason < 0 and wrote
            and healthy and remedy):
        print("VERDICT=ksp_failure_is_silent_and_the_script_completes")
        return 0
    print("VERDICT=solver_failure_was_reported")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
