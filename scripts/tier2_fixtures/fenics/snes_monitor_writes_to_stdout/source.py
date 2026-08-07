"""Tier-2 for fenics hyperelasticity#6: the PETSc SNES residual monitor is
switched on through the petsc_options dict of dolfinx.fem.petsc.NonlinearProblem
('snes_monitor': ''), and it writes to STDOUT, not stderr.

A hyperelastic (compressible Neo-Hookean) cantilever under a body force is
solved in a CHILD process whose stdout and stderr are captured separately,
because the two streams are indistinguishable once they are merged. Observed
signal: the child's stdout carries one line per Newton iteration of the form
"  0 SNES Function norm 2.304650717691e-02", its stderr carries none of them,
and the final state is read back from problem.solver.getConvergedReason() and
problem.solver.getIterationNumber(). The norm falls by orders of magnitude over
the run, which is what a healthy Newton looks like.

Mutation control: T2_MUTATE=1 drops 'snes_monitor' from petsc_options; the same
solve then prints no monitor line on either stream.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

MUTATE = os.environ.get("T2_MUTATE") == "1"
CHILD = os.environ.get("T2_SNES_CHILD") == "1"


def child() -> int:
    import numpy as np
    import ufl
    from mpi4py import MPI

    import dolfinx
    import dolfinx.fem.petsc

    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    d = msh.geometry.dim
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (d,)))
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    f_var = ufl.variable(ufl.Identity(d) + ufl.grad(u))
    j = ufl.det(f_var)
    mu, lmbda = 1.0, 10.0
    psi = (mu / 2) * (ufl.tr(f_var.T * f_var) - d) - mu * ufl.ln(j) \
        + (lmbda / 2) * ufl.ln(j) ** 2
    piola = ufl.diff(psi, f_var)
    body = dolfinx.fem.Constant(msh, (0.0, -0.2))
    res = ufl.inner(piola, ufl.grad(v)) * ufl.dx - ufl.inner(body, v) * ufl.dx
    msh.topology.create_connectivity(d - 1, d)
    left = dolfinx.mesh.locate_entities_boundary(
        msh, d - 1, lambda x: np.isclose(x[0], 0.0))
    bc = dolfinx.fem.dirichletbc(
        np.zeros(d), dolfinx.fem.locate_dofs_topological(V, d - 1, left), V)
    opts = {"snes_type": "newtonls", "ksp_type": "preonly", "pc_type": "lu"}
    if not MUTATE:
        opts["snes_monitor"] = ""
    problem = dolfinx.fem.petsc.NonlinearProblem(
        res, u, bcs=[bc], petsc_options_prefix="t2_hy6_", petsc_options=opts)
    problem.solve()
    print(f"CHILD_reason={problem.solver.getConvergedReason()} "
          f"CHILD_its={problem.solver.getIterationNumber()}", file=sys.stderr)
    return 0


def main() -> int:
    env = dict(os.environ)
    env["T2_SNES_CHILD"] = "1"
    run = subprocess.run([sys.executable, os.path.abspath(__file__)],
                         env=env, capture_output=True, text=True, timeout=240)
    out_lines = [ln for ln in run.stdout.splitlines()
                 if "SNES Function norm" in ln]
    err_lines = [ln for ln in run.stderr.splitlines()
                 if "SNES Function norm" in ln]
    print(f"child_returncode={run.returncode}")
    print(f"monitor_lines_on_stdout={len(out_lines)}")
    print(f"monitor_lines_on_stderr={len(err_lines)}")
    for ln in out_lines[:3]:
        print(f"stdout_sample|{ln}")
    state = [ln for ln in run.stderr.splitlines() if ln.startswith("CHILD_")]
    print(f"child_final_state={state[0] if state else '(none)'}")

    reason = its = -1
    if state:
        parts = dict(p.split("=") for p in state[0].split())
        reason = int(parts["CHILD_reason"])
        its = int(parts["CHILD_its"])
    print(f"snes_converged_reason={reason} snes_iterations={its}")

    drop = 0.0
    if len(out_lines) >= 2:
        first = float(out_lines[0].split()[-1])
        last = float(out_lines[-1].split()[-1])
        drop = first / last if last > 0.0 else float("inf")
        print(f"residual_drop_factor={drop:.3e}")

    on_stdout = len(out_lines) > 1 and len(err_lines) == 0
    print(f"monitor_goes_to_stdout_only={on_stdout}")
    print(f"monitor_line_count_matches_iterations="
          f"{len(out_lines) == its + 1}")
    healthy = drop > 1e3
    print(f"residual_falls_by_orders_of_magnitude={healthy}")
    if on_stdout and reason > 0 and len(out_lines) == its + 1 and healthy:
        print("VERDICT=snes_monitor_prints_one_stdout_line_per_iteration")
        return 0
    print("VERDICT=no_snes_monitor_output")
    return 1


if __name__ == "__main__":
    raise SystemExit(child() if CHILD else main())
