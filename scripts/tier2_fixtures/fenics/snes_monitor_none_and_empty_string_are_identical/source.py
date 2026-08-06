"""Tier-2 for fenics nonlinear_pde#6: the SNES iteration history is switched on
with "snes_monitor": None in petsc_options, the empty string does exactly the
same thing, and WITHOUT it a solve reports nothing but its final reason.

Wrong variant: no "snes_monitor" key at all. The run then produces zero
per-iteration lines, so a failure cannot be told apart from a too-large load by
looking at the output -- only the final integer reason is left.

Each arm runs in a CHILD process, because the monitor is written by PETSc at C
level onto file descriptor 1 and has to be captured there. Arms: monitor None,
monitor '', no monitor, and one diverging arm (Bratu lambda = 20 past the fold)
with the monitor on.

Observed on dolfinx 0.10.0 / PETSc 3.24: 'snes_monitor': None and
'snes_monitor': '' emit character-identical stdout, one line per iteration of
the form '  0 SNES Function norm 3.027343750000e-02', count = iterations + 1;
dropping the key gives zero such lines; the healthy arm's norm falls by many
orders of magnitude per iteration while the past-the-fold arm's norm stalls
(final/first > 0.1) before it gives up.

Mutation control: T2_MUTATE=1 gives the third arm the monitor as well, so the
zero-line observation is lost.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

MUTATE = os.environ.get("T2_MUTATE") == "1"
ARM = os.environ.get("T2_NP6_ARM", "")

N = 16


def child(arm: str) -> int:
    import ufl
    from mpi4py import MPI

    import dolfinx
    import dolfinx.fem.petsc

    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), dofs, V)
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    lmbda = 20.0 if arm == "fold" else 1.0
    lam = dolfinx.fem.Constant(msh, lmbda)
    F = (ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - lam * ufl.exp(u) * v * ufl.dx)
    opts = {"snes_type": "newtonls", "ksp_type": "preonly", "pc_type": "lu",
            "snes_max_it": 30, "snes_linesearch_type": "basic"}
    if arm == "none":
        opts["snes_monitor"] = None
    elif arm == "empty":
        opts["snes_monitor"] = ""
    elif arm == "off":
        if MUTATE:  # pathology removed: the monitor is on after all
            opts["snes_monitor"] = None
    elif arm == "fold":
        opts["snes_monitor"] = None
    problem = dolfinx.fem.petsc.NonlinearProblem(
        F, u, bcs=[bc], petsc_options_prefix=f"t2_np6_{arm}_",
        petsc_options=opts)
    problem.solve()
    print(f"CHILD_reason={problem.solver.getConvergedReason()} "
          f"CHILD_its={problem.solver.getIterationNumber()}", file=sys.stderr)
    return 0


def run_arm(arm: str):
    env = dict(os.environ)
    env["T2_NP6_ARM"] = arm
    r = subprocess.run([sys.executable, os.path.abspath(__file__)],
                       env=env, capture_output=True, text=True, timeout=240)
    lines = [ln for ln in r.stdout.splitlines() if "SNES Function norm" in ln]
    state = [ln for ln in r.stderr.splitlines() if ln.startswith("CHILD_")]
    reason = its = -99
    if state:
        parts = dict(p.split("=") for p in state[0].split())
        reason, its = int(parts["CHILD_reason"]), int(parts["CHILD_its"])
    print(f"arm={arm} returncode={r.returncode} monitor_lines={len(lines)} "
          f"reason={reason} iterations={its}")
    return lines, reason, its


def norms(lines):
    return [float(ln.split()[-1]) for ln in lines]


def main() -> int:
    l_none, r_none, it_none = run_arm("none")
    l_empty, r_empty, _ = run_arm("empty")
    l_off, r_off, _ = run_arm("off")
    l_fold, r_fold, it_fold = run_arm("fold")
    for ln in l_none[:2]:
        print(f"sample_none|{ln}")
    for ln in l_empty[:2]:
        print(f"sample_empty|{ln}")

    same = bool(l_none) and l_none == l_empty
    print(f"none_and_empty_string_produce_identical_lines={same}")
    print(f"monitor_line_count_matches_iterations={len(l_none) == it_none + 1}")
    print(f"no_monitor_arm_has_zero_monitor_lines={len(l_off) == 0}")
    print(f"reason_is_the_same_with_and_without_the_monitor="
          f"{r_none == r_empty == r_off}")

    n_ok = norms(l_none)
    n_fold = norms(l_fold)
    healthy = len(n_ok) > 1 and n_ok[-1] < 1.0e-6 * n_ok[0]
    stalls = len(n_fold) > 5 and n_fold[-1] > 0.1 * n_fold[0]
    print(f"healthy_run_residual_falls_by_orders_of_magnitude={healthy}")
    print(f"failing_run_residual_stalls={stalls}")
    print(f"fold_arm_reason={r_fold} fold_arm_iterations={it_fold}")

    if same and len(l_off) == 0 and healthy and stalls and r_none > 0:
        print("VERDICT=monitor_none_equals_empty_and_the_history_is_the_"
              "diagnosis")
        return 0
    print("VERDICT=monitor_output_did_not_differ")
    return 1


if __name__ == "__main__":
    raise SystemExit(child(ARM) if ARM else main())
