"""Tier-2 for fenics contact#2: a penalty stiffness that is too large makes
SNES diverge, and what it reports is a divergence-TOLERANCE trip, not any
condition-number message.

Wrong variant: the same penalty obstacle problem as contact#1 solved cold at
gamma = 1e12. The run is executed as a child process with every monitor PETSc
offers switched on (snes_monitor, ksp_monitor, snes_converged_reason,
ksp_converged_reason), so the parent can inspect the complete set of lines the
library emits rather than assert something about it.

Observed. The only lines PETSc produces are SNES function norms, KSP residual
norms, "Residual norms for <prefix> solve.", the linear converged-reason line
and the nonlinear converged-reason line. There is no condition-number warning
of any kind — the previously quoted "PETSc condition-number warning > 1e14"
does not exist. What does appear is
    Nonlinear t2_dtol_ solve did not converge due to DIVERGED_DTOL iterations 1
with getConvergedReason() = -9, while in the very same run the inner linear
solve reports
    Linear t2_dtol_ solve converged due to CONVERGED_ITS iterations 1
so the linear algebra is perfectly healthy. The mechanism is measured, not
assumed: the SNES function norm after the first Newton step is 1.053800e+10 at
gamma = 1e12 and 1.053800e+01 at gamma = 1e3 — it grows in direct proportion to
the penalty, which is why it crosses the default snes_divergence_tolerance of
10000 relative to the initial residual long before conditioning matters.

Mutation control: T2_MUTATE=1 runs the slot at the modest penalty, which
converges; the DIVERGED_DTOL line and the proportional residual growth are lost.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

MUTATE = os.environ.get("T2_MUTATE") == "1"

REF_GAMMA = 1.0e3
SLOT_GAMMA = REF_GAMMA if MUTATE else 1.0e12

CHILD = '''
import sys
import numpy as np, ufl
from mpi4py import MPI
from dolfinx import fem, mesh
import dolfinx.fem.petsc as dfp
G = float(sys.argv[1])
N = 24
msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
msh.topology.create_connectivity(1, 2)
V = fem.functionspace(msh, ("Lagrange", 1))
facets = mesh.exterior_facet_indices(msh.topology)
dofs = fem.locate_dofs_topological(V, 1, facets)
bc = fem.dirichletbc(0.0, dofs, V)
u = fem.Function(V)
v = ufl.TestFunction(V)
phi = fem.Constant(msh, -0.2)
gamma = fem.Constant(msh, G)
f = fem.Constant(msh, -10.0)
F = (ufl.dot(ufl.grad(u), ufl.grad(v)) - f*v
     - gamma*ufl.max_value(phi - u, 0.0)*v)*ufl.dx
p = dfp.NonlinearProblem(
    F, u, bcs=[bc], petsc_options_prefix="t2_dtol_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                   "snes_rtol": 1e-9, "snes_atol": 1e-10, "snes_max_it": 30,
                   "snes_linesearch_type": "basic",
                   "snes_monitor": None, "snes_converged_reason": None,
                   "ksp_monitor": None, "ksp_converged_reason": None})
raised = ""
try:
    p.solve()
except Exception as exc:
    raised = f"{type(exc).__name__}: {exc}"
print(f"CHILD_RAISED={raised!r}")
print(f"CHILD_REASON={p.solver.getConvergedReason()}")
print(f"CHILD_ITS={p.solver.getIterationNumber()}")
'''

KNOWN = (
    re.compile(r"^\s*\d+ SNES Function norm "),
    re.compile(r"^\s*\d+ KSP Residual norm "),
    re.compile(r"^\s*Residual norms for \S+ solve\.$"),
    re.compile(r"^\s*Linear \S+ solve (converged|did not converge) "),
    re.compile(r"^\s*Nonlinear \S+ solve (converged|did not converge) "),
)


def run(path, gamma):
    res = subprocess.run([sys.executable, path, repr(gamma)],
                         capture_output=True, text=True, timeout=600)
    return (res.stdout or "") + (res.stderr or "")


def first_step_norm(text):
    for line in text.splitlines():
        m = re.match(r"^\s*1 SNES Function norm (\S+)", line)
        if m:
            return float(m.group(1))
    return float("nan")


def field(text, key):
    for line in text.splitlines():
        if line.startswith(key):
            return line.split("=", 1)[1].strip()
    return ""


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "stiff_penalty.py")
        with open(path, "w") as fh:
            fh.write(CHILD)
        slot = run(path, SLOT_GAMMA)
        ref = run(path, REF_GAMMA)

    reason = int(field(slot, "CHILD_REASON") or 0)
    its = int(field(slot, "CHILD_ITS") or -1)
    raised = field(slot, "CHILD_RAISED")
    print(f"slot_gamma={SLOT_GAMMA:.0e} reason={reason} iterations={its} "
          f"raised={raised}")
    print(f"slot_reason_is_diverged_dtol={reason == -9}")
    print(f"slot_diverged_after_one_iteration={its == 1}")
    dtol = "solve did not converge due to DIVERGED_DTOL" in slot
    print(f"slot_reported_diverged_dtol_line={dtol}")
    ksp_ok = "solve converged due to CONVERGED_ITS" in slot
    print(f"inner_linear_solve_reported_healthy={ksp_ok}")
    for line in slot.splitlines():
        if "solve converged due to" in line or "did not converge due to" in line:
            print(f"petsc_said:{line.strip()}")

    both = slot + ref
    cond_lines = [ln for ln in both.splitlines() if "condition" in ln.lower()]
    print(f"lines_mentioning_condition_number={cond_lines}")
    print(f"no_condition_number_warning_anywhere={not cond_lines}")

    unknown = [ln for ln in slot.splitlines()
               if ln.strip() and not ln.startswith("CHILD_")
               and not any(p.match(ln) for p in KNOWN)]
    print(f"unclassified_petsc_lines={unknown}")
    print(f"only_monitor_and_reason_lines_emitted={not unknown}")

    n_slot, n_ref = first_step_norm(slot), first_step_norm(ref)
    ratio = n_slot / n_ref if n_ref else float("nan")
    print(f"first_newton_step_residual slot={n_slot:.6e} ref={n_ref:.6e} "
          f"ratio={ratio:.6e} gamma_ratio={SLOT_GAMMA / REF_GAMMA:.6e}")
    proportional = abs(ratio - SLOT_GAMMA / REF_GAMMA) <= 0.01 * (
        SLOT_GAMMA / REF_GAMMA)
    grew = ratio > 1.0e3
    print(f"residual_grows_in_proportion_to_the_penalty="
          f"{bool(proportional and grew)}")

    if (reason == -9 and its == 1 and dtol and ksp_ok and not cond_lines
            and not unknown and proportional and grew and raised == "''"):
        print("VERDICT=stiff_penalty_trips_dtol_no_conditioning_message")
        return 0
    print("VERDICT=stiff_penalty_reported_conditioning")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
