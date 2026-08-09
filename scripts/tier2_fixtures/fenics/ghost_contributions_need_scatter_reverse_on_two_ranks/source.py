"""Tier-2 for fenics matrix_free_poisson#4: `fem.assemble_vector(array, form)`
only sums the cells owned by the calling rank; ghost contributions must be
returned to their owner with `y.scatter_reverse(la.InsertMode.add)` after every
in-place assembly, including inside the operator action.

Wrong variant: an action_A that assembles in place and zeroes the Dirichlet rows
but never scatters the ghost contributions back. Right variant: the same action
with y.scatter_reverse(la.InsertMode.add) before the bc.set.

This is an MPI-only failure, so the fixture is a driver: it writes one child
script and runs it four times -- 1 rank and 2 ranks, with and without the reverse
scatter -- through the mpirun that ships next to this interpreter, and compares.

Observed on dolfinx 0.10.0 / MPICH, 16x16 unit square, P2, rtol 1e-8, 200
iteration cap: on 1 rank the missing scatter changes nothing at all, the run
converges in 103 iterations to the same field. On 2 ranks the same script never
reaches rtol -- it exits the loop at the 200 iteration cap with the residual ratio
stalled around 1e-5 -- and the field it returns is wrong, its maximum is about
50% above the correct 0.5. The claim's quoted range [-2.84e+02, +1.13e+03] is NOT
what this configuration produces: the field stays O(1), the failure is a stall
plus a wrong answer rather than an overflow. Putting the reverse scatter back
makes the 2 rank run converge in the same 103 iterations and agree with the
serial field to 1e-12.

Mutation control: T2_MUTATE=1 gives the run under test the reverse scatter, so
the 2 rank run converges and the divergence tokens go False.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

CHILD = r'''
import os, json
from mpi4py import MPI
import numpy as np
import dolfinx, ufl
from dolfinx import fem, la, mesh

SCATTER = os.environ["T2_SCATTER"] == "1"
dtype = dolfinx.default_scalar_type
comm = MPI.COMM_WORLD
N, DEG, MAXIT, RTOL = 16, 2, 200, 1e-8

msh = mesh.create_unit_square(comm, N, N)
V = fem.functionspace(msh, ("Lagrange", DEG))
tdim = msh.topology.dim
msh.topology.create_connectivity(tdim - 1, tdim)
facets = mesh.exterior_facet_indices(msh.topology)
bdofs = fem.locate_dofs_topological(V, tdim - 1, facets)
uD = fem.Function(V, dtype=dtype)
uD.interpolate(lambda x: 0.5 * x[0])
bc = fem.dirichletbc(uD, bdofs)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 10.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L_fem = fem.form(ufl.inner(f, v) * ufl.dx, dtype=dtype)
ui = fem.Function(V, dtype=dtype)
M_fem = fem.form(ufl.action(a, ui), dtype=dtype)

def action_A(xv, yv):
    ui.x.array[:] = xv.array
    ui.x.scatter_forward()
    yv.array[:] = 0.0
    fem.assemble_vector(yv.array, M_fem)
    if SCATTER:
        yv.scatter_reverse(la.InsertMode.add)
    bc.set(yv.array, alpha=0.0)

b = fem.assemble_vector(L_fem)
ui.x.array[:] = 0.0
bc.set(ui.x.array, alpha=-1.0)
fem.assemble_vector(b.array, M_fem)
b.scatter_reverse(la.InsertMode.add)
bc.set(b.array, alpha=0.0)
b.scatter_forward()
nr = b.index_map.size_local

def gdot(v0, v1):
    return comm.allreduce(np.vdot(v0[:nr], v1[:nr]), MPI.SUM)

uh = fem.Function(V, dtype=dtype)
yv = la.vector(b.index_map, 1, dtype)
action_A(uh.x, yv)
r = b.array - yv.array
p = la.vector(b.index_map, 1, dtype)
p.array[:] = r
rn0 = rn = gdot(r, r)
its, conv, log = MAXIT, False, {}
for k in range(MAXIT):
    action_A(p, yv)
    alpha = rn / gdot(p.array, yv.array)
    uh.x.array[:] += alpha * p.array
    r -= alpha * yv.array
    rn, rn_old = gdot(r, r), rn
    if (k + 1) in (100, 150, 200):
        log[k + 1] = float(np.sqrt(abs(rn / rn0)))
    if rn / rn0 < RTOL ** 2:
        its, conv = k + 1, True
        break
    p.array[:] = (rn / rn_old) * p.array + r
uh.x.scatter_forward()
bc.set(uh.x.array, alpha=1.0)
uh.x.scatter_forward()
gmin = comm.allreduce(float(uh.x.array[:nr].min()), MPI.MIN)
gmax = comm.allreduce(float(uh.x.array[:nr].max()), MPI.MAX)
l2 = comm.allreduce(float(fem.assemble_scalar(
    fem.form(ufl.inner(uh, uh) * ufl.dx, dtype=dtype))), MPI.SUM) ** 0.5
if comm.rank == 0:
    print("RESULT=" + json.dumps(dict(ranks=comm.size, scatter=SCATTER,
                                      converged=conv, iterations=its,
                                      residual_log=log, umin=gmin, umax=gmax,
                                      l2=l2)))
'''

MUTATE = os.environ.get("T2_MUTATE") == "1"


def mpirun() -> str | None:
    cand = os.path.join(os.path.dirname(sys.executable), "mpirun")
    if os.path.exists(cand):
        return cand
    return shutil.which("mpirun")


def run(child: str, ranks: int, scatter: bool) -> dict | None:
    env = os.environ.copy()
    env["T2_SCATTER"] = "1" if scatter else "0"
    env.pop("T2_MUTATE", None)
    cmd = [sys.executable, child]
    if ranks > 1:
        mp = mpirun()
        if mp is None:
            print("mpirun_not_found=True")
            return None
        cmd = [mp, "-n", str(ranks), sys.executable, child]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=240)
    for line in (proc.stdout + proc.stderr).splitlines():
        if line.startswith("RESULT="):
            print(f"ranks={ranks} scatter={scatter} child_{line}")
            return json.loads(line[len("RESULT="):])
    print(f"ranks={ranks} scatter={scatter} child_produced_no_result "
          f"rc={proc.returncode}")
    print((proc.stdout + proc.stderr)[-1500:])
    return None


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="t2mf4_")
    child = os.path.join(tmp, "child_matrix_free.py")
    with open(child, "w") as fh:
        fh.write(CHILD)
    print(f"mpirun={mpirun()}")

    under_test = MUTATE  # scatter_reverse present only in the mutant
    s1 = run(child, 1, under_test)
    p1 = run(child, 2, under_test)
    ref = run(child, 2, True)
    if s1 is None or p1 is None or ref is None:
        print("VERDICT=children_did_not_run")
        return 1
    if MUTATE:
        print("mutation=scatter_reverse_present_inside_the_operator_action")

    serial_ok = bool(s1["converged"])
    par_failed = not bool(p1["converged"])
    wrong = abs(p1["umax"] - s1["umax"]) > 0.1 * abs(s1["umax"])
    ref_ok = bool(ref["converged"]) and abs(ref["umax"] - s1["umax"]) < 1e-12
    print(f"serial_run_is_unaffected={serial_ok}")
    print(f"two_rank_run_missed_rtol_within_the_iteration_cap={par_failed}")
    print(f"two_rank_field_max_differs_from_serial_by_over_ten_percent={wrong}")
    print(f"two_rank_run_with_scatter_reverse_matches_serial={ref_ok}")
    if serial_ok and par_failed and wrong and ref_ok:
        print("VERDICT=missing_scatter_reverse_breaks_only_the_parallel_run")
        return 0
    print("VERDICT=missing_scatter_reverse_was_harmless_on_two_ranks")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
