"""Tier-2 for fenics matrix_free_poisson#6: any scalar diagnostic must be reduced
-- `comm.allreduce(fem.assemble_scalar(fem.form(...)), op=MPI.SUM)` -- and
`u.x.array.min()/.max()` are RANK-LOCAL, so they need MPI.MIN / MPI.MAX over
`u.x.array[:nr]`.

Wrong variant: report fem.assemble_scalar(...) and u.x.array.min()/.max()
straight out, as one would in serial. Right variant: allreduce them.

This is an MPI-only failure, so the fixture is a driver: it writes one child
script and runs it on 1 rank and on 2 ranks, reporting the diagnostic raw and
reduced, through the mpirun that ships next to this interpreter.

Observed on dolfinx 0.10.0 / MPICH, 16x16 unit square, P2, converged matrix-free
solve: on 2 ranks the unreduced integral norm reads 2.944950e-01 against the
correct 3.291926e-01, i.e. about 11% too SMALL -- never an exception, never a NaN,
just a quietly optimistic number, and u.x.array.min() on rank 0 reads 9.375e-02
where the true global minimum is -6.7e-19. The allreduced value equals the serial
value to 1e-12. How much the raw number understates the truth depends only on how
many cells the reporting rank owns, so the ratio is not the 0.70 of the claim's
example; the sign of the error is structural.

Mutation control: T2_MUTATE=1 makes the run under test allreduce its diagnostics,
so the two-rank number matches the serial one and the understatement token goes
False.
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

REDUCE = os.environ["T2_REDUCE"] == "1"
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
its, conv = MAXIT, False
for k in range(MAXIT):
    action_A(p, yv)
    alpha = rn / gdot(p.array, yv.array)
    uh.x.array[:] += alpha * p.array
    r -= alpha * yv.array
    rn, rn_old = gdot(r, r), rn
    if rn / rn0 < RTOL ** 2:
        its, conv = k + 1, True
        break
    p.array[:] = (rn / rn_old) * p.array + r
uh.x.scatter_forward()
bc.set(uh.x.array, alpha=1.0)
uh.x.scatter_forward()

sq_form = fem.form(ufl.inner(uh, uh) * ufl.dx, dtype=dtype)
raw_sq = float(fem.assemble_scalar(sq_form))          # RANK-LOCAL
red_sq = comm.allreduce(raw_sq, op=MPI.SUM)           # reduced
raw_min = float(uh.x.array.min())                     # RANK-LOCAL, with ghosts
raw_max = float(uh.x.array.max())
red_min = comm.allreduce(float(uh.x.array[:nr].min()), MPI.MIN)
red_max = comm.allreduce(float(uh.x.array[:nr].max()), MPI.MAX)
reported_norm = (red_sq if REDUCE else raw_sq) ** 0.5
if comm.rank == 0:
    print("RESULT=" + json.dumps(dict(
        ranks=comm.size, reduce=REDUCE, converged=conv, iterations=its,
        reported_norm=reported_norm, raw_norm=raw_sq ** 0.5,
        reduced_norm=red_sq ** 0.5, raw_min=raw_min, raw_max=raw_max,
        reduced_min=red_min, reduced_max=red_max,
        finite=bool(np.isfinite(reported_norm)))))
'''

MUTATE = os.environ.get("T2_MUTATE") == "1"


def mpirun() -> str | None:
    cand = os.path.join(os.path.dirname(sys.executable), "mpirun")
    if os.path.exists(cand):
        return cand
    return shutil.which("mpirun")


def run(child: str, ranks: int, reduce_: bool) -> dict | None:
    env = os.environ.copy()
    env["T2_REDUCE"] = "1" if reduce_ else "0"
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
            print(f"ranks={ranks} reduce={reduce_} child_{line}")
            return json.loads(line[len("RESULT="):])
    print(f"ranks={ranks} reduce={reduce_} child_produced_no_result "
          f"rc={proc.returncode}")
    print((proc.stdout + proc.stderr)[-1500:])
    return None


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="t2mf6_")
    child = os.path.join(tmp, "child_matrix_free.py")
    with open(child, "w") as fh:
        fh.write(CHILD)
    print(f"mpirun={mpirun()}")

    under_test = MUTATE  # allreduce only in the mutant
    s1 = run(child, 1, under_test)
    p1 = run(child, 2, under_test)
    if s1 is None or p1 is None:
        print("VERDICT=children_did_not_run")
        return 1
    if MUTATE:
        print("mutation=scalar_diagnostics_allreduced")

    truth = s1["reported_norm"] if not MUTATE else s1["reduced_norm"]
    ratio = p1["reported_norm"] / truth
    print(f"serial_norm={truth:.6e} two_rank_reported_norm="
          f"{p1['reported_norm']:.6e} ratio={ratio:.4f}")
    both_converged = bool(s1["converged"]) and bool(p1["converged"])
    too_small = ratio < 0.99
    finite = bool(p1["finite"])
    reduced_ok = abs(p1["reduced_norm"] - s1["reduced_norm"]) < 1e-12
    local_extremum_wrong = (abs(p1["raw_min"] - p1["reduced_min"]) > 1e-9
                            or abs(p1["raw_max"] - p1["reduced_max"]) > 1e-9)
    serial_agrees = abs(s1["raw_norm"] - s1["reduced_norm"]) < 1e-14
    print(f"both_runs_converged={both_converged}")
    print(f"two_rank_reported_norm_is_smaller_than_the_serial_value={too_small}")
    print(f"the_too_small_number_is_finite_and_nothing_raised={finite}")
    print(f"rank_local_extremum_disagrees_with_the_reduced_one="
          f"{local_extremum_wrong}")
    print(f"allreduced_norm_equals_the_serial_norm={reduced_ok}")
    print(f"on_one_rank_raw_and_reduced_are_identical={serial_agrees}")
    if both_converged and too_small and finite and local_extremum_wrong \
            and reduced_ok and serial_agrees:
        print("VERDICT=unreduced_scalar_diagnostics_are_quietly_too_small")
        return 0
    print("VERDICT=unreduced_scalar_diagnostics_matched_the_serial_value")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
