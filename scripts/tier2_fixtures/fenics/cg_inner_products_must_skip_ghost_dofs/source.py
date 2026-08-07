"""Tier-2 for fenics matrix_free_poisson#5: the CG inner products must run over
OWNED DOFs only -- `nr = b.index_map.size_local` then
`comm.allreduce(np.vdot(v0[:nr], v1[:nr]), MPI.SUM)`. Using the full ghost-padded
array double-counts every shared DOF.

Wrong variant: gdot(v0, v1) = comm.allreduce(np.vdot(v0, v1), MPI.SUM) over the
whole local array including ghosts. Right variant: slice to [:nr] first.

This is an MPI-only failure, so the fixture is a driver: it writes one child
script and runs it on 1 rank and on 2 ranks, with the ghost-padded and the
owned-only inner product, through the mpirun that ships next to this interpreter.

Observed on dolfinx 0.10.0 / MPICH, 16x16 unit square, P2, rtol 1e-8, 200
iteration cap: on 1 rank the ghost-padded dot product is identical to the correct
one -- there are no ghosts -- and CG converges in 103 iterations. On 2 ranks the
same script never converges and rnorm/rnorm0 climbs monotonically past every
logged iteration, of order 1e2 at 100, 1e3 at 180 and above, and the returned
field reaches a magnitude of order 1e4 where the correct range is [0, 0.5]. The
magnitudes quoted in the claim (1.2e+04 at 100, 4.0e+13 at 180, field 3.3e+08)
are larger than this configuration gives, but the shape -- serial clean, two ranks
growing without bound, no exception anywhere -- is exactly as described.

Mutation control: T2_MUTATE=1 slices the inner product to the owned DOFs in the
run under test; the 2 rank run then converges and the growth tokens go False.
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

OWNED = os.environ["T2_OWNED"] == "1"
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
n_ghosts = b.array.size - nr

def gdot(v0, v1):
    if OWNED:
        return comm.allreduce(np.vdot(v0[:nr], v1[:nr]), MPI.SUM)
    return comm.allreduce(np.vdot(v0, v1), MPI.SUM)

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
    if (k + 1) in (100, 140, 180, 200):
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
if comm.rank == 0:
    print("RESULT=" + json.dumps(dict(ranks=comm.size, owned=OWNED,
                                      converged=conv, iterations=its,
                                      residual_log=log, umin=gmin, umax=gmax,
                                      rank0_ghosts=int(n_ghosts))))
'''

MUTATE = os.environ.get("T2_MUTATE") == "1"


def mpirun() -> str | None:
    cand = os.path.join(os.path.dirname(sys.executable), "mpirun")
    if os.path.exists(cand):
        return cand
    return shutil.which("mpirun")


def run(child: str, ranks: int, owned: bool) -> dict | None:
    env = os.environ.copy()
    env["T2_OWNED"] = "1" if owned else "0"
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
            print(f"ranks={ranks} owned={owned} child_{line}")
            return json.loads(line[len("RESULT="):])
    print(f"ranks={ranks} owned={owned} child_produced_no_result "
          f"rc={proc.returncode}")
    print((proc.stdout + proc.stderr)[-1500:])
    return None


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="t2mf5_")
    child = os.path.join(tmp, "child_matrix_free.py")
    with open(child, "w") as fh:
        fh.write(CHILD)
    print(f"mpirun={mpirun()}")

    under_test = MUTATE  # owned-only slicing only in the mutant
    s1 = run(child, 1, under_test)
    p1 = run(child, 2, under_test)
    ref = run(child, 2, True)
    if s1 is None or p1 is None or ref is None:
        print("VERDICT=children_did_not_run")
        return 1
    if MUTATE:
        print("mutation=inner_products_sliced_to_the_owned_dofs")

    seq = [p1["residual_log"][k] for k in sorted(p1["residual_log"], key=int)]
    ghosts = p1["rank0_ghosts"] > 0 and s1["rank0_ghosts"] == 0
    serial_ok = bool(s1["converged"])
    par_failed = not bool(p1["converged"])
    grew = len(seq) >= 2 and all(q > 1.0 for q in seq) and seq[-1] > 10.0 * seq[0]
    blown = max(abs(p1["umin"]), abs(p1["umax"])) > 1e3
    ref_ok = bool(ref["converged"]) and abs(ref["umax"] - s1["umax"]) < 1e-12
    print(f"one_rank_has_no_ghost_dofs_two_ranks_do={ghosts}")
    print(f"serial_run_converges_normally={serial_ok}")
    print(f"two_rank_run_never_converged={par_failed}")
    print(f"two_rank_residual_ratio_climbs_above_one_and_keeps_growing={grew}")
    print(f"two_rank_field_magnitude_exceeds_1e3={blown}")
    print(f"owned_only_inner_product_fixes_the_two_rank_run={ref_ok}")
    if ghosts and serial_ok and par_failed and grew and blown and ref_ok:
        print("VERDICT=ghost_padded_inner_products_double_count_and_cg_diverges")
        return 0
    print("VERDICT=ghost_padded_inner_products_were_harmless")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
