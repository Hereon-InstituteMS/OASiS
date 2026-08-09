"""Tier-2 for fenics multiphase#4: `phi.x.array.min()` / `.max()` are
RANK-LOCAL and include ghost entries, and an unguarded `print` emits one copy of
every step line per rank.

A shrinking droplet placed off centre is written into a P1 function step by step
(interpolation only, so the field is bit-identical whatever the rank count) and
the step line is printed exactly as an unguarded template would. Under
mpirun -n 2 the run emits two copies of every step line and the two ranks report
different ranges, so the printed number is not the global range. Note what is
NOT reproducible: the claim says none of the per-rank ranges is the global one,
but with this partition rank 0 owns both extremes and its unguarded print is
accidentally right - only rank 1 is wrong. The local arrays do carry ghost
entries, which the worker reports.

Mutation control: T2_MUTATE=1 slices to `phi.x.array[:V.dofmap.index_map
.size_local]`, reduces with comm.allreduce(..., MPI.MIN / MPI.MAX) and guards
the print with `if comm.rank == 0:`; the duplication disappears and the reduced
range matches the serial range exactly.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N, NSTEP, EPS = 32, 3, 3.0 / 32

WORKER = r'''
import os, sys, tempfile
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2w_"))
import numpy as np
from mpi4py import MPI
import dolfinx
CORRECT = sys.argv[1] == "correct"
N, NSTEP, EPS = %d, %d, %.10f
comm = MPI.COMM_WORLD
msh = dolfinx.mesh.create_unit_square(comm, N, N)
V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
phi = dolfinx.fem.Function(V)
n_own = V.dofmap.index_map.size_local
print("worker ghosts rank=%%d owned=%%d local_len=%%d" %% (
    comm.rank, n_own, len(phi.x.array)), flush=True)
for step in range(NSTEP):
    r = 0.15 - 0.01*step
    phi.interpolate(lambda x: np.tanh(
        (r - np.sqrt((x[0]-0.25)**2 + (x[1]-0.5)**2)) / (EPS*np.sqrt(2.0))))
    if CORRECT:
        n = V.dofmap.index_map.size_local
        lo = comm.allreduce(float(phi.x.array[:n].min()), MPI.MIN)
        hi = comm.allreduce(float(phi.x.array[:n].max()), MPI.MAX)
        if comm.rank == 0:
            print("worker step=%%d phi in [%%.4f, %%.4f]" %% (step, lo, hi),
                  flush=True)
    else:
        print("worker step=%%d phi in [%%.4f, %%.4f]" %% (
            step, phi.x.array.min(), phi.x.array.max()), flush=True)
''' % (N, NSTEP, EPS)


def run_parallel(kind: str, nranks: int = 2):
    tmp = tempfile.mkdtemp(prefix="t2_mp4_")
    path = os.path.join(tmp, "worker.py")
    with open(path, "w") as fh:
        fh.write(WORKER)
    exe = sys.executable
    mpirun = os.path.join(os.path.dirname(exe), "mpirun")
    if not os.path.exists(mpirun):
        return None
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PMI_", "HYDRA_"))}
    env.pop("T2_MUTATE", None)
    try:
        res = subprocess.run([mpirun, "-n", str(nranks), exe, path, kind],
                             capture_output=True, text=True, env=env,
                             timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return res.stdout + res.stderr


def serial_ranges():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    phi = dolfinx.fem.Function(V)
    out = []
    for step in range(NSTEP):
        r = 0.15 - 0.01 * step
        phi.interpolate(lambda x, r=r: np.tanh(
            (r - np.sqrt((x[0] - 0.25) ** 2 + (x[1] - 0.5) ** 2))
            / (EPS * np.sqrt(2.0))))
        out.append((float(phi.x.array.min()), float(phi.x.array.max())))
    return out


def parse(out: str):
    got = []
    for line in out.splitlines():
        if line.startswith("worker step="):
            lo, hi = line.split("[")[1].rstrip("]").split(",")
            got.append((int(line.split("step=")[1].split()[0]),
                        float(lo), float(hi)))
    return got


def main() -> int:
    serial = serial_ranges()
    print(f"serial_step0_range=[{serial[0][0]:.4f}, {serial[0][1]:.4f}]")
    out = run_parallel("correct" if MUTATE else "naive")
    if out is None:
        print("VERDICT=mpirun_unavailable")
        return 1
    for line in out.splitlines():
        if line.startswith("worker "):
            print(line)
    got = parse(out)
    per_step = [sum(1 for g in got if g[0] == s) for s in range(NSTEP)]

    if MUTATE:
        one_each = all(c == 1 for c in per_step)
        matches = all(abs(g[1] - serial[g[0]][0]) < 1e-4
                      and abs(g[2] - serial[g[0]][1]) < 1e-4 for g in got)
        print(f"reduced_print_appears_once_per_step={one_each}")
        print(f"reduced_range_matches_the_serial_range={matches}")
        print("VERDICT=reduction_and_rank_guard_give_the_global_range")
        return 0 if (one_each and matches and got) else 1

    duplicated = all(c == 2 for c in per_step)
    step0 = [g for g in got if g[0] == 0]
    disagree = (len(step0) == 2
                and (abs(step0[0][1] - step0[1][1]) > 1e-4
                     or abs(step0[0][2] - step0[1][2]) > 1e-4))
    some_wrong = len(step0) == 2 and any(
        abs(g[1] - serial[0][0]) > 1e-4 or abs(g[2] - serial[0][1]) > 1e-4
        for g in step0)
    ghosts = [int(line.split("local_len=")[1]) - int(
        line.split("owned=")[1].split()[0])
        for line in out.splitlines() if line.startswith("worker ghosts")]
    has_ghosts = bool(ghosts) and all(g > 0 for g in ghosts)
    print(f"every_step_line_printed_once_per_rank={duplicated}")
    print(f"the_two_ranks_disagree_about_the_range={disagree}")
    print(f"at_least_one_rank_misreported_the_global_range={some_wrong}")
    print(f"local_array_carries_ghost_entries_on_every_rank={has_ghosts}")
    if duplicated and disagree and some_wrong and has_ghosts:
        print("VERDICT=local_min_max_is_rank_local_and_prints_duplicate")
        return 0
    print("VERDICT=unguarded_local_range_was_fine")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
