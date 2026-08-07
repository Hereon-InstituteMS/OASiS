"""Tier-2 for fenics time_dependent_heat#8: `T.x.array.min()` / `.max()` are
RANK-LOCAL and include ghost entries, and an unguarded `print` emits one copy per
rank. The fix is to slice to `T.x.array[:V.dofmap.index_map.size_local]`, reduce
with `comm.allreduce(..., MPI.MIN/MPI.MAX)` and guard the print with
`if comm.rank == 0:`.

This fixture is its own MPI worker: run without T2_TDH8_WORKER it drives two
subprocesses - one plain serial run and one `mpiexec -n 2` run of the same file -
and compares what they printed. The worker solves the transient heat problem on a
16x16 unit square (T = 1 on the left wall, 0 on the right, 40 backward-Euler
steps of dt = 0.05, i.e. essentially the steady state) and reports the range.

Observed: serial prints 'range_line T in [0.0000, 1.0000]'. Under mpiexec -n 2
the same unguarded code prints the line TWICE, once per rank, and neither copy is
the global range: the two lines are [0.0000, 0.8125] and [0.1875, 1.0000]. Both
local arrays are longer than the owned range (165 against 142 entries and 162
against 147), so the figures also include ghost entries. The guarded allreduce
over owned dofs prints one single line, [0.0000, 1.0000], matching serial.

Mutation control: T2_MUTATE=1 makes the worker use the guarded allreduce; the
duplicate line and the wrong ranges are then gone.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

N, DT, NSTEP = 16, 0.05, 40
CORRECT = os.environ.get("T2_MUTATE") == "1"


def worker() -> int:
    import numpy as np
    import ufl
    from mpi4py import MPI

    import dolfinx
    import dolfinx.fem.petsc

    comm = MPI.COMM_WORLD
    msh = dolfinx.mesh.create_unit_square(comm, N, N)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    T_n = dolfinx.fem.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt = dolfinx.fem.Constant(msh, DT)
    a = (u / dt) * v * ufl.dx + ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (T_n / dt) * v * ufl.dx
    left = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    right = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 1.0))
    bcs = [dolfinx.fem.dirichletbc(
               dolfinx.fem.Constant(msh, 1.0),
               dolfinx.fem.locate_dofs_topological(V, fdim, left), V),
           dolfinx.fem.dirichletbc(
               dolfinx.fem.Constant(msh, 0.0),
               dolfinx.fem.locate_dofs_topological(V, fdim, right), V)]
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=bcs, u=dolfinx.fem.Function(V),
        petsc_options_prefix="t2_tdh8_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    for _ in range(NSTEP):
        T_h = prob.solve()
        T_n.x.array[:] = T_h.x.array

    owned = V.dofmap.index_map.size_local
    print(f"rank={comm.rank} owned_dofs={owned} "
          f"local_array_length={T_h.x.array.size}", flush=True)
    if CORRECT:
        gmin = comm.allreduce(float(T_h.x.array[:owned].min()), op=MPI.MIN)
        gmax = comm.allreduce(float(T_h.x.array[:owned].max()), op=MPI.MAX)
        if comm.rank == 0:
            print(f"range_line T in [{gmin:.4f}, {gmax:.4f}]", flush=True)
    else:
        print(f"range_line T in [{float(T_h.x.array.min()):.4f}, "
              f"{float(T_h.x.array.max()):.4f}]", flush=True)
    return 0


def drive(nranks: int, cache: str) -> str:
    env = dict(os.environ)
    env["T2_TDH8_WORKER"] = "1"
    env["XDG_CACHE_HOME"] = cache
    cmd = [sys.executable, os.path.abspath(__file__)]
    if nranks > 1:
        mpiexec = os.path.join(os.path.dirname(sys.executable), "mpiexec")
        if not os.path.exists(mpiexec):
            print(f"mpiexec_missing_at={mpiexec}")
            return ""
        cmd = [mpiexec, "-n", str(nranks)] + cmd
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       timeout=200)
    if r.returncode != 0:
        print(f"worker_failed_rc={r.returncode}")
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
    return r.stdout


def main() -> int:
    cache = tempfile.mkdtemp(prefix="ffcx_t2_tdh8_")
    serial = drive(1, cache)
    two = drive(2, cache)
    pat = re.compile(r"range_line T in \[(-?\d+\.\d+), (-?\d+\.\d+)\]")
    ser = pat.findall(serial)
    par = pat.findall(two)
    ranks = re.findall(r"rank=(\d+) owned_dofs=(\d+) local_array_length=(\d+)",
                       two)
    print(f"guarded_allreduce_used_by_worker={CORRECT}")
    print(f"serial_range_line=T in [{ser[0][0]}, {ser[0][1]}]" if ser
          else "serial_range_line=absent")
    for lo, hi in par:
        print(f"two_rank_range_line=T in [{lo}, {hi}]")
    print(f"two_rank_range_line_count={len(par)}")
    for r, owned, total in ranks:
        print(f"two_rank_dofs rank={r} owned={owned} with_ghosts={total}")

    ghosts = any(int(t) > int(o) for _r, o, t in ranks)
    once_per_rank = len(par) == 2
    differs = bool(ser) and any((lo, hi) != ser[0] for lo, hi in par)
    matches = bool(ser) and len(par) == 1 and par[0] == ser[0]
    print(f"serial_range_is_the_full_data_range={bool(ser) and ser[0] == ('0.0000', '1.0000')}")
    print(f"ghost_entries_are_included_in_the_local_array={ghosts}")
    print(f"range_line_printed_once_per_rank={once_per_rank}")
    print(f"some_rank_range_differs_from_the_serial_range={differs}")
    print(f"guarded_reduction_prints_one_line_matching_serial={matches}")

    if (bool(ser) and ser[0] == ("0.0000", "1.0000") and ghosts
            and once_per_rank and differs):
        print("VERDICT=min_max_is_rank_local_and_the_print_is_duplicated")
        return 0
    print("VERDICT=serial_and_parallel_agreed")
    return 1


if __name__ == "__main__":
    if os.environ.get("T2_TDH8_WORKER") == "1":
        raise SystemExit(worker())
    raise SystemExit(main())
