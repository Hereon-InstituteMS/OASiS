"""Tier-2 for fenics time_dependent_heat#8: `T.x.array.min()` / `.max()` are
RANK-LOCAL and include ghost entries, and an unguarded `print` emits one copy per
rank. Slice to `T.x.array[:V.dofmap.index_map.size_local]`, reduce with
`comm.allreduce(..., MPI.MIN/MPI.MAX)` and guard the print with
`if comm.rank == 0:`.

32x32 unit square, T = 0 initially, a hot Dirichlet patch (T = 1) on the part of
the left wall below y = 0.25, everything else insulated, 5 backward Euler steps of
dt = 0.01 solved with CG so the run works on any number of ranks. The step line
is printed exactly as an unguarded template would print it. Under mpirun -n 2 the
line appears twice per step with different numbers, and the rank that does not own
the hot patch reports a maximum far below the prescribed 1 - the global range the
serial run prints.

Mutation control: T2_MUTATE=1 reduces over owned dofs and guards the print; one
line per step appears and it matches the serial range.
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

N, NSTEP, DT = 32, 5, 0.01

BODY = r'''
import os, sys, tempfile
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2w_"))
import numpy as np, ufl
from mpi4py import MPI
import dolfinx, dolfinx.fem.petsc
CORRECT = len(sys.argv) > 1 and sys.argv[1] == "correct"
N, NSTEP, DT = %d, %d, %f
comm = MPI.COMM_WORLD
msh = dolfinx.mesh.create_unit_square(comm, N, N)
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(fdim, tdim)
V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
T_n = dolfinx.fem.Function(V)
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
dt_c = dolfinx.fem.Constant(msh, DT)
a = (u/dt_c)*v*ufl.dx + ufl.dot(ufl.grad(u), ufl.grad(v))*ufl.dx
L = (T_n/dt_c)*v*ufl.dx
patch = dolfinx.mesh.locate_entities_boundary(
    msh, fdim, lambda x: np.isclose(x[0], 0.0) & (x[1] < 0.25))
bcs = [dolfinx.fem.dirichletbc(
    dolfinx.fem.Constant(msh, 1.0),
    dolfinx.fem.locate_dofs_topological(V, fdim, patch), V)]
prob = dolfinx.fem.petsc.LinearProblem(
    a, L, bcs=bcs, petsc_options_prefix="t2_tdh8_",
    petsc_options={"ksp_type": "cg", "pc_type": "jacobi", "ksp_rtol": 1e-12,
                   "ksp_max_it": 2000})
for step in range(NSTEP):
    T_h = prob.solve()
    if isinstance(T_h, tuple):
        T_h = T_h[0]
    if CORRECT:
        n = V.dofmap.index_map.size_local
        lo = comm.allreduce(float(T_h.x.array[:n].min()), MPI.MIN)
        hi = comm.allreduce(float(T_h.x.array[:n].max()), MPI.MAX)
        if comm.rank == 0:
            print("probe step=%%d T in [%%.4f, %%.4f]" %% (step, lo, hi),
                  flush=True)
    else:
        print("probe step=%%d T in [%%.4f, %%.4f]" %% (
            step, T_h.x.array.min(), T_h.x.array.max()), flush=True)
    T_n.x.array[:] = T_h.x.array
''' % (N, NSTEP, DT)


def write_worker() -> str:
    path = os.path.join(tempfile.mkdtemp(prefix="t2_tdh8_"), "worker.py")
    with open(path, "w") as fh:
        fh.write(BODY)
    return path


def run(args: list[str], nranks: int = 1):
    path = write_worker()
    exe = sys.executable
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PMI_", "HYDRA_"))}
    env.pop("T2_MUTATE", None)
    cmd = [exe, path] + args
    if nranks > 1:
        mpirun = os.path.join(os.path.dirname(exe), "mpirun")
        if not os.path.exists(mpirun):
            return None
        cmd = [mpirun, "-n", str(nranks), exe, path] + args
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, env=env,
                             timeout=180)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return res.stdout + res.stderr


def parse(out: str):
    rows = []
    for line in out.splitlines():
        if line.startswith("probe step="):
            lo, hi = line.split("[")[1].rstrip("]").split(",")
            rows.append((int(line.split("step=")[1].split()[0]),
                         float(lo), float(hi)))
    return rows


def main() -> int:
    ser = run([], 1)
    par = run(["correct" if MUTATE else "naive"], 2)
    if ser is None or par is None:
        print("VERDICT=mpirun_unavailable")
        return 1
    s_rows = parse(ser)
    p_rows = parse(par)
    print(f"serial_line: T in [{s_rows[-1][1]:.4f}, {s_rows[-1][2]:.4f}]")
    for line in par.splitlines():
        if line.startswith("probe "):
            print(f"mpirun2 {line}")
    last = [r for r in p_rows if r[0] == NSTEP - 1]
    ref = s_rows[-1]

    if MUTATE:
        one = len(last) == 1
        match = one and abs(last[0][1] - ref[1]) < 1e-4 and abs(
            last[0][2] - ref[2]) < 1e-4
        print(f"reduced_print_appears_once_per_step={one}")
        print(f"reduced_range_matches_the_serial_range={match}")
        print("VERDICT=owned_slice_plus_allreduce_gives_the_global_range")
        return 0 if match else 1

    dup = len(last) == 2
    disagree = dup and (abs(last[0][1] - last[1][1]) > 1e-4
                        or abs(last[0][2] - last[1][2]) > 1e-4)
    wrong = dup and any(abs(r[1] - ref[1]) > 1e-4 or abs(r[2] - ref[2]) > 1e-4
                        for r in last)
    print(f"every_step_line_printed_once_per_rank={dup}")
    print(f"the_two_ranks_disagree_about_the_range={disagree}")
    print(f"at_least_one_rank_misreported_the_global_range={wrong}")
    print(f"serial_run_sees_the_prescribed_maximum="
          f"{abs(ref[2] - 1.0) < 1e-9}")
    if dup and disagree and wrong and abs(ref[2] - 1.0) < 1e-9:
        print("VERDICT=unguarded_local_min_max_is_not_the_global_range")
        return 0
    print("VERDICT=unguarded_local_range_was_fine")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
