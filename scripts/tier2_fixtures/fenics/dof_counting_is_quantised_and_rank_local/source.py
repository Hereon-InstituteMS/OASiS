"""Tier-2 for fenics multiphase#3: do not measure the phase volume by counting
degrees of freedom (`np.sum(phi.x.array > 0) / len(phi.x.array)`). It is
quantised to the DOF grid, so it is far too coarse to see the interface move,
and in parallel it counts GHOST entries and is rank-local.

Shrinking Allen-Cahn droplet, 32x32 unit square, eps = 3h, 25 backward Euler
steps of dt = 1e-3. The DOF-count fraction changes in almost none of the steps
while the assembled integral of conditional(phi > 0, 1, 0) keeps moving. The
same field is then measured under mpirun -n 2 in a subprocess: the two ranks
print two different DOF-count fractions, neither equal to the serial one,
whereas the assembled integral reduced with comm.allreduce(..., MPI.SUM)
reproduces the serial value to round-off.

Mutation control: T2_MUTATE=1 measures the phase volume with the assembled
integral everywhere - serially and on both ranks - and the DOF-count findings
are then not produced at all.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N, NSTEP, DT, EPS_OVER_H, R = 32, 25, 1e-3, 3.0, 0.25

WORKER = r'''
import os, sys, tempfile
os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2w_"))
import numpy as np, ufl
from mpi4py import MPI
import dolfinx
NAIVE = sys.argv[1] == "naive"
N, EPS_OVER_H, R = %d, %f, %f
comm = MPI.COMM_WORLD
msh = dolfinx.mesh.create_unit_square(comm, N, N)
eps = EPS_OVER_H / N
V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
phi = dolfinx.fem.Function(V)
phi.interpolate(lambda x: np.tanh(
    (R - np.sqrt((x[0]-0.5)**2 + (x[1]-0.5)**2)) / (eps*np.sqrt(2.0))))
if NAIVE:
    frac = float(np.sum(phi.x.array > 0.0)) / len(phi.x.array)
    print("worker rank=%%d ranks=%%d naive_dof_fraction=%%.6f" %% (
        comm.rank, comm.size, frac), flush=True)
else:
    form = dolfinx.fem.form(ufl.conditional(ufl.gt(phi, 0.0), 1.0, 0.0)*ufl.dx)
    val = comm.allreduce(dolfinx.fem.assemble_scalar(form), MPI.SUM)
    print("worker rank=%%d ranks=%%d reduced_integral=%%.9f" %% (
        comm.rank, comm.size, val), flush=True)
''' % (N, EPS_OVER_H, R)


def run_parallel(kind: str, nranks: int = 2):
    tmp = tempfile.mkdtemp(prefix="t2_mp3_")
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


def serial_history():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    eps = EPS_OVER_H / N
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    phi, phi_n = dolfinx.fem.Function(V), dolfinx.fem.Function(V)

    def ic(x):
        d = R - np.sqrt((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2)
        return np.tanh(d / (eps * np.sqrt(2.0)))

    phi.interpolate(ic)
    phi_n.interpolate(ic)
    v = ufl.TestFunction(V)
    dt_c = dolfinx.fem.Constant(msh, DT)
    eps_c = dolfinx.fem.Constant(msh, eps)
    F = ((phi - phi_n) / dt_c * v * ufl.dx
         + eps_c * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
         + (1.0 / eps_c) * (phi ** 3 - phi) * v * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, phi, petsc_options_prefix="t2_mp3_",
        petsc_options={"snes_type": "newtonls", "ksp_type": "preonly",
                       "pc_type": "lu"})
    area = dolfinx.fem.form(
        ufl.conditional(ufl.gt(phi, 0.0), 1.0, 0.0) * ufl.dx)
    fracs, areas = [], []
    for step in range(NSTEP + 1):
        if step:
            prob.solve()
        fracs.append(float(np.sum(phi.x.array > 0.0)) / len(phi.x.array))
        areas.append(float(dolfinx.fem.assemble_scalar(area)))
        if step:
            phi_n.x.array[:] = phi.x.array
    return fracs, areas


def changes(seq) -> int:
    return sum(1 for i in range(len(seq) - 1) if seq[i + 1] != seq[i])


def main() -> int:
    fracs, areas = serial_history()
    n_frac, n_area = changes(fracs), changes(areas)
    print(f"serial_dof_fraction_first={fracs[0]:.6f} "
          f"last={fracs[-1]:.6f} steps_with_a_change={n_frac}")
    print(f"serial_integral_first={areas[0]:.6f} "
          f"last={areas[-1]:.6f} steps_with_a_change={n_area}")

    kind = "correct" if MUTATE else "naive"
    out = run_parallel(kind)
    if out is None:
        print("VERDICT=mpirun_unavailable")
        return 1
    for line in out.splitlines():
        if line.startswith("worker "):
            print(line)

    ok = n_area > 0
    if MUTATE:
        vals = [float(t.split("=")[1]) for line in out.splitlines()
                for t in line.split() if t.startswith("reduced_integral=")]
        agree = (len(vals) == 2 and abs(vals[0] - vals[1]) < 1e-12
                 and abs(vals[0] - areas[0]) < 1e-12)
        print(f"assembled_integral_is_rank_independent={agree}")
        print("VERDICT=integral_measure_is_grid_free_and_rank_independent")
        return 0 if (agree and ok) else 1

    vals = [float(t.split("=")[1]) for line in out.splitlines()
            for t in line.split() if t.startswith("naive_dof_fraction=")]
    two_ranks = len(vals) == 2
    differ = two_ranks and abs(vals[0] - vals[1]) > 1e-9
    off_serial = two_ranks and all(abs(v - fracs[0]) > 1e-9 for v in vals)
    frozen = n_frac <= NSTEP // 5
    print(f"dof_count_fraction_is_frozen_over_most_of_the_run={frozen}")
    print(f"integral_detects_more_motion_than_dof_counting={n_area > n_frac}")
    print(f"two_ranks_printed_two_different_dof_fractions={differ}")
    print(f"neither_rank_matched_the_serial_dof_fraction={off_serial}")
    if frozen and n_area > n_frac and differ and off_serial:
        print("VERDICT=dof_counting_is_quantised_and_rank_local")
        return 0
    print("VERDICT=dof_counting_was_adequate")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
