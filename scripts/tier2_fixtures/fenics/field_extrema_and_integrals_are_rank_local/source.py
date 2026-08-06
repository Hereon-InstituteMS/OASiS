"""Tier-2 for fenics magnetostatics#9: field extrema and integrals are per-rank
quantities. `B.x.array.max()` is rank-local and includes ghost entries, and
`fem.assemble_scalar` returns only this rank's share of the integral. Nothing
errors; the numbers just quietly depend on how many ranks you happened to use.

This fixture is itself a single process. It writes a helper script and launches
it under `mpiexec -n 2` with the same interpreter, then reads the per-rank lines
back. The magnetostatics quantity being reported is max|B| for B = curl(Az) on
the coil problem, plus the total current integral int Jz dx. The coil is small and sits in the
corner at (-0.35, -0.35) so that the peak of |B| falls inside one rank's
partition and is not even visible in the other rank's ghost layer.

Wrong variant: the helper reports B.x.array.max() and
fem.assemble_scalar(...) as they come. Observed on dolfinx 0.10.0: the two
ranks print different maxima and different "total currents", neither integral
being the real one (they sum to it), and no warning of any kind.

FINDING against the claim text: the claim says "none of them is the true global
maximum". That is too strong -- the local array carries ghost entries, so one
rank's rank-local maximum usually IS the global maximum while the other rank's
is below it. The fixture pins the measured version: the ranks disagree and at
least one of them is wrong.

Mutation control: T2_MUTATE=1 makes the helper reduce first --
comm.allreduce(owned_max, MPI.MAX) and comm.allreduce(assemble_scalar,
MPI.SUM) -- and then every rank reports the same, correct numbers.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

MUTATE = os.environ.get("T2_MUTATE") == "1"

HELPER = '''
import os
import numpy as np
import ufl
from mpi4py import MPI
import dolfinx

MUTATE = os.environ.get("T2_MUTATE") == "1"
MU0 = 4.0e-7 * np.pi
J0 = 1.0e6
R_COIL = 0.07

comm = MPI.COMM_WORLD
msh = dolfinx.mesh.create_rectangle(
    comm, [np.array([-0.5, -0.5]), np.array([0.5, 0.5])], [24, 24])
tdim = msh.topology.dim
ncells = msh.topology.index_map(tdim).size_local
mid = dolfinx.mesh.compute_midpoints(
    msh, tdim, np.arange(ncells, dtype=np.int32)).T
DG0 = dolfinx.fem.functionspace(msh, ("DG", 0))
Jz = dolfinx.fem.Function(DG0)
Jz.x.array[:] = 0.0
Jz.x.array[:ncells][((mid[0] + 0.35) ** 2
                    + (mid[1] + 0.35) ** 2) < R_COIL ** 2] = J0
V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = (1.0 / MU0) * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = Jz * v * ufl.dx
msh.topology.create_connectivity(tdim - 1, tdim)
bdofs = dolfinx.fem.locate_dofs_topological(
    V, tdim - 1, dolfinx.mesh.exterior_facet_indices(msh.topology))
bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), bdofs, V)
import dolfinx.fem.petsc
prob = dolfinx.fem.petsc.LinearProblem(
    a, L, bcs=[bc], petsc_options_prefix="t2_ms9_",
    petsc_options={"ksp_type": "cg", "pc_type": "hypre", "ksp_rtol": 1e-10})
Az = prob.solve()
if isinstance(Az, tuple):
    Az = Az[0]
W = dolfinx.fem.functionspace(msh, ("DG", 0, (2,)))
B = dolfinx.fem.Function(W)
B.interpolate(dolfinx.fem.Expression(
    ufl.curl(Az), W.element.interpolation_points))

arr = B.x.array.reshape(-1, 2)
mag = np.sqrt((arr ** 2).sum(axis=1))
im = W.dofmap.index_map
owned = im.size_local
naive_max = float(mag.max())                       # includes ghost entries
owned_max = float(mag[:owned].max()) if owned else -np.inf
naive_int = float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jz * ufl.dx)))

global_max = comm.allreduce(owned_max, MPI.MAX)
global_int = comm.allreduce(naive_int, MPI.SUM)
rep_max = global_max if MUTATE else naive_max
rep_int = global_int if MUTATE else naive_int
print("T2RANK rank=%d size=%d owned=%d ghosts=%d array_rows=%d "
      "reported_max=%.12e reported_integral=%.12e global_max=%.12e "
      "global_integral=%.12e" % (comm.rank, comm.size, owned, im.num_ghosts,
                                 arr.shape[0], rep_max, rep_int, global_max,
                                 global_int), flush=True)
'''


def main() -> int:
    here = os.path.dirname(sys.executable)
    mpiexec = os.path.join(here, "mpiexec")
    if not os.path.exists(mpiexec):
        mpiexec = shutil.which("mpiexec") or ""
    print(f"mpiexec={mpiexec}")
    if not mpiexec:
        print("VERDICT=could_not_launch_two_ranks")
        return 1

    tmp = tempfile.mkdtemp(prefix="t2_ms9_")
    helper = os.path.join(tmp, "helper.py")
    with open(helper, "w") as fh:
        fh.write(HELPER)
    env = os.environ.copy()
    if MUTATE:
        env["T2_MUTATE"] = "1"
    else:
        env.pop("T2_MUTATE", None)
    proc = subprocess.run([mpiexec, "-n", "2", sys.executable, "-u", helper],
                          capture_output=True, text=True, timeout=400, env=env)
    out = proc.stdout + proc.stderr
    rows = []
    for line in out.splitlines():
        if line.startswith("T2RANK"):
            print(line)
            rows.append({k: v for k, v in
                         re.findall(r"(\w+)=(-?[\w.+-]+)", line)})
    print(f"ranks_that_reported={len(rows)}")
    if len(rows) != 2:
        print("mpi_child_output_tail=" + out.strip().splitlines()[-1][:200]
              if out.strip() else "mpi_child_output_tail=<empty>")
        print("VERDICT=could_not_launch_two_ranks")
        return 1

    maxima = [float(r["reported_max"]) for r in rows]
    integrals = [float(r["reported_integral"]) for r in rows]
    gmax = float(rows[0]["global_max"])
    gint = float(rows[0]["global_integral"])
    ghosts = [int(r["ghosts"]) for r in rows]
    owned = [int(r["owned"]) for r in rows]
    rowsn = [int(r["array_rows"]) for r in rows]
    print(f"reported_maxima={maxima}")
    print(f"reported_integrals={integrals}")
    print(f"global_max={gmax:.12e} global_integral={gint:.12e}")

    print(f"array_includes_ghost_entries="
          f"{all(g > 0 for g in ghosts) and all(r > o for r, o in zip(rowsn, owned))}")
    diff_max = abs(maxima[0] - maxima[1]) > 1e-3 * abs(gmax)
    diff_int = abs(integrals[0] - integrals[1]) > 1e-3 * abs(gint)
    print(f"ranks_report_different_maxima={diff_max}")
    print(f"ranks_report_different_integrals={diff_int}")
    wrong = sum(1 for m in maxima if abs(m - gmax) > 1e-3 * abs(gmax))
    print(f"ranks_whose_max_is_not_the_global_max={wrong}")
    print(f"at_least_one_rank_reports_a_wrong_max={wrong >= 1}")
    sums = abs(sum(integrals) - gint) <= 1e-9 * abs(gint)
    print(f"per_rank_integrals_sum_to_the_global_one={sums}")
    print(f"child_returncode={proc.returncode}")

    if (proc.returncode == 0 and all(g > 0 for g in ghosts) and diff_max
            and diff_int and wrong >= 1 and sums):
        print("VERDICT=extrema_and_integrals_are_rank_local_without_a_reduction")
        return 0
    print("VERDICT=every_rank_reported_the_same_global_numbers")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
