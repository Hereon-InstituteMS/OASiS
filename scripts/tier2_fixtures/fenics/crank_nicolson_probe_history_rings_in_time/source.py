"""Tier-2 for fenics time_dependent_heat#5: Crank-Nicolson (theta = 0.5) rings in
TIME when the initial condition is incompatible with the boundary data, and the
ringing does not go away with mesh refinement. Backward Euler (theta = 1) is
L-stable and does not ring.

Unit square, T = 0 initially, the left wall stepped from 0 to 1 at t = 0, no
source, dt = 8*h^2 with h = 1/32 so that k*dt/h^2 = 8. One interior point, one
cell from the wall, is probed every step with `dolfinx.fem.Function.eval`
against a `dolfinx.geometry.bb_tree` cell candidate - exactly the detector the
claim prescribes - and the same run is repeated on a 16x16 mesh to show
refinement does not help.

Observed on 32x32: Crank-Nicolson goes 0.6033, 0.9148, 0.8291, 0.9274, 0.8814
over the first five steps - ten sign changes in the increments over twelve steps
- while backward Euler is monotone 0.7009, 0.8268, 0.8685, 0.8901, 0.9037 with
zero sign changes. The oscillation stays inside [0, 1] (the measured field range
is exactly [0.0000, 1.0000] for both schemes), so no bounds check and no min/max
print can catch it; only the probe history shows it. The ringing is present on
16x16 as well as 32x32 at the same dt, so refinement does not remove it.

Mutation control: T2_MUTATE=1 selects theta = 1 (backward Euler) as the checked
scheme; the probe history is then monotone.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402
import dolfinx.geometry  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

NFINE, NCOARSE, NSTEP = 32, 16, 12
DT = 8.0 / NFINE ** 2          # k*dt/h^2 = 8 on the fine mesh


def probe_history(n: int, theta: float, tag: str):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    T_n = dolfinx.fem.Function(V)          # T = 0 initially
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    dt = dolfinx.fem.Constant(msh, DT)
    th = dolfinx.fem.Constant(msh, float(theta))
    om = dolfinx.fem.Constant(msh, float(1.0 - theta))
    a = (u / dt) * v * ufl.dx + th * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ((T_n / dt) * v * ufl.dx
         - om * ufl.dot(ufl.grad(T_n), ufl.grad(v)) * ufl.dx)
    left = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(msh, 1.0),
        dolfinx.fem.locate_dofs_topological(V, fdim, left), V)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], u=dolfinx.fem.Function(V),
        petsc_options_prefix=f"t2_tdh5_{tag}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    point = np.array([[1.0 / n, 0.5, 0.0]], dtype=np.float64)
    tree = dolfinx.geometry.bb_tree(msh, tdim)
    cand = dolfinx.geometry.compute_collisions_points(tree, point)
    cells = dolfinx.geometry.compute_colliding_cells(msh, cand, point)
    cell = cells.links(0)[:1]

    hist, lo, hi = [], 0.0, 0.0
    for _ in range(NSTEP):
        T_h = prob.solve()
        T_n.x.array[:] = T_h.x.array
        hist.append(float(T_h.eval(point, cell)[0]))
        lo = min(lo, float(T_h.x.array.min()))
        hi = max(hi, float(T_h.x.array.max()))
    return np.array(hist), lo, hi


def sign_changes(hist) -> int:
    d = np.diff(hist)
    return int(np.sum(d[:-1] * d[1:] < 0))


def main() -> int:
    out = {}
    for n in (NCOARSE, NFINE):
        for name, theta in (("crank_nicolson", 0.5), ("backward_euler", 1.0)):
            hist, lo, hi = probe_history(n, theta, f"{name}_{n}")
            out[(n, name)] = (hist, lo, hi, sign_changes(hist))
            print(f"mesh={n}x{n} scheme={name} dt_over_h2={DT * n * n:.2f} "
                  f"probe_first5={np.array2string(hist[:5], precision=4)} "
                  f"increment_sign_changes={out[(n, name)][3]} "
                  f"field_range=[{lo:.4f}, {hi:.4f}]")

    cn, be = out[(NFINE, "crank_nicolson")], out[(NFINE, "backward_euler")]
    print(f"cn_probe_history_is_non_monotone={cn[3] >= 2}")
    print(f"be_probe_history_is_monotone={be[3] == 0}")
    print(f"cn_never_leaves_the_zero_one_range="
          f"{cn[1] >= -1e-12 and cn[2] <= 1.0 + 1e-12}")
    print(f"cn_rings_on_both_meshes="
          f"{out[(NCOARSE, 'crank_nicolson')][3] >= 2 and cn[3] >= 2}")
    print(f"be_monotone_on_both_meshes="
          f"{out[(NCOARSE, 'backward_euler')][3] == 0 and be[3] == 0}")

    sel = "backward_euler" if MUTATE else "crank_nicolson"
    print(f"selected_scheme={sel}")
    print(f"selected_probe_history_is_non_monotone={out[(NFINE, sel)][3] >= 2}")

    if (out[(NFINE, sel)][3] >= 2 and cn[3] >= 2 and be[3] == 0
            and cn[1] >= -1e-12 and cn[2] <= 1.0 + 1e-12
            and out[(NCOARSE, "crank_nicolson")][3] >= 2
            and out[(NCOARSE, "backward_euler")][3] == 0):
        print("VERDICT=crank_nicolson_rings_in_time_inside_the_physical_range")
        return 0
    print("VERDICT=no_ringing_seen")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
