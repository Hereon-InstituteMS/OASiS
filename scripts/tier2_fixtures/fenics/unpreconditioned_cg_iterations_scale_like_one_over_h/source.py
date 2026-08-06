"""Tier-2 for fenics matrix_free_poisson#7: unpreconditioned CG iteration count
scales like O(1/h) -- halving the mesh size roughly DOUBLES the iteration count at
fixed polynomial degree, and raising the degree at fixed mesh also multiplies it.
Both hold for triangles and quadrilaterals. The run does not fail, it just gets
slow, and a fixed max_iter that was comfortable on the coarse mesh starts tripping
the non-convergence exit on the fine one.

Wrong variant: a max_iter budget taken from the coarse mesh and kept fixed while
the mesh is refined. Right variant: a budget that scales like 1/h (or a
preconditioner; 'gamg' and 'hypre' are not available matrix-free because they need
the assembled entries).

Observed on dolfinx 0.10.0 at rtol 1e-8, unit square, 8/16/32 cells per side:
triangles P1 22, 46, 95; triangles P2 51, 103, 204; quadrilaterals P1 15, 32, 63;
quadrilaterals P2 44, 87, 172. Every per-halving ratio lies between 1.7 and 2.5,
and at every mesh the P2 count is about 2.3 to 2.9 times the P1 count. With the
budget frozen at 1.2x the coarse count, the 32x32 run exits its for-loop with
converged=False.

Mutation control: T2_MUTATE=1 scales the budget with 1/h (four times the coarse
budget for a four-times finer mesh); the fine run then converges and the
fixed-budget token goes False.
"""
from __future__ import annotations

import math
import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dolfinx import fem, la, mesh  # noqa: E402

DTYPE = dolfinx.default_scalar_type
SIZES = (8, 16, 32)
RTOL = 1e-8


def cg_count(n: int, degree: int, cell, max_iter: int = 4000):
    comm = MPI.COMM_WORLD
    msh = mesh.create_unit_square(comm, n, n, cell_type=cell)
    V = fem.functionspace(msh, ("Lagrange", degree))
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facets = mesh.exterior_facet_indices(msh.topology)
    bdofs = fem.locate_dofs_topological(V, tdim - 1, facets)
    uD = fem.Function(V, dtype=DTYPE)
    uD.interpolate(lambda x: 0.5 * x[0])
    bc = fem.dirichletbc(uD, bdofs)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    f = 10.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L_fem = fem.form(ufl.inner(f, v) * ufl.dx, dtype=DTYPE)
    ui = fem.Function(V, dtype=DTYPE)
    M_fem = fem.form(ufl.action(a, ui), dtype=DTYPE)

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

    uh = fem.Function(V, dtype=DTYPE)
    yv = la.vector(b.index_map, 1, DTYPE)
    action_A(uh.x, yv)
    r = b.array - yv.array
    p = la.vector(b.index_map, 1, DTYPE)
    p.array[:] = r
    rn0 = rn = gdot(r, r)
    its, conv = max_iter, False
    for k in range(max_iter):
        action_A(p, yv)
        alpha = rn / gdot(p.array, yv.array)
        uh.x.array[:] += alpha * p.array
        r -= alpha * yv.array
        rn, rn_old = gdot(r, r), rn
        if rn / rn0 < RTOL ** 2:
            its, conv = k + 1, True
            break
        p.array[:] = (rn / rn_old) * p.array + r
    return its, conv


def main() -> int:
    counts: dict[tuple[str, int], list[int]] = {}
    for cell in (mesh.CellType.triangle, mesh.CellType.quadrilateral):
        for degree in (1, 2):
            row = [cg_count(n, degree, cell)[0] for n in SIZES]
            counts[(cell.name, degree)] = row
            print(f"cg_iterations cell={cell.name} degree={degree} "
                  f"n={list(SIZES)} its={row}")

    ratios = []
    for key, row in counts.items():
        rr = [round(b / a, 2) for a, b in zip(row, row[1:])]
        ratios += rr
        print(f"per_halving_ratio cell={key[0]} degree={key[1]} ratios={rr}")
    doubling = all(1.7 <= q <= 2.5 for q in ratios)

    deg_factor = []
    for cellname in ("triangle", "quadrilateral"):
        r1, r2 = counts[(cellname, 1)], counts[(cellname, 2)]
        f = [round(b / a, 2) for a, b in zip(r1, r2)]
        deg_factor += f
        print(f"degree2_over_degree1 cell={cellname} factors={f}")
    degree_costs_more = all(q > 1.5 for q in deg_factor)

    coarse = counts[("triangle", 1)][0]
    budget = math.ceil(1.2 * coarse)
    scale = SIZES[-1] // SIZES[0]
    used = budget * scale if MUTATE else budget
    its_fine, conv_fine = cg_count(SIZES[-1], 1, mesh.CellType.triangle,
                                   max_iter=used)
    print(f"coarse_count={coarse} fixed_budget={budget} budget_used={used} "
          f"fine_mesh_converged={conv_fine} fine_iterations={its_fine}")
    if MUTATE:
        print("mutation=budget_scaled_like_one_over_h")

    print(f"iteration_count_doubles_when_h_halves={doubling}")
    print(f"raising_the_degree_multiplies_the_count={degree_costs_more}")
    print(f"fine_mesh_hit_the_coarse_max_iter={not conv_fine}")
    print(f"nothing_raised_it_just_ran_out_of_iterations=True")
    if doubling and degree_costs_more and not conv_fine:
        print("VERDICT=unpreconditioned_cg_count_scales_like_one_over_h")
        return 0
    print("VERDICT=cg_count_was_mesh_independent")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
