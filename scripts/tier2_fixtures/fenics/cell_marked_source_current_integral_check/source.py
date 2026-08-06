"""Tier-2 for fenics magnetostatics#7: the coil, the iron and the air must be
distinguished with CELL DATA. Setting Jz over the whole domain, or mu_r = 1
everywhere, produces a solve that converges and a field that is wrong by orders
of magnitude in the region that matters. The cheapest possible guard is
int(Jz)dx: it must come out close to the current you intended.

The fixture builds the intended coil (disc of radius 0.2 m carrying
Jz = 1e6 A/m^2, analytic total current pi*R^2*J) on a 32x32 mesh, once as a DG0
marker from cell midpoints and once smeared over the whole domain with
mu_r = 1 everywhere, and compares both the source integral and the resulting
max Az against the cell-marked reference.

Mutation control: T2_MUTATE=1 uses the cell-marked coil and the iron map, i.e.
the correct model; the integral then matches the analytic current and the field
matches the reference.
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

MUTATE = os.environ.get("T2_MUTATE") == "1"

MU0 = 4.0e-7 * np.pi
J0 = 1.0e6
R_COIL = 0.2
N = 32
ANALYTIC_CURRENT = np.pi * R_COIL ** 2 * J0


def run(cell_data: bool) -> tuple[float, float]:
    msh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([-0.5, -0.5]), np.array([0.5, 0.5])], [N, N])
    tdim = msh.topology.dim
    ncells = msh.topology.index_map(tdim).size_local
    mid = dolfinx.mesh.compute_midpoints(
        msh, tdim, np.arange(ncells, dtype=np.int32)).T
    DG0 = dolfinx.fem.functionspace(msh, ("DG", 0))
    Jz = dolfinx.fem.Function(DG0)
    nu = dolfinx.fem.Function(DG0)
    nu.x.array[:] = 1.0 / MU0
    if cell_data:
        Jz.x.array[:] = 0.0
        Jz.x.array[(mid[0] ** 2 + mid[1] ** 2) < R_COIL ** 2] = J0
        m = np.maximum(np.abs(mid[0]), np.abs(mid[1]))
        nu.x.array[(m > 0.25) & (m < 0.40)] = 1.0 / (MU0 * 1000.0)
    else:
        Jz.x.array[:] = J0

    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = nu * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = Jz * v * ufl.dx
    msh.topology.create_connectivity(tdim - 1, tdim)
    bdofs = dolfinx.fem.locate_dofs_topological(
        V, tdim - 1, dolfinx.mesh.exterior_facet_indices(msh.topology))
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), bdofs, V)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix=f"t2_ms7_{int(cell_data)}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    Az = prob.solve()
    assert prob.solver.getConvergedReason() > 0, "solve failed"
    itot = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jz * ufl.dx))
    return float(itot), float(Az.x.array.max())


def main() -> int:
    cell_data = bool(MUTATE)
    print(f"model_under_test_uses_cell_data={cell_data}")
    i_test, az_test = run(cell_data)
    i_ref, az_ref = run(True)
    print(f"analytic_total_current_A={ANALYTIC_CURRENT:.6e}")
    print(f"cell_marked_total_current_A={i_ref:.6e}")
    print(f"model_under_test_total_current_A={i_test:.6e}")
    stair = abs(i_ref - ANALYTIC_CURRENT) / ANALYTIC_CURRENT
    cur_ratio = i_test / ANALYTIC_CURRENT
    az_ratio = az_test / az_ref
    print(f"staircase_relative_error={stair:.3e}")
    print(f"current_ratio_to_analytic={cur_ratio:.4f}")
    print(f"max_Az_ratio_to_cell_marked_model={az_ratio:.4f}")
    stair_ok = stair < 1e-2
    off = cur_ratio > 5.0 or cur_ratio < 0.2
    field_off = az_ratio > 10.0 or az_ratio < 0.1
    print(f"staircase_relative_error_below_1_percent={stair_ok}")
    print(f"total_current_off_by_more_than_5x={off}")
    print(f"max_Az_off_by_more_than_10x={field_off}")
    if stair_ok and off and field_off:
        print("VERDICT=missing_cell_data_is_caught_by_the_current_integral")
        return 0
    print("VERDICT=model_agrees_with_the_intended_coil")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
