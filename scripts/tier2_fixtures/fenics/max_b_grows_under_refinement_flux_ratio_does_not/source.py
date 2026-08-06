"""Tier-2 for fenics magnetostatics#1: max|B| is NOT a convergence indicator
for a problem with a permeability jump. The field is singular at the material
corner and the jump follows the mesh, so refining the mesh makes the reported
maximum GROW while the KSP reports converged every time. The mesh-independent
check that does work is the boundary flux of nu*grad(Az), which equals minus
the total current at every resolution.

Coil disc (r = 0.2 m, Jz = 1e6 A/m^2) inside a square iron ring with
mu_r = 1000, air elsewhere, Az = 0 on the outer boundary, solved at 8x8, 16x16
and 32x32 with MUMPS-LU.

Mutation control: T2_MUTATE=1 removes the permeability jump (mu_r = 1
everywhere). max|B| then converges instead of growing, which is what a
convergence indicator is supposed to do.
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


def iron_cells(mid: np.ndarray) -> np.ndarray:
    m = np.maximum(np.abs(mid[0]), np.abs(mid[1]))
    return (m > 0.25) & (m < 0.40)


def solve(n: int, mur_iron: float) -> tuple[int, float, float]:
    msh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD, [np.array([-0.5, -0.5]), np.array([0.5, 0.5])], [n, n])
    tdim = msh.topology.dim
    ncells = msh.topology.index_map(tdim).size_local
    mid = dolfinx.mesh.compute_midpoints(
        msh, tdim, np.arange(ncells, dtype=np.int32)).T

    DG0 = dolfinx.fem.functionspace(msh, ("DG", 0))
    nu = dolfinx.fem.Function(DG0)
    nu.x.array[:] = 1.0 / MU0
    nu.x.array[iron_cells(mid)] = 1.0 / (MU0 * mur_iron)
    Jz = dolfinx.fem.Function(DG0)
    Jz.x.array[:] = 0.0
    Jz.x.array[(mid[0] ** 2 + mid[1] ** 2) < R_COIL ** 2] = J0

    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = nu * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = Jz * v * ufl.dx
    msh.topology.create_connectivity(tdim - 1, tdim)
    bdofs = dolfinx.fem.locate_dofs_topological(
        V, tdim - 1, dolfinx.mesh.exterior_facet_indices(msh.topology))
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), bdofs, V)
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix=f"t2_ms1_{n}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "pc_factor_mat_solver_type": "mumps"})
    Az = prob.solve()
    reason = prob.solver.getConvergedReason()

    W = dolfinx.fem.functionspace(msh, ("DG", 0, (2,)))
    B = dolfinx.fem.Function(W)
    B.interpolate(dolfinx.fem.Expression(
        ufl.curl(Az), W.element.interpolation_points))
    max_b = float(np.sqrt((B.x.array.reshape(-1, 2) ** 2).sum(axis=1)).max())

    nrm = ufl.FacetNormal(msh)
    flux = dolfinx.fem.assemble_scalar(dolfinx.fem.form(
        nu * ufl.dot(ufl.grad(Az), nrm) * ufl.ds))
    itot = dolfinx.fem.assemble_scalar(dolfinx.fem.form(Jz * ufl.dx))
    return reason, max_b, float(-flux / itot)


def main() -> int:
    mur = 1.0 if MUTATE else 1000.0
    print(f"mu_r_iron={mur}")
    maxb, ratios, reasons = [], [], []
    for n in (8, 16, 32):
        reason, mb, ratio = solve(n, mur)
        reasons.append(reason)
        maxb.append(mb)
        ratios.append(ratio)
        print(f"level n={n} ksp_reason={reason} max_B_T={mb:.6e} "
              f"minus_flux_over_total_current={ratio:.10f}")
    grows = all(maxb[i + 1] > 1.05 * maxb[i] for i in range(len(maxb) - 1))
    flux_ok = all(abs(r - 1.0) < 1e-9 for r in ratios)
    conv = all(r > 0 for r in reasons)
    print(f"max_B_ratio_finest_over_coarsest={maxb[-1] / maxb[0]:.4f}")
    print(f"every_ksp_reported_converged={conv}")
    print(f"max_B_grows_monotonically_under_refinement={grows}")
    print(f"flux_ratio_is_one_at_every_level={flux_ok}")
    if grows and flux_ok and conv:
        print("VERDICT=max_B_tracks_the_mesh_while_the_flux_check_is_invariant")
        return 0
    print("VERDICT=max_B_is_a_usable_convergence_indicator_here")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
