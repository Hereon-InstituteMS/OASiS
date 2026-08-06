"""Tier-2 for fenics convection_diffusion#1: the SUPG tau must be computed PER
CELL from ufl.CellDiameter inside the dolfinx fem.form. A single global
fem.Constant tau, taken from the coarse mesh, does not shrink as h -> 0 and
leaves a fixed streamline-diffusion floor.

Wrong variant: tau = fem.Constant(msh, tau(h_coarse)) reused on every mesh.
Right variant: tau built from ufl.CellDiameter so it tracks the local cell.

MMS on the unit square, u = sin(pi x) sin(pi y), b = (1, 1), kappa = 0.01, P1,
N = 8 / 16 / 32. Observed on dolfinx 0.10.0: CellDiameter tau gives L2 errors
9.7546e-03 -> 2.5916e-03 -> 8.7443e-04 (rates 1.91, 1.57); the frozen tau gives
9.7546e-03 -> 4.1174e-03 -> 3.9380e-03, i.e. rate 0.06 on the last refinement —
the error stops moving, exactly the claim's correction that a constant tau does
not halve the rate, it STALLS convergence.

Mutation control: T2_MUTATE=1 makes the primary tau the CellDiameter one, so
the primary error no longer stalls and the fixture loses its own expectation.
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

from dolfinx.fem.petsc import LinearProblem  # noqa: E402

KAPPA = 0.01
MESHES = (8, 16, 32)


def frozen_tau(n: int) -> float:
    """tau evaluated once on the n x n mesh, then never updated."""
    bnorm = np.sqrt(2.0)
    h = np.sqrt(2.0) / n
    pe = bnorm * h / (2.0 * KAPPA)
    return float(h / (2.0 * bnorm) * (1.0 / np.tanh(pe) - 1.0 / pe))


def solve(n: int, tau_const: float | None) -> float:
    """L2 error of the SUPG solution; tau_const=None means per-cell tau."""
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(msh)
    b = dolfinx.fem.Constant(msh, np.array([1.0, 1.0]))
    kappa = dolfinx.fem.Constant(msh, KAPPA)
    u_exact = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    f = -kappa * ufl.div(ufl.grad(u_exact)) + ufl.dot(b, ufl.grad(u_exact))

    a = (kappa * ufl.inner(ufl.grad(u), ufl.grad(v))
         + ufl.dot(b, ufl.grad(u)) * v) * ufl.dx
    L = f * v * ufl.dx
    bnorm = ufl.sqrt(ufl.dot(b, b))
    if tau_const is None:
        h = ufl.CellDiameter(msh)
        pe = bnorm * h / (2.0 * kappa)
        tau = h / (2.0 * bnorm) * (1.0 / ufl.tanh(pe) - 1.0 / pe)
    else:
        tau = dolfinx.fem.Constant(msh, float(tau_const))
    res = -kappa * ufl.div(ufl.grad(u)) + ufl.dot(b, ufl.grad(u))
    a += tau * res * ufl.dot(b, ufl.grad(v)) * ufl.dx
    L += tau * f * ufl.dot(b, ufl.grad(v)) * ufl.dx

    bfacets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(msh, 0.0),
        dolfinx.fem.locate_dofs_topological(V, fdim, bfacets), V)
    problem = LinearProblem(
        a, L, bcs=[bc], petsc_options_prefix="t2_cd1_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    err = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form((uh - u_exact) ** 2 * ufl.dx))
    return float(np.sqrt(abs(err)))


def rates(errs):
    return [float(np.log2(errs[i] / errs[i + 1])) for i in range(len(errs) - 1)]


def main() -> int:
    tau0 = frozen_tau(MESHES[0])
    print(f"frozen_tau_from_coarse_mesh={tau0:.6f}")
    primary_tau = None if MUTATE else tau0
    primary = [solve(n, primary_tau) for n in MESHES]
    percell = [solve(n, None) for n in MESHES]
    r_primary, r_cell = rates(primary), rates(percell)
    for i, n in enumerate(MESHES):
        print(f"N={n} primary_l2={primary[i]:.4e} percell_l2={percell[i]:.4e}")
    print("primary_rates=" + " ".join(f"{r:.2f}" for r in r_primary))
    print("percell_rates=" + " ".join(f"{r:.2f}" for r in r_cell))

    cell_converges = all(r > 1.4 for r in r_cell)
    primary_stalls = r_primary[-1] < 0.3
    ratio = primary[-2] / primary[-1]
    print(f"primary_error_ratio_last_refinement={ratio:.3f}")
    print(f"percell_tau_keeps_converging_above_rate_1p4={cell_converges}")
    print(f"primary_tau_rate_collapses_below_0p3={primary_stalls}")
    print(f"primary_error_barely_moves_ratio_below_1p3={ratio < 1.3}")
    if cell_converges and primary_stalls and ratio < 1.3:
        print("VERDICT=global_constant_tau_stalls_supg_convergence")
        return 0
    print("VERDICT=constant_tau_converged_like_the_percell_one")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
