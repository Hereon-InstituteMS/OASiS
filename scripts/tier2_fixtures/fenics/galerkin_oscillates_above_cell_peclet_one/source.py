"""Tier-2 for fenics convection_diffusion#0: without stabilisation the Galerkin
method oscillates whenever the CELL Peclet number Pe_h = |b|*h/(2*kappa)
exceeds ~1, and the correction in the claim — the oscillation DOES damp under
refinement, it just needs a mesh that resolves the layer, which is the real
(cost) argument for stabilising.

Wrong variant: plain Galerkin P1 on the unit square, b=(1,0), kappa=1e-3, u=0
at x=0 and u=1 at x=1 (an exponential outflow layer). Right variant: the same
form plus the SUPG streamline term with the per-cell tau.

Observed on dolfinx 0.10.0: the Galerkin undershoot is -6.2798 / -3.0025 /
-1.7205 at N = 8 / 16 / 32 (Pe_h = 62.5 / 31.25 / 15.625) — large, and
shrinking monotonically with refinement — while SUPG on the same meshes stays
at -0.0416 / -0.0377 / -0.0312.

Mutation control: T2_MUTATE=1 runs SUPG as the primary scheme, so the primary
undershoot never exceeds one and the fixture loses its own expectation.
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

KAPPA = 1.0e-3
MESHES = (8, 16, 32)


def solve(n: int, stabilised: bool) -> float:
    """Return min(u_h) of the outflow-layer problem on an n x n mesh."""
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    b = dolfinx.fem.Constant(msh, np.array([1.0, 0.0]))
    kappa = dolfinx.fem.Constant(msh, KAPPA)
    a = (kappa * ufl.inner(ufl.grad(u), ufl.grad(v))
         + ufl.dot(b, ufl.grad(u)) * v) * ufl.dx
    L = dolfinx.fem.Constant(msh, 0.0) * v * ufl.dx
    if stabilised:
        h = ufl.CellDiameter(msh)
        bnorm = ufl.sqrt(ufl.dot(b, b))
        pe = bnorm * h / (2.0 * kappa)
        tau = h / (2.0 * bnorm) * (1.0 / ufl.tanh(pe) - 1.0 / pe)
        res = -kappa * ufl.div(ufl.grad(u)) + ufl.dot(b, ufl.grad(u))
        a += tau * res * ufl.dot(b, ufl.grad(v)) * ufl.dx
    left = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0))
    right = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 1.0))
    bcs = [
        dolfinx.fem.dirichletbc(
            dolfinx.fem.Constant(msh, 0.0),
            dolfinx.fem.locate_dofs_topological(V, fdim, left), V),
        dolfinx.fem.dirichletbc(
            dolfinx.fem.Constant(msh, 1.0),
            dolfinx.fem.locate_dofs_topological(V, fdim, right), V),
    ]
    problem = LinearProblem(
        a, L, bcs=bcs, petsc_options_prefix="t2_cd0_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()
    return float(np.min(uh.x.array))


def main() -> int:
    primary_stabilised = MUTATE
    primary, supg, peclet = [], [], []
    for n in MESHES:
        primary.append(solve(n, primary_stabilised))
        supg.append(solve(n, True))
        peclet.append(1.0 * (1.0 / n) / (2.0 * KAPPA))
    for n, pe, p, s in zip(MESHES, peclet, primary, supg):
        print(f"N={n} cell_peclet={pe:.4f} primary_min={p:+.4f} supg_min={s:+.4f}")

    pe_above_one = all(pe > 1.0 for pe in peclet)
    coarse_undershoot = primary[0] < -1.0
    damps = all(abs(primary[i + 1]) < abs(primary[i])
                for i in range(len(primary) - 1))
    supg_small = all(s > -0.1 for s in supg)

    print(f"cell_peclet_above_one_on_every_mesh={pe_above_one}")
    print(f"primary_undershoot_below_minus_one_on_coarse_mesh={coarse_undershoot}")
    print(f"primary_undershoot_damps_monotonically_under_refinement={damps}")
    print(f"supg_undershoot_stays_within_ten_percent={supg_small}")
    if pe_above_one and coarse_undershoot and damps and supg_small:
        print("VERDICT=galerkin_oscillates_above_cell_peclet_one_and_damps_only_by_resolving_the_layer")
        return 0
    print("VERDICT=no_cell_peclet_oscillation_observed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
