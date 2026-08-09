"""Tier-2 for fenics convection_diffusion#2: "DG methods are a cleaner
alternative for pure advection (no diffusion); for vanishing kappa the SUPG
solution oscillates between elements, an upwind DG element produces a smooth
Function with no parameter tuning."

Half of that is true and half is not, and the fixture measures both halves on
ONE mesh and ONE problem: unit square, b = (1, 0.3), kappa = 1e-9, inflow data
u = 1 for y <= 0.5 and u = 0 above on x = 0 (plus u = 1 on y = 0), 16x16
triangles. The exact solution is that step transported along the
characteristics, so every value must lie in [0, 1] and anything outside it is
oscillation.

Wrong variant: continuous P1 with the SUPG streamline term. Right variant: the
LOWEST-ORDER upwind DG space.

Observed on dolfinx 0.10.0: CG + SUPG runs to min -0.05761 / max +1.11074, so
it does oscillate. But upwind DG at degree 1 is WORSE, not smooth:
min -0.10328 / max +1.14700. Only degree-0 upwind DG is oscillation free
(min 0.0, max 1.0). The claim's remedy therefore holds only at lowest order;
at degree >= 1 upwind DG needs a limiter exactly like every other scheme.

Mutation control: T2_MUTATE=1 makes the degree-0 upwind DG space the primary
scheme, the primary range violation disappears and the fixture loses its own
expectation.
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

N = 16
BVEC = (1.0, 0.3)
KAPPA = 1.0e-9
OPTS = {"ksp_type": "preonly", "pc_type": "lu"}


def inflow_values(x):
    return np.where(x[1] <= 0.5, 1.0, 0.0)


def new_mesh():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    return msh, fdim


def cg_supg():
    msh, fdim = new_mesh()
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    b = dolfinx.fem.Constant(msh, np.array(BVEC))
    kappa = dolfinx.fem.Constant(msh, KAPPA)
    h = ufl.CellDiameter(msh)
    bnorm = ufl.sqrt(ufl.dot(b, b))
    pe = bnorm * h / (2.0 * kappa)
    tau = h / (2.0 * bnorm) * (1.0 / ufl.tanh(pe) - 1.0 / pe)
    a = (kappa * ufl.inner(ufl.grad(u), ufl.grad(v))
         + ufl.dot(b, ufl.grad(u)) * v) * ufl.dx
    a += tau * (-kappa * ufl.div(ufl.grad(u)) + ufl.dot(b, ufl.grad(u))) \
        * ufl.dot(b, ufl.grad(v)) * ufl.dx
    L = dolfinx.fem.Constant(msh, 0.0) * v * ufl.dx
    g = dolfinx.fem.Function(V)
    g.interpolate(inflow_values)
    facets = dolfinx.mesh.locate_entities_boundary(
        msh, fdim, lambda x: np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0))
    bc = dolfinx.fem.dirichletbc(
        g, dolfinx.fem.locate_dofs_topological(V, fdim, facets))
    return LinearProblem(a, L, bcs=[bc], petsc_options_prefix="t2_cd2cg_",
                         petsc_options=OPTS).solve()


def dg_upwind(degree: int):
    msh, _ = new_mesh()
    V = dolfinx.fem.functionspace(msh, ("DG", degree))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    b = dolfinx.fem.Constant(msh, np.array(BVEC))
    n = ufl.FacetNormal(msh)
    bn = ufl.dot(b, n)
    upw = (bn + abs(bn)) / 2.0        # outflow part, zero on inflow facets
    inflow = (bn - abs(bn)) / 2.0     # inflow part, zero on outflow facets
    g = dolfinx.fem.Function(V)
    g.interpolate(inflow_values)
    a = (-ufl.dot(ufl.grad(v), b * u) * ufl.dx
         + ufl.jump(v) * (upw('+') * u('+') - upw('-') * u('-')) * ufl.dS
         + v * upw * u * ufl.ds)
    L = -v * inflow * g * ufl.ds
    return LinearProblem(a, L, bcs=[], petsc_options_prefix="t2_cd2dg_",
                         petsc_options=OPTS).solve()


def rng(uh):
    a = uh.x.array
    return float(np.min(a)), float(np.max(a))


def main() -> int:
    dg0 = rng(dg_upwind(0))
    dg1 = rng(dg_upwind(1))
    supg = rng(cg_supg())
    primary = dg0 if MUTATE else supg
    print(f"cg_supg_range=[{supg[0]:+.5f}, {supg[1]:+.5f}]")
    print(f"upwind_dg1_range=[{dg1[0]:+.5f}, {dg1[1]:+.5f}]")
    print(f"upwind_dg0_range=[{dg0[0]:+.5f}, {dg0[1]:+.5f}]")
    print(f"primary_range=[{primary[0]:+.5f}, {primary[1]:+.5f}]")

    tol = 1.0e-6
    primary_bad = primary[0] < -tol or primary[1] > 1.0 + tol
    dg1_bad = dg1[0] < -tol or dg1[1] > 1.0 + tol
    dg1_worse = (dg1[1] - 1.0) > (supg[1] - 1.0) and -dg1[0] > -supg[0]
    dg0_clean = dg0[0] >= -tol and dg0[1] <= 1.0 + tol

    print(f"primary_violates_the_zero_one_range={primary_bad}")
    print(f"upwind_dg_degree1_also_violates_the_range={dg1_bad}")
    print(f"upwind_dg_degree1_oscillates_more_than_supg={dg1_worse}")
    print(f"upwind_dg_degree0_is_range_preserving={dg0_clean}")
    if primary_bad and dg1_bad and dg1_worse and dg0_clean:
        print("VERDICT=upwind_dg_is_oscillation_free_only_at_lowest_order")
        return 0
    print("VERDICT=upwind_dg_was_smooth_at_every_degree")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
