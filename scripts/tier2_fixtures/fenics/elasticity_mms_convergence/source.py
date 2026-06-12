"""Tier-2 Layer-C: fenics linear_elasticity MMS gate.

Second physics anchored under Layer C (the first is
fenics::poisson). Tests the catalog's vector-valued API
surface:

  * basix.ufl.element('Lagrange', cell, k, shape=(gdim,))
  * fem.functionspace returns a VectorFunctionSpace-like
    object whose collapse gives a P1/P2-vector subspace
  * Lamé constitutive: sigma = 2 μ ε + λ tr(ε) I
  * fem.dirichletbc on a vector-valued function
  * LinearProblem with MUMPS

MMS (plane strain, E = 1.0, ν = 0.3):

  u_exact(x, y) = (sin(π·x) · sin(π·y),
                   sin(2π·x) · sin(2π·y))

  ε = sym(grad(u_exact))
  σ = 2 μ ε + λ tr(ε) I
  f = -div(σ)        (computed by UFL symbolically)

  Dirichlet: u = u_exact on ∂Ω (full traction-free is
  ill-posed for elasticity without a constraint; pure
  Dirichlet eliminates rigid-body modes).

Expected at h=1/32:
  P1 L2 ≲ 5e-3
  P2 L2 ≲ 1e-4
  P1 EOC h=1/16 → h=1/32 ∈ [1.7, 2.3]
"""
from __future__ import annotations

import logging
import math
import sys

logging.disable(logging.CRITICAL)


def run_elasticity_mms(nx: int, order: int) -> float:
    from mpi4py import MPI
    import numpy as np
    import basix.ufl
    import ufl
    from dolfinx import mesh, fem, default_scalar_type
    from dolfinx.fem.petsc import LinearProblem

    E = 1.0
    nu = 0.3
    mu_val = E / (2.0 * (1.0 + nu))
    lam_val = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, nx, nx, mesh.CellType.triangle)
    gdim = domain.geometry.dim
    V = fem.functionspace(
        domain,
        basix.ufl.element("Lagrange",
                           domain.basix_cell(), order,
                           shape=(gdim,)))

    x = ufl.SpatialCoordinate(domain)
    u_ex_expr = ufl.as_vector([
        ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
        ufl.sin(2.0 * ufl.pi * x[0])
            * ufl.sin(2.0 * ufl.pi * x[1]),
    ])

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return (2.0 * mu_val * eps(w)
                + lam_val * ufl.tr(eps(w))
                  * ufl.Identity(gdim))

    f_expr = -ufl.div(sigma(u_ex_expr))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.inner(f_expr, v) * ufl.dx

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    bdry = mesh.exterior_facet_indices(domain.topology)
    dofs = fem.locate_dofs_topological(V, fdim, bdry)

    # u_exact interpolated onto V → boundary Dirichlet
    u_bc = fem.Function(V)
    def _u_bc_callable(pts):
        xx = pts[0]
        yy = pts[1]
        out = np.zeros((gdim, xx.shape[0]))
        out[0] = np.sin(np.pi * xx) * np.sin(np.pi * yy)
        out[1] = (np.sin(2.0 * np.pi * xx)
                  * np.sin(2.0 * np.pi * yy))
        return out
    u_bc.interpolate(_u_bc_callable)
    bc = fem.dirichletbc(u_bc, dofs)

    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options_prefix=f"el_mms_p{order}_{nx}_",
        petsc_options={"ksp_type": "preonly",
                        "pc_type": "lu",
                        "pc_factor_mat_solver_type":
                            "mumps"})
    uh = problem.solve()

    err_form = fem.form(
        ufl.inner(uh - u_ex_expr, uh - u_ex_expr)
        * ufl.dx(metadata={
            "quadrature_degree": order + 2}))
    err_l2_sq = domain.comm.allreduce(
        fem.assemble_scalar(err_form))
    return float(math.sqrt(err_l2_sq))


def main() -> int:
    # Empirical floors: the manufactured solution's u_y
    # component oscillates with 2π·x and 2π·y so the P1
    # error is higher than for plain Poisson at the same h.
    #   P1 nx=32 measured ≈ 8.99e-3
    #   P2 nx=32 measured ≈ 5.88e-5
    floor = {1: 1.2e-2, 2: 1e-4}
    results: dict[int, float] = {}
    for k, tol in floor.items():
        err = run_elasticity_mms(nx=32, order=k)
        results[k] = err
        print(f"P{k}_nx32_l2err={err:.6e}_tol={tol:.0e}")

    err_h16 = run_elasticity_mms(nx=16, order=1)
    err_h32 = results[1]
    eoc = (math.log(err_h16 / err_h32) / math.log(2.0)
           if err_h32 > 0 else float("nan"))
    print(f"P1_eoc_h16_to_h32={eoc:.3f}_expected=2.0")

    fail_reasons = []
    for k, err in results.items():
        if err > floor[k]:
            fail_reasons.append(
                f"P{k} L2err {err:.3e} > {floor[k]:.0e}")
    if not (1.7 <= eoc <= 2.3):
        fail_reasons.append(
            f"P1 EOC {eoc:.3f} outside [1.7, 2.3]")

    if not fail_reasons:
        return 0
    for r in fail_reasons:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
