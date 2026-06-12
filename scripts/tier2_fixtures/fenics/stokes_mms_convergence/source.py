"""Tier-2 Layer-C: fenics Stokes MMS gate (Taylor-Hood).

Third physics anchored under Layer C. Tests the
catalog's MIXED (vector velocity + scalar pressure)
API:

  * basix.ufl.mixed_element([P2_vec, P1_scalar])
  * fem.functionspace returns a mixed space; .sub(0)/
    .sub(1).collapse() partition into V (velocity) and
    Q (pressure)
  * Indefinite saddle-point system → MUMPS direct
    (NOT CG)
  * Pressure pinning required (Dirichlet u on ∂Ω
    leaves p determined only up to a constant — the
    chosen MMS has zero-mean p; we use a Lagrange-
    multiplier-style mean constraint or pin a single
    DOF; this fixture pins p at a single point)

Manufactured solution (divergence-free, ν = 1):

  Stream function ψ = sin(π·x)² · sin(π·y)²
  u = curl(ψ) = (∂y ψ, -∂x ψ)
              = (π sin(π·x)² sin(2π·y),
                -π sin(2π·x) sin(π·y)²)

  div(u) = 0  (by construction)
  p = sin(π·x) · cos(π·y)   (mean ∫p dx = 0 on [0,1]²)
  f = -ν Δu + grad(p)    (UFL-symbolic)

Weak Stokes:
  ν inner(grad(u), grad(v))*dx - p*div(v)*dx
        - q*div(u)*dx = inner(f, v)*dx

Expected at h=1/32, P2/P1:
  ||u - u_exact||_L2 ≲ 5e-4   (P2 velocity, O(h^3))
  ||p - p_exact||_L2 ≲ 5e-3   (P1 pressure, O(h^2))
"""
from __future__ import annotations

import logging
import math
import sys

logging.disable(logging.CRITICAL)


def run_stokes_mms(nx: int) -> tuple[float, float]:
    from mpi4py import MPI
    import numpy as np
    import basix.ufl
    import ufl
    from dolfinx import mesh, fem, default_scalar_type
    from dolfinx.fem.petsc import LinearProblem

    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, nx, nx, mesh.CellType.triangle)
    gdim = domain.geometry.dim

    P2 = basix.ufl.element(
        "Lagrange", domain.basix_cell(), 2,
        shape=(gdim,))
    P1 = basix.ufl.element(
        "Lagrange", domain.basix_cell(), 1)
    TH = basix.ufl.mixed_element([P2, P1])
    W = fem.functionspace(domain, TH)

    x = ufl.SpatialCoordinate(domain)
    nu = fem.Constant(domain, default_scalar_type(1.0))
    u_ex = ufl.as_vector([
        ufl.pi * ufl.sin(ufl.pi * x[0]) ** 2
            * ufl.sin(2.0 * ufl.pi * x[1]),
        -ufl.pi * ufl.sin(2.0 * ufl.pi * x[0])
            * ufl.sin(ufl.pi * x[1]) ** 2,
    ])
    p_ex = (ufl.sin(ufl.pi * x[0])
            * ufl.cos(ufl.pi * x[1]))
    f = -nu * ufl.div(ufl.grad(u_ex)) + ufl.grad(p_ex)

    (u, p) = ufl.TrialFunctions(W)
    (v, q) = ufl.TestFunctions(W)
    a = (nu * ufl.inner(ufl.grad(u), ufl.grad(v))
         * ufl.dx
         - p * ufl.div(v) * ufl.dx
         - q * ufl.div(u) * ufl.dx)
    # RHS: only velocity equation has a non-zero load;
    # the continuity equation has zero RHS.
    L = (ufl.inner(f, v) * ufl.dx
         + ufl.inner(fem.Constant(domain,
                                   default_scalar_type(0.0)),
                     q) * ufl.dx)

    # Dirichlet u = u_exact on ∂Ω (full no-flow + tangential)
    W0, W0_to_W = W.sub(0).collapse()
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    bdry = mesh.exterior_facet_indices(domain.topology)
    dofs_u = fem.locate_dofs_topological(
        (W.sub(0), W0), fdim, bdry)
    u_bc_func = fem.Function(W0)

    def u_bc_callable(pts):
        xx = pts[0]
        yy = pts[1]
        ux = (np.pi
              * np.sin(np.pi * xx) ** 2
              * np.sin(2.0 * np.pi * yy))
        uy = (-np.pi
              * np.sin(2.0 * np.pi * xx)
              * np.sin(np.pi * yy) ** 2)
        out = np.zeros((gdim, xx.shape[0]))
        out[0] = ux
        out[1] = uy
        return out
    u_bc_func.interpolate(u_bc_callable)
    bc_u = fem.dirichletbc(u_bc_func, dofs_u, W.sub(0))

    # Pin pressure at a single point (origin) to fix the
    # indeterminacy. Find the pressure DOF closest to (0, 0).
    W1, W1_to_W = W.sub(1).collapse()
    dof_coords = W1.tabulate_dof_coordinates()
    dists = np.linalg.norm(dof_coords[:, :2], axis=1)
    pin_local = int(np.argmin(dists))
    pin_global = W1_to_W[pin_local]
    # Pressure at origin in p_ex: sin(0)·cos(0) = 0.
    pin_val = 0.0
    # Pin a single pressure DOF via parent-space index.
    bc_p = fem.dirichletbc(
        default_scalar_type(pin_val),
        np.array([pin_global], dtype=np.int32),
        W.sub(1))

    problem = LinearProblem(
        a, L, bcs=[bc_u, bc_p],
        petsc_options_prefix=f"stokes_mms_{nx}_",
        petsc_options={"ksp_type": "preonly",
                        "pc_type": "lu",
                        "pc_factor_mat_solver_type":
                            "mumps"})
    sol = problem.solve()
    u_h = sol.sub(0).collapse()
    p_h = sol.sub(1).collapse()

    # L2 errors
    err_u_form = fem.form(
        ufl.inner(u_h - u_ex, u_h - u_ex)
        * ufl.dx(metadata={"quadrature_degree": 4}))
    err_p_form = fem.form(
        (p_h - p_ex) ** 2
        * ufl.dx(metadata={"quadrature_degree": 4}))
    err_u = math.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(err_u_form)))
    err_p = math.sqrt(domain.comm.allreduce(
        fem.assemble_scalar(err_p_form)))
    return err_u, err_p


def main() -> int:
    err_u32, err_p32 = run_stokes_mms(nx=32)
    err_u16, err_p16 = run_stokes_mms(nx=16)
    eoc_u = (math.log(err_u16 / err_u32) / math.log(2.0)
             if err_u32 > 0 else float("nan"))
    eoc_p = (math.log(err_p16 / err_p32) / math.log(2.0)
             if err_p32 > 0 else float("nan"))
    print(f"P2_u_h32_l2err={err_u32:.6e}_tol=5e-04")
    print(f"P1_p_h32_l2err={err_p32:.6e}_tol=5e-03")
    print(f"P2_u_eoc_h16_to_h32={eoc_u:.3f}_expected=3.0")
    print(f"P1_p_eoc_h16_to_h32={eoc_p:.3f}_expected=2.0")

    fail_reasons = []
    if err_u32 > 5e-4:
        fail_reasons.append(
            f"P2 u L2err {err_u32:.3e} > 5e-4")
    if err_p32 > 5e-3:
        fail_reasons.append(
            f"P1 p L2err {err_p32:.3e} > 5e-3")
    if not (2.5 <= eoc_u <= 3.5):
        fail_reasons.append(
            f"P2 u EOC {eoc_u:.3f} outside [2.5, 3.5]")
    # P1 pressure SUPER-CONVERGES on Cartesian-symmetric
    # Stokes Taylor-Hood meshes: theoretical 2.0 but
    # measured 2.6+ is well-documented. Widen the band.
    if not (1.7 <= eoc_p <= 3.0):
        fail_reasons.append(
            f"P1 p EOC {eoc_p:.3f} outside [1.7, 3.0]")

    if not fail_reasons:
        return 0
    for r in fail_reasons:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
