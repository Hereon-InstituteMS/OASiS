"""Tier-2 Layer-C: skfem Stokes MMS — Taylor-Hood P2/P1.

Cross-backend mirror of fenics stokes_mms_convergence.
Same MMS, distinct catalog API:

  * MeshTri.refined (NOT MeshQuad — Taylor-Hood needs
    a P2-vector / P1-scalar pair; ElementVector +
    ElementTriP2 / ElementTriP1)
  * ElementVector(ElementTriP2()) for velocity
  * ElementTriP1() for pressure
  * skfem.models.general.divergence for B-block
  * scipy.sparse.bmat to build the [[K, B^T], [B, 0]]
    saddle-point matrix
  * Pin a single pressure DOF (Stokes-Dirichlet has p
    only determined up to constant)
  * scipy spsolve direct (indefinite — CG diverges)

MMS (matches fenics):
  ψ = sin(πx)²·sin(πy)²
  u = curl(ψ) = (π·sin(πx)²·sin(2πy),
                -π·sin(2πx)·sin(πy)²)
  p = sin(πx)·cos(πy)            (zero mean)
  f = -ν·Δu + grad(p)            (analytic; computed below)
  ν = 1
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy.sparse import bmat
from scipy.sparse.linalg import spsolve

import skfem
from skfem import (
    MeshTri, Basis, BilinearForm, LinearForm, Functional,
    ElementVector, ElementTriP1, ElementTriP2,
    asm, condense, solve)
from skfem.helpers import grad, ddot
from skfem.models.general import divergence


def run_stokes_mms(refine: int) -> tuple[float, float]:
    m = MeshTri().refined(refine)
    e_u = ElementVector(ElementTriP2())
    e_p = ElementTriP1()
    # Trial / test must share quadrature for the
    # mixed divergence(basis_u, basis_p) bilinear form
    # (skfem ValueError: "Quadrature mismatch" otherwise).
    basis_u = Basis(m, e_u, intorder=4)
    basis_p = Basis(m, e_p, intorder=4)
    nu = 1.0

    @BilinearForm
    def stiffness(u, v, w):
        # ν · grad(u) : grad(v)
        return nu * ddot(grad(u), grad(v))

    K = asm(stiffness, basis_u)
    # skfem catalog pattern (src/backends/skfem/generators/
    # stokes.py): K12 = -asm(divergence, ib_u, ib_p). So
    # divergence gives +∫q·div(u) and we flip sign here to
    # form the standard saddle-point block
    # [[K, -B^T], [-B, 0]] with B = ∫q·div(u).
    B = -asm(divergence, basis_u, basis_p)

    # f1, f2 source from MMS, evaluated at quadrature pts
    # via @LinearForm with vector-test-function v.
    @LinearForm
    def load(v, w):
        x, y = w.x[0], w.x[1]
        pi = np.pi
        # u_exact, p_exact, then f = -ν·Δu + grad(p)
        # Δu1 = ?, Δu2 = ?
        # u1 = π·sin(πx)²·sin(2πy)
        # ∂xx u1 = π·2π²·[cos(πx)²−sin(πx)²]·sin(2πy)
        #        = 2π³ cos(2πx) sin(2πy)
        # ∂yy u1 = π·sin(πx)²·(-4π²·sin(2πy))
        #        = -4π³ sin(πx)² sin(2πy)
        # Δu1 = 2π³ cos(2πx) sin(2πy)
        #       - 4π³ sin(πx)² sin(2πy)
        # u2 = -π·sin(2πx)·sin(πy)²
        # ∂xx u2 = -π · (-4π²·sin(2πx)) · sin(πy)²
        #        = 4π³ sin(2πx) sin(πy)²
        # ∂yy u2 = -π · sin(2πx) · 2π²·[cos(πy)²−sin(πy)²]
        #        = -2π³ sin(2πx) cos(2πy)
        # Δu2 = 4π³ sin(2πx) sin(πy)²
        #       - 2π³ sin(2πx) cos(2πy)
        lap_u1 = (2.0 * pi ** 3
                  * np.cos(2.0 * pi * x)
                  * np.sin(2.0 * pi * y)
                  - 4.0 * pi ** 3
                  * np.sin(pi * x) ** 2
                  * np.sin(2.0 * pi * y))
        lap_u2 = (4.0 * pi ** 3
                  * np.sin(2.0 * pi * x)
                  * np.sin(pi * y) ** 2
                  - 2.0 * pi ** 3
                  * np.sin(2.0 * pi * x)
                  * np.cos(2.0 * pi * y))
        # grad(p):
        # p = sin(πx)·cos(πy)
        # ∂xp =  π·cos(πx)·cos(πy)
        # ∂yp = -π·sin(πx)·sin(πy)
        gp_x = pi * np.cos(pi * x) * np.cos(pi * y)
        gp_y = -pi * np.sin(pi * x) * np.sin(pi * y)
        f1 = -nu * lap_u1 + gp_x
        f2 = -nu * lap_u2 + gp_y
        return f1 * v.value[0] + f2 * v.value[1]

    F_u = asm(load, basis_u)
    F_p = np.zeros(basis_p.N)

    # Block: [[K, B^T], [B, 0]]; F = [F_u; F_p]
    A = bmat([[K, B.T], [B, None]], format='csr')
    F = np.concatenate([F_u, F_p])

    # BCs:
    # u_exact on ∂Ω, pinning every boundary velocity DOF
    # to the manufactured value; single pressure pin at
    # the closest-to-origin pressure DOF (value 0).
    D_u = basis_u.get_dofs().flatten()
    doflocs_u = basis_u.doflocs
    bx = doflocs_u[0]
    by = doflocs_u[1]
    # x-component DOFs at 0::2, y-component at 1::2.
    u_init = np.zeros(basis_u.N)
    u_init[0::2] = (np.pi
                    * np.sin(np.pi * bx[0::2]) ** 2
                    * np.sin(2.0 * np.pi * by[0::2]))
    u_init[1::2] = (-np.pi
                    * np.sin(2.0 * np.pi * bx[1::2])
                    * np.sin(np.pi * by[1::2]) ** 2)

    # Pressure pin at the DOF closest to (0, 0).
    pdofs = basis_p.doflocs.T
    pin_idx = int(np.argmin(
        np.linalg.norm(pdofs[:, :2], axis=1)))
    # In the block system, pressure DOFs are offset by
    # basis_u.N.
    pin_global = basis_u.N + pin_idx

    # Build x_full (initial guess in the BC-constrained
    # full-space):
    x_full = np.zeros(A.shape[0])
    x_full[:basis_u.N] = u_init
    x_full[pin_global] = 0.0
    D = np.concatenate([D_u,
                        np.array([pin_global],
                                  dtype=np.int64)])
    sol_full = solve(*condense(A, F, D=D, x=x_full))
    u_h = sol_full[:basis_u.N]
    p_h = sol_full[basis_u.N:]

    @Functional
    def err_u_sq(w):
        x, y = w.x[0], w.x[1]
        pi = np.pi
        u_ex_1 = (pi * np.sin(pi * x) ** 2
                  * np.sin(2.0 * pi * y))
        u_ex_2 = (-pi * np.sin(2.0 * pi * x)
                  * np.sin(pi * y) ** 2)
        uh1 = w["u_h"].value[0]
        uh2 = w["u_h"].value[1]
        return (uh1 - u_ex_1) ** 2 + (uh2 - u_ex_2) ** 2

    @Functional
    def err_p_sq(w):
        x, y = w.x[0], w.x[1]
        p_ex = np.sin(np.pi * x) * np.cos(np.pi * y)
        ph = w["p_h"]
        return (ph - p_ex) ** 2

    err_u = math.sqrt(err_u_sq.assemble(
        basis_u, u_h=basis_u.interpolate(u_h)))
    err_p = math.sqrt(err_p_sq.assemble(
        basis_p, p_h=basis_p.interpolate(p_h)))
    return err_u, err_p


def main() -> int:
    print(f"skfem_version={skfem.__version__}")
    err_u_h32, err_p_h32 = run_stokes_mms(refine=5)
    err_u_h16, err_p_h16 = run_stokes_mms(refine=4)
    eoc_u = (math.log(err_u_h16 / err_u_h32)
             / math.log(2.0)
             if err_u_h32 > 0 else float("nan"))
    eoc_p = (math.log(err_p_h16 / err_p_h32)
             / math.log(2.0)
             if err_p_h32 > 0 else float("nan"))
    print(f"P2_u_h32_l2err={err_u_h32:.6e}_tol=5e-04")
    print(f"P1_p_h32_l2err={err_p_h32:.6e}_tol=2e-02")
    print(f"P2_u_eoc_h16_to_h32={eoc_u:.3f}_expected=3.0")
    print(f"P1_p_eoc_h16_to_h32={eoc_p:.3f}_expected=2.0")

    fail_reasons = []
    if err_u_h32 > 5e-4:
        fail_reasons.append(
            f"P2 u L2 err {err_u_h32:.3e} > 5e-4")
    if err_p_h32 > 2e-2:
        fail_reasons.append(
            f"P1 p L2 err {err_p_h32:.3e} > 2e-2")
    if not (2.5 <= eoc_u <= 3.5):
        fail_reasons.append(
            f"P2 u EOC {eoc_u:.3f} outside [2.5, 3.5]")
    if not (1.5 <= eoc_p <= 3.0):
        fail_reasons.append(
            f"P1 p EOC {eoc_p:.3f} outside [1.5, 3.0]")

    if not fail_reasons:
        return 0
    for r in fail_reasons:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
