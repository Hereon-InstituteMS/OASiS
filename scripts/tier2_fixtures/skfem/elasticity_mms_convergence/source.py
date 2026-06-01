"""Tier-2 Layer-C: skfem linear_elasticity MMS gate.

Cross-backend mirror of fenics elasticity_mms_convergence.
Same MMS, same boundary conditions, different catalog API:

  * MeshTri (Cartesian unit-square, .refined)
  * ElementVector(ElementTriP1())  (vs basix.ufl.element
    shape=(gdim,) in fenics)
  * BilinearForm with explicit Lamé stress assembly
  * LinearForm with f_x and f_y separated (vs UFL
    auto-derive in fenics)
  * skfem.condense + skfem.solve

Manufactured solution (matches fenics elasticity gate):
  u_exact = (sin(π·x)·sin(π·y),
             sin(2π·x)·sin(2π·y))
  Plane strain, E = 1.0, ν = 0.3
  σ = 2·μ·sym(grad(u_exact)) + λ·div(u_exact)·I
  f = -div(σ)   (computed analytically below)

Expected at refine=5 (h ≈ 1/32):
  P1 L2 ≲ 1.3e-2 (similar to fenics 8.99e-3 plus
                   margin for skfem's vertex-only
                   quadrature on the error functional)
  P1 EOC refine=4 → refine=5 ∈ [1.7, 2.3]
"""
from __future__ import annotations

import math
import sys

import numpy as np

import skfem
from skfem import (
    MeshTri, Basis, BilinearForm, LinearForm, Functional,
    ElementVector, ElementTriP1, condense, solve)
from skfem.helpers import sym_grad, trace, eye, ddot


def run_elasticity_mms(refine: int) -> tuple[float, int]:
    E = 1.0
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    m = MeshTri().refined(refine)
    elem = ElementVector(ElementTriP1())
    basis = Basis(m, elem)

    def eps(w):
        return sym_grad(w)

    @BilinearForm
    def stiffness(u, v, w):
        return (2.0 * mu * ddot(eps(u), eps(v))
                + lam * trace(eps(u)) * trace(eps(v)))

    # Source f = -div(σ(u_exact)) computed analytically.
    # u_exact = (sin(πx)·sin(πy), sin(2πx)·sin(2πy))
    # ε_xx = π·cos(πx)·sin(πy)
    # ε_yy = 2π·sin(2πx)·cos(2πy)
    # ε_xy = ½·(π·sin(πx)·cos(πy) + 2π·cos(2πx)·sin(2πy))
    # σ = 2μ·ε + λ·(ε_xx+ε_yy)·I
    # f = -div(σ).  Symbolic = numpy:
    @LinearForm
    def load(v, w):
        x, y = w.x[0], w.x[1]
        px = np.pi * x
        py = np.pi * y
        p2x = 2.0 * np.pi * x
        p2y = 2.0 * np.pi * y

        # u = (u1, u2)
        # u1 = sin(πx)·sin(πy)
        # u2 = sin(2πx)·sin(2πy)
        # Δu1 = -2π²·sin(πx)·sin(πy)
        # Δu2 = -8π²·sin(2πx)·sin(2πy)
        lap_u1 = -2.0 * np.pi ** 2 * np.sin(px) * np.sin(py)
        lap_u2 = (-8.0 * np.pi ** 2
                  * np.sin(p2x) * np.sin(p2y))

        # div(u) = ∂u1/∂x + ∂u2/∂y
        #        = π·cos(πx)·sin(πy)
        #        + 2π·sin(2πx)·cos(2πy)
        # grad(div(u)) = (
        #   ∂/∂x[π·cos(πx)·sin(πy)
        #        + 2π·sin(2πx)·cos(2πy)],
        #   ∂/∂y[π·cos(πx)·sin(πy)
        #        + 2π·sin(2πx)·cos(2πy)])
        # = (-π²·sin(πx)·sin(πy)
        #     + 4π²·cos(2πx)·cos(2πy),
        #    π²·cos(πx)·cos(πy)
        #     - 4π²·sin(2πx)·sin(2πy))
        gdiv_x = (-(np.pi ** 2) * np.sin(px) * np.sin(py)
                  + 4.0 * (np.pi ** 2)
                    * np.cos(p2x) * np.cos(p2y))
        gdiv_y = ((np.pi ** 2) * np.cos(px) * np.cos(py)
                  - 4.0 * (np.pi ** 2)
                    * np.sin(p2x) * np.sin(p2y))

        # Navier: -div(σ) = -μ·Δu - (λ + μ)·grad(div u)
        # so f = -μ·Δu - (λ + μ)·grad(div u)
        f1 = -mu * lap_u1 - (lam + mu) * gdiv_x
        f2 = -mu * lap_u2 - (lam + mu) * gdiv_y
        return f1 * v.value[0] + f2 * v.value[1]

    K = stiffness.assemble(basis)
    F = load.assemble(basis)

    # Boundary: u = u_exact on ∂Ω.
    # Set as the interpolation of u_exact onto the boundary
    # dofs.
    D = basis.get_dofs()
    u = np.zeros(basis.N)
    # ElementVector(ElementTriP1()) interleaves dofs:
    #   dof[2i]   = x-component at node i
    #   dof[2i+1] = y-component at node i
    # doflocs has shape (gdim, basis.N) where consecutive
    # pairs share the same physical (x, y).
    doflocs = basis.doflocs
    bx = doflocs[0]
    by = doflocs[1]
    # At every even index, the dof is u_x; at every odd
    # index, the dof is u_y. Evaluate the manufactured
    # values at the matching physical location.
    u[0::2] = (np.sin(np.pi * bx[0::2])
               * np.sin(np.pi * by[0::2]))
    u[1::2] = (np.sin(2.0 * np.pi * bx[1::2])
               * np.sin(2.0 * np.pi * by[1::2]))

    # condense + solve via the catalog pattern:
    #   solve(*condense(K, F, x=u, D=D))
    # condense with x= returns a tuple whose layout depends
    # on expand/I args; the *unpack form re-applies the BC
    # values automatically.
    u = solve(*condense(K, F, x=u, D=D))

    # L2 error using Functional with vector quadrature
    @Functional
    def l2err(w):
        x, y = w.x[0], w.x[1]
        u_ex_x = np.sin(np.pi * x) * np.sin(np.pi * y)
        u_ex_y = (np.sin(2.0 * np.pi * x)
                  * np.sin(2.0 * np.pi * y))
        uh_x = w["uh"].value[0]
        uh_y = w["uh"].value[1]
        return (uh_x - u_ex_x) ** 2 + (uh_y - u_ex_y) ** 2

    err_sq = l2err.assemble(basis, uh=basis.interpolate(u))
    return float(math.sqrt(err_sq)), basis.N


def main() -> int:
    print(f"skfem_version={skfem.__version__}")
    err_h32, dofs_h32 = run_elasticity_mms(refine=5)
    err_h16, _ = run_elasticity_mms(refine=4)
    eoc = (math.log(err_h16 / err_h32) / math.log(2.0)
           if err_h32 > 0 else float("nan"))
    print(f"P1_h32_l2err={err_h32:.6e}_tol=1.3e-02")
    print(f"P1_h16_l2err={err_h16:.6e}")
    print(f"P1_eoc_h16_to_h32={eoc:.3f}_expected=2.0")
    print(f"dofs_h32={dofs_h32}")

    fail_reasons = []
    if err_h32 > 1.3e-2:
        fail_reasons.append(
            f"P1 h=1/32 L2 err {err_h32:.3e} > 1.3e-2")
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
