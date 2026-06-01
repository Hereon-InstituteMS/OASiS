"""Tier-2 Layer-C: NGSolve linear_elasticity MMS gate.

Cross-backend mirror of fenics + skfem
elasticity_mms_convergence. Same MMS, distinct catalog
API surface:

  * Mesh(unit_square.GenerateMesh(maxh=...))
  * VectorH1(mesh, order=k, dirichlet="...")
  * Grad / Trace / Id / sym in NGSolve's CoefficientFunction
  * BilinearForm + LinearForm with .Assemble()
  * a.mat.Inverse(fes.FreeDofs()) * f.vec  (direct solve)

Manufactured solution (matches fenics + skfem):
  u_exact = (sin(π·x)·sin(π·y),
             sin(2π·x)·sin(2π·y))
  Plane strain, E = 1.0, ν = 0.3

  Navier form: -μ·Δu - (λ + μ)·grad(div(u)) = f
"""
from __future__ import annotations

import logging
import math
import sys

logging.disable(logging.CRITICAL)

import ngsolve as ngs
from netgen.geom2d import unit_square


def run_elasticity_mms(maxh: float, order: int) -> float:
    E = 1.0
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes = ngs.VectorH1(mesh, order=order,
                        dirichlet="bottom|right|top|left")
    u, v = fes.TnT()

    x = ngs.x
    y = ngs.y

    # Manufactured solution
    u_ex = ngs.CoefficientFunction((
        ngs.sin(math.pi * x) * ngs.sin(math.pi * y),
        ngs.sin(2.0 * math.pi * x)
            * ngs.sin(2.0 * math.pi * y),
    ))

    # f = -μ·Δu - (λ+μ)·grad(div u)
    # Δu component-wise:
    #   Δu1 = -2π²·sin(πx)·sin(πy)
    #   Δu2 = -8π²·sin(2πx)·sin(2πy)
    lap_u = ngs.CoefficientFunction((
        -2.0 * math.pi ** 2
            * ngs.sin(math.pi * x)
            * ngs.sin(math.pi * y),
        -8.0 * math.pi ** 2
            * ngs.sin(2.0 * math.pi * x)
            * ngs.sin(2.0 * math.pi * y),
    ))
    # div(u) = π·cos(πx)·sin(πy)
    #        + 2π·sin(2πx)·cos(2πy)
    # grad(div u):
    #   ∂x: -π²·sin(πx)·sin(πy)
    #         + 4π²·cos(2πx)·cos(2πy)
    #   ∂y:  π²·cos(πx)·cos(πy)
    #         - 4π²·sin(2πx)·sin(2πy)
    gdiv = ngs.CoefficientFunction((
        -math.pi ** 2
            * ngs.sin(math.pi * x)
            * ngs.sin(math.pi * y)
        + 4.0 * math.pi ** 2
            * ngs.cos(2.0 * math.pi * x)
            * ngs.cos(2.0 * math.pi * y),
        math.pi ** 2
            * ngs.cos(math.pi * x)
            * ngs.cos(math.pi * y)
        - 4.0 * math.pi ** 2
            * ngs.sin(2.0 * math.pi * x)
            * ngs.sin(2.0 * math.pi * y),
    ))
    f = -mu * lap_u - (lam + mu) * gdiv

    def eps(w):
        return 0.5 * (ngs.Grad(w) + ngs.Grad(w).trans)

    def sigma(w):
        return (2.0 * mu * eps(w)
                + lam * ngs.Trace(eps(w))
                  * ngs.Id(2))

    a = ngs.BilinearForm(
        ngs.InnerProduct(sigma(u), eps(v))
        * ngs.dx).Assemble()
    ll = ngs.LinearForm(
        ngs.InnerProduct(f, v) * ngs.dx).Assemble()

    gfu = ngs.GridFunction(fes)
    # Set Dirichlet boundary values to u_exact
    gfu.Set(u_ex, definedon=mesh.Boundaries(
        "bottom|right|top|left"))

    r = ll.vec.CreateVector()
    r.data = ll.vec - a.mat * gfu.vec
    gfu.vec.data += (a.mat.Inverse(fes.FreeDofs())
                     * r)

    err = ngs.Integrate(
        ngs.InnerProduct(gfu - u_ex, gfu - u_ex), mesh)
    return float(math.sqrt(err))


def main() -> int:
    print(f"ngsolve_version={ngs.__version__}")
    results: dict[int, float] = {}
    expected_floor = {1: 1.5e-2, 2: 5e-4}
    for k, tol in expected_floor.items():
        err = run_elasticity_mms(maxh=1.0 / 16, order=k)
        results[k] = err
        print(f"order{k}_maxh16_l2err={err:.6e}"
              f"_tol={tol:.0e}")

    err_h08 = run_elasticity_mms(maxh=1.0 / 8, order=1)
    err_h16 = results[1]
    eoc = (math.log(err_h08 / err_h16) / math.log(2.0)
           if err_h16 > 0 else float("nan"))
    print(f"P1_eoc_maxh8_to_maxh16="
          f"{eoc:.3f}_expected=2.0")

    fail_reasons = []
    for k, err in results.items():
        if err > expected_floor[k]:
            fail_reasons.append(
                f"order={k} L2err {err:.3e} > "
                f"{expected_floor[k]:.0e}")
    if not (1.5 <= eoc <= 2.5):
        fail_reasons.append(
            f"P1 EOC {eoc:.3f} outside [1.5, 2.5]")

    if not fail_reasons:
        return 0
    for r in fail_reasons:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
