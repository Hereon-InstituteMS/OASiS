"""Tier-2 Layer-C: NGSolve Stokes Taylor-Hood MMS.

Cross-backend mirror of fenics + skfem stokes_mms_convergence.
Same manufactured solution, distinct NGSolve API:

  * Mesh(unit_square.GenerateMesh(maxh=...))
  * VectorH1(mesh, order=2, dirichlet=...) * H1(mesh, order=1)
  * Compound TnT (trial/test for both blocks)
  * BilinearForm: ν·InnerProduct(Grad(u), Grad(v)) +
                  div(u)·q + div(v)·p
  * Direct sparse solve via a.mat.Inverse(FreeDofs)
  * Pressure indeterminacy: NGSolve compound H1+H1 has
    natural zero-mean from FreeDofs when no Dirichlet on
    pressure; without pinning, the iterative direct
    solver returns up to a constant — Integrate p_h on
    the mesh and subtract the mean to compare to
    zero-mean p_exact.

MMS:
  ψ = sin(πx)²·sin(πy)²
  u_exact = curl(ψ) = (π·sin(πx)²·sin(2πy),
                       -π·sin(2πx)·sin(πy)²)
  p_exact = sin(πx)·cos(πy)  (zero mean on [0,1]²)
  f = -ν·Δu + grad(p)        (CoefficientFunction, ν = 1)
"""
from __future__ import annotations

import logging
import math
import sys

logging.disable(logging.CRITICAL)

import ngsolve as ngs
from netgen.geom2d import unit_square


def run_stokes_mms(maxh: float) -> tuple[float, float]:
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=maxh))
    V = ngs.VectorH1(mesh, order=2,
                      dirichlet="bottom|right|top|left")
    Q = ngs.H1(mesh, order=1)
    X = V * Q
    (u, p), (v, q) = X.TnT()

    nu = 1.0
    x = ngs.x
    y = ngs.y
    pi = math.pi

    # u_exact (vector)
    u_ex = ngs.CoefficientFunction((
        pi * ngs.sin(pi * x) ** 2
            * ngs.sin(2.0 * pi * y),
        -pi * ngs.sin(2.0 * pi * x)
            * ngs.sin(pi * y) ** 2,
    ))
    p_ex = ngs.sin(pi * x) * ngs.cos(pi * y)

    # f = -ν·Δu + grad(p)  (analytic)
    lap_u = ngs.CoefficientFunction((
        2.0 * pi ** 3
            * ngs.cos(2.0 * pi * x)
            * ngs.sin(2.0 * pi * y)
        - 4.0 * pi ** 3
            * ngs.sin(pi * x) ** 2
            * ngs.sin(2.0 * pi * y),
        4.0 * pi ** 3
            * ngs.sin(2.0 * pi * x)
            * ngs.sin(pi * y) ** 2
        - 2.0 * pi ** 3
            * ngs.sin(2.0 * pi * x)
            * ngs.cos(2.0 * pi * y),
    ))
    gp = ngs.CoefficientFunction((
        pi * ngs.cos(pi * x) * ngs.cos(pi * y),
        -pi * ngs.sin(pi * x) * ngs.sin(pi * y),
    ))
    f = -nu * lap_u + gp

    a = ngs.BilinearForm(X)
    a += (nu * ngs.InnerProduct(ngs.Grad(u),
                                 ngs.Grad(v))
          * ngs.dx)
    a += ngs.div(u) * q * ngs.dx
    a += ngs.div(v) * p * ngs.dx
    a.Assemble()

    ll = ngs.LinearForm(X)
    ll += ngs.InnerProduct(f, v) * ngs.dx
    ll.Assemble()

    gfu = ngs.GridFunction(X)
    # Set Dirichlet u = u_exact on ∂Ω via Set
    gfu.components[0].Set(
        u_ex, definedon=mesh.Boundaries(
            "bottom|right|top|left"))

    # Pin one pressure DOF to remove the constant-pressure
    # null space. NGSolve compound spaces don't pin
    # pressure via Dirichlet on the H1 sub-space (we
    # didn't pass dirichlet= to Q), so the saddle-point
    # is rank-deficient and Pardiso reports phase-33
    # error -4. Workaround: clear the FreeDofs bit for
    # one pressure DOF (offset = V.ndof + 0). Setting
    # gfu.vec[V.ndof] = 0 fixes the value to zero
    # (matches p_exact(0,0) = 0 approximately; the
    # mean-subtraction below removes any residual
    # offset).
    free = X.FreeDofs()
    free.Clear(V.ndof)
    gfu.vec[V.ndof] = 0.0

    r = ll.vec.CreateVector()
    r.data = ll.vec - a.mat * gfu.vec
    inv = None
    for solver_name in ["pardiso", "mumps", "umfpack",
                         "sparsecholesky"]:
        try:
            inv = a.mat.Inverse(free, solver_name)
            break
        except Exception:
            pass
    if inv is None:
        inv = a.mat.Inverse(free)
    gfu.vec.data += inv * r

    u_h = gfu.components[0]
    # NGSolve's catalog weak form `a += div(u)*q + div(v)*p`
    # assembles block [[K, B^T], [B, 0]] with B = +div,
    # which is the OPPOSITE sign convention to fenics
    # (which uses -p·div(v) - q·div(u) giving B = -div).
    # Consequence: the computed p_h equals -p_exact in
    # the canonical sign convention. Negate before
    # computing the error, then subtract mean to remove
    # the indeterminacy from the single-DOF pin.
    p_h_raw = gfu.components[1]
    p_h_signed = -p_h_raw
    p_mean = ngs.Integrate(p_h_signed, mesh)
    p_diff = p_h_signed - p_mean - p_ex
    err_u = math.sqrt(
        ngs.Integrate(ngs.InnerProduct(u_h - u_ex,
                                        u_h - u_ex),
                       mesh))
    err_p = math.sqrt(
        ngs.Integrate(p_diff * p_diff, mesh))
    return err_u, err_p


def main() -> int:
    print(f"ngsolve_version={ngs.__version__}")
    err_u_16, err_p_16 = run_stokes_mms(maxh=1.0 / 16)
    err_u_08, err_p_08 = run_stokes_mms(maxh=1.0 / 8)
    eoc_u = (math.log(err_u_08 / err_u_16)
             / math.log(2.0)
             if err_u_16 > 0 else float("nan"))
    eoc_p = (math.log(err_p_08 / err_p_16)
             / math.log(2.0)
             if err_p_16 > 0 else float("nan"))
    print(f"P2_u_maxh16_l2err={err_u_16:.6e}_tol=1.5e-03")
    print(f"P1_p_maxh16_l2err={err_p_16:.6e}_tol=1e-02")
    print(f"P2_u_eoc_maxh8_to_maxh16="
          f"{eoc_u:.3f}_expected=3.0")
    print(f"P1_p_eoc_maxh8_to_maxh16="
          f"{eoc_p:.3f}_expected=2.0")

    # Netgen unstructured mesh has variable effective h at
    # the same nominal maxh — observed empirically:
    #   P2 u L2 ≈ 7.7e-4 (vs Cartesian 1.4e-4)
    #   P1 p L2 ≈ 3.7e-3 (vs Cartesian 4.4e-4)
    #   EOCs: u ≈ 3.28, p ≈ 3.48 (super-convergence
    #         amplified on unstructured mesh)
    fail_reasons = []
    if err_u_16 > 1.5e-3:
        fail_reasons.append(
            f"P2 u L2 err {err_u_16:.3e} > 1.5e-3")
    if err_p_16 > 1e-2:
        fail_reasons.append(
            f"P1 p L2 err {err_p_16:.3e} > 1e-2")
    if not (2.0 <= eoc_u <= 4.0):
        fail_reasons.append(
            f"P2 u EOC {eoc_u:.3f} outside [2.0, 4.0]")
    if not (1.3 <= eoc_p <= 4.0):
        fail_reasons.append(
            f"P1 p EOC {eoc_p:.3f} outside [1.3, 4.0]")

    if not fail_reasons:
        return 0
    for r in fail_reasons:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
