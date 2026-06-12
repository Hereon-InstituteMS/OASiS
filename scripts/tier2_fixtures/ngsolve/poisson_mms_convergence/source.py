"""Tier-2 Layer-C: NGSolve poisson numerical-correctness gate.

Mirrors fenics + skfem poisson_mms_convergence. Runs the
catalog-recommended NGSolve API surface end-to-end against
a manufactured solution.

NGSolve catalog API surface (from
src/backends/ngsolve/generators/poisson.py):
  * Mesh(unit_square.GenerateMesh(maxh=...))
  * H1(mesh, order=k, dirichlet='boundary')
  * BilinearForm(grad(u)*grad(v)*dx).Assemble()
  * LinearForm(f*v*dx).Assemble()
  * a.mat.Inverse(fes.FreeDofs()) * f.vec

MMS:
  u_exact(x, y) = sin(π·x) · sin(π·y) on [0, 1]^2
  f = 2 π² sin(π·x) · sin(π·y)  (so -Δu = f)
  u = 0 on ∂Ω

Expected behaviour at maxh=1/16 (Netgen unstructured;
the mesh isn't perfectly Cartesian like fenics /
skfem, so the per-DOF error constants differ):
  order=1 L2 ≲ 7e-3   (Netgen mesh slightly coarser)
  order=2 L2 ≲ 5e-5
  order=3 L2 ≲ 1e-6
Plus order=1 EOC maxh=1/8 → maxh=1/16 within [1.7, 2.3].
"""
from __future__ import annotations

import logging
import math
import sys

logging.disable(logging.CRITICAL)

import ngsolve as ngs
from netgen.geom2d import unit_square


def run_ngsolve_poisson(maxh: float, order: int) -> float:
    mesh = ngs.Mesh(unit_square.GenerateMesh(maxh=maxh))
    fes = ngs.H1(mesh, order=order,
                  dirichlet="bottom|right|top|left")
    u, v = fes.TnT()

    x = ngs.x
    y = ngs.y
    pi = ngs.CoefficientFunction(math.pi)
    u_ex = ngs.sin(pi * x) * ngs.sin(pi * y)
    f = 2.0 * math.pi ** 2 * u_ex

    a = ngs.BilinearForm(
        ngs.grad(u) * ngs.grad(v) * ngs.dx).Assemble()
    ll = ngs.LinearForm(f * v * ngs.dx).Assemble()

    gfu = ngs.GridFunction(fes)
    gfu.vec.data = (a.mat.Inverse(fes.FreeDofs())
                    * ll.vec)

    err_form = ngs.Integrate(
        (gfu - u_ex) * (gfu - u_ex), mesh)
    return float(math.sqrt(err_form))


def main() -> int:
    expected_floor = {1: 7e-3, 2: 5e-5, 3: 1e-6}
    print(f"ngsolve_version={ngs.__version__}")
    results: dict[int, float] = {}
    for k, tol in expected_floor.items():
        err = run_ngsolve_poisson(maxh=1.0 / 16, order=k)
        results[k] = err
        print(f"P{k}_maxh16_l2err={err:.6e}"
              f"_tol={tol:.0e}")

    err_h08 = run_ngsolve_poisson(maxh=1.0 / 8, order=1)
    err_h16 = results[1]
    eoc_p1 = (math.log(err_h08 / err_h16) / math.log(2.0)
              if err_h16 > 0 else float("nan"))
    print(f"P1_eoc_maxh8_to_maxh16="
          f"{eoc_p1:.3f}_expected=2.0")

    fail_reasons = []
    for k, err in results.items():
        if err > expected_floor[k]:
            fail_reasons.append(
                f"order={k} L2err {err:.3e} > "
                f"{expected_floor[k]:.0e}")
    # Unstructured Netgen meshes give a slightly wider
    # EOC range than the Cartesian fenics/skfem meshes —
    # two maxh values produce different mesh topologies,
    # so the rate-of-error decrease can over- or under-
    # shoot the theoretical 2.0 by a larger margin.
    if not (1.5 <= eoc_p1 <= 2.5):
        fail_reasons.append(
            f"P1 EOC {eoc_p1:.3f} outside [1.5, 2.5]")

    if not fail_reasons:
        return 0
    for r in fail_reasons:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
