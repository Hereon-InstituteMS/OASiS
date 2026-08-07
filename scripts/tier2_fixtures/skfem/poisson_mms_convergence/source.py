"""Tier-2 Layer-C: skfem poisson numerical-correctness gate.

Mirrors fenics/poisson_mms_convergence. Runs the catalog-
recommended skfem API surface end-to-end against a
manufactured solution and asserts numerical convergence.

skfem catalog API surface (from
src/backends/skfem/generators/poisson.py):
  * MeshTri (Cartesian unit-square mesh, .refined)
  * Basis with ElementTriPk (P1 / P2 / P3)
  * BilinearForm / LinearForm for custom RHS
  * skfem.condense + skfem.solve
  * skfem.models.poisson.laplace (alternative pre-built)

MMS:
  u_exact(x, y) = sin(π·x) · sin(π·y) on [0, 1]^2
  f = 2 π² sin(π·x) · sin(π·y)  (so -Δu = f)
  u = 0 on ∂Ω

Expected behaviour at h=1/32 (refine 5 from MeshTri()):
  P1 L2 error ≲ 5e-3   (O(h²))
  P2 L2 error ≲ 5e-5   (O(h³))
  P3 L2 error ≲ 5e-7   (O(h⁴))
Plus P1 EOC h=1/16 → h=1/32 within [1.7, 2.3].

Mutation control: this is a correctness gate, so the mutation
breaks the discretisation the gate certifies rather than
removing a pitfall. T2_MUTATE=1 drops the manufactured load
constant from 2π² to π², so f no longer equals -Δu and the
computed field is half the exact one. The L2 errors then sit
around 2.5e-1 at every order and the EOC collapses to ~0, so
"FAIL:" (a forbid_in_output string) appears four times and the
fixture goes red. Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

import skfem
from skfem import (
    MeshTri, Basis, BilinearForm, LinearForm, solve, condense)
from skfem.helpers import dot, grad
from scipy.sparse.linalg import spsolve  # noqa: F401

MUTATE = os.environ.get("T2_MUTATE") == "1"

# MMS consistency: -Δ(sin πx sin πy) = 2π² sin πx sin πy, so
# the load constant must be 2π².  Under mutation it is π².
LOAD_CONST = 2.0 if not MUTATE else 1.0


def run_skfem_poisson(refine: int, order: int) -> float:
    """Solve Poisson via the catalog-recommended skfem
    pattern and return L2 error vs the manufactured
    sin(πx)sin(πy) solution."""
    elem_cls = {1: skfem.ElementTriP1,
                2: skfem.ElementTriP2,
                3: skfem.ElementTriP3}[order]
    m = MeshTri().refined(refine)
    basis = Basis(m, elem_cls())

    @BilinearForm
    def stiffness(u, v, w):
        return dot(grad(u), grad(v))

    @LinearForm
    def load(v, w):
        x, y = w.x[0], w.x[1]
        return (LOAD_CONST * np.pi ** 2
                * np.sin(np.pi * x)
                * np.sin(np.pi * y)) * v

    K = stiffness.assemble(basis)
    F = load.assemble(basis)

    D = basis.get_dofs()
    u = solve(*condense(K, F, D=D))

    # L2 error: integrate (u_h - u_exact)^2 over the mesh.
    @LinearForm
    def err_sq(v, w):
        x, y = w.x[0], w.x[1]
        u_ex = np.sin(np.pi * x) * np.sin(np.pi * y)
        # 'v' here is the test function — to get an L2
        # functional we multiply (uh - u_ex)^2 by 1 (an
        # integration against the constant 1).
        return (w["uh"] - u_ex) ** 2 * 0.0 + 1.0
    # Simpler: use a Functional-like manual quadrature
    # by computing on the basis directly:
    from skfem import Functional

    @Functional
    def l2err(w):
        x, y = w.x[0], w.x[1]
        u_ex = np.sin(np.pi * x) * np.sin(np.pi * y)
        return (w["uh"] - u_ex) ** 2

    err_sq_val = l2err.assemble(basis, uh=basis.interpolate(u))
    return float(np.sqrt(err_sq_val))


def main() -> int:
    expected_floor = {1: 5e-3, 2: 5e-5, 3: 5e-7}
    print(f"skfem_version={skfem.__version__}")
    results: dict[int, float] = {}
    # refine=5 → 1024 triangles in MeshTri (h ~ 1/32)
    for k, tol in expected_floor.items():
        err = run_skfem_poisson(refine=5, order=k)
        results[k] = err
        print(f"P{k}_h32_l2err={err:.6e}_tol={tol:.0e}")

    # EOC: refine=4 vs refine=5 (h=1/16 vs h=1/32) for P1
    err_h16 = run_skfem_poisson(refine=4, order=1)
    err_h32 = results[1]
    eoc_p1 = (math.log(err_h16 / err_h32) / math.log(2.0)
              if err_h32 > 0 else float("nan"))
    print(f"P1_eoc_h16_to_h32={eoc_p1:.3f}_expected=2.0")

    fail_reasons = []
    for k, err in results.items():
        if err > expected_floor[k]:
            fail_reasons.append(
                f"P{k} L2err {err:.3e} > "
                f"{expected_floor[k]:.0e}")
    if not (1.7 <= eoc_p1 <= 2.3):
        fail_reasons.append(
            f"P1 EOC {eoc_p1:.3f} outside [1.7, 2.3]")

    if not fail_reasons:
        return 0
    for r in fail_reasons:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
