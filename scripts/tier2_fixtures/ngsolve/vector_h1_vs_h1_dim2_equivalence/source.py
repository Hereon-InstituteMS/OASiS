"""Tier-2: VectorH1 and H1(dim=2) are operationally equivalent for elasticity.

Catalog-drift falsification: the existing NGSolve linear_elasticity
catalog claims 'Use VectorH1(mesh, order=2, dirichlet=...), NOT
H1(mesh, dim=2)' — implying H1(dim=2) is wrong or creates a
CompoundFESpace. That claim is false.

This fixture solves the same plane-strain elasticity problem with
both spaces and asserts:
  * Both succeed (no exception).
  * H1(dim=2) returns a class named 'H1' (not 'CompoundFESpace').
  * Their solution norms agree to 1e-10 (operational equivalence).
  * VectorH1 has dim=1 with ndof = 2 * (H1 dim=2).ndof (flat layout
    vs block layout).

Also confirms MatrixValued(H1, symmetric=True) exists and gfu.
components on a VectorH1 returns a tuple of ComponentGridFunction.
"""
from __future__ import annotations

import sys

import numpy as np
import netgen.geom2d as g2
from ngsolve import (
    H1,
    BilinearForm,
    CoefficientFunction,
    GridFunction,
    Grad,
    Id,
    InnerProduct,
    LinearForm,
    MatrixValued,
    Mesh,
    Trace,
    VectorH1,
    dx,
)


def _solve(fes) -> np.ndarray:
    u, v = fes.TnT()
    mu, lam = 0.5, 1.0

    def strain(w):
        return 0.5 * (Grad(w) + Grad(w).trans)

    def stress(w):
        return 2 * mu * strain(w) + lam * Trace(strain(w)) * Id(2)

    a = BilinearForm(InnerProduct(stress(u), strain(v)) * dx).Assemble()
    rhs = LinearForm(CoefficientFunction((0, -1)) * v * dx).Assemble()
    gfu = GridFunction(fes)
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * rhs.vec
    return np.array(gfu.vec)


def main() -> int:
    mesh = Mesh(g2.unit_square.GenerateMesh(maxh=0.5))

    fes_vec = VectorH1(mesh, order=2, dirichlet="left")
    fes_dim = H1(mesh, order=2, dim=2, dirichlet="left")

    type_vec = type(fes_vec).__name__
    type_dim = type(fes_dim).__name__
    print(f"vector_h1_type={type_vec}")
    print(f"h1_dim2_type={type_dim}")
    print(f"vector_h1_ndof={fes_vec.ndof}")
    print(f"h1_dim2_ndof={fes_dim.ndof}")
    print(f"vector_h1_dim={fes_vec.dim}")
    print(f"h1_dim2_dim={fes_dim.dim}")

    arr_vec = _solve(fes_vec)
    arr_dim = _solve(fes_dim)
    nrm_vec = float(np.linalg.norm(arr_vec))
    nrm_dim = float(np.linalg.norm(arr_dim))
    rel_diff = abs(nrm_vec - nrm_dim) / max(nrm_vec, 1e-300)
    print(f"vector_h1_norm={nrm_vec:.10e}")
    print(f"h1_dim2_norm={nrm_dim:.10e}")
    print(f"norm_rel_diff={rel_diff:.3e}")

    # Total scalar dofs match (flat vs block layout)
    total_scalar_vec = fes_vec.ndof
    total_scalar_dim = fes_dim.ndof * fes_dim.dim
    print(f"total_scalar_dofs_vec={total_scalar_vec}")
    print(f"total_scalar_dofs_dim={total_scalar_dim}")
    same_dofs = total_scalar_vec == total_scalar_dim

    # MatrixValued + symmetric=True kwarg shape
    mv = MatrixValued(H1(mesh, order=1), symmetric=True)
    type_mv = type(mv).__name__
    print(f"matrix_valued_type={type_mv} ndof={mv.ndof}")

    # gfu.components on VectorH1
    gfu_vec = GridFunction(fes_vec)
    comps = gfu_vec.components
    print(f"components_kind={type(comps).__name__} len={len(comps)}")
    print(f"component0_type={type(comps[0]).__name__}")

    ok = (
        type_vec == "VectorH1"
        and type_dim == "H1"
        and rel_diff < 1e-10
        and same_dofs
        and type_mv == "MatrixValued"
        and type(comps).__name__ == "tuple"
        and type(comps[0]).__name__ == "ComponentGridFunction"
    )
    if ok:
        return 0
    print("ERROR: equivalence assertions failed", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
