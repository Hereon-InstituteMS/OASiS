"""Tier-2: UFL element classes removed in 2024+; basix.ufl
replaces them.

The fenics stokes generator (and any catalog text that
mentions ufl.FiniteElement / VectorElement / MixedElement /
TensorElement) is wrong for dolfinx 0.10+. Those names were
REMOVED from the ufl module — running an old example raises
AttributeError immediately.

Real construction (dolfinx 0.10 / ufl 2025.2 / basix 0.10):

  import basix.ufl
  P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
  Pv = basix.ufl.element("Lagrange", domain.basix_cell(), 2,
                         shape=(gdim,))
  TH = basix.ufl.mixed_element([Pv, P1])

This fixture asserts:
  * ufl.FiniteElement / VectorElement / MixedElement /
    TensorElement are all ABSENT.
  * basix.ufl.element / mixed_element / blocked_element are
    all PRESENT.
  * Constructing a Taylor-Hood mixed element via basix.ufl
    succeeds at runtime.
"""
from __future__ import annotations

import sys

import basix.ufl
import dolfinx
from dolfinx import fem, mesh
from mpi4py import MPI
import ufl


REMOVED_UFL_NAMES = (
    "FiniteElement",
    "VectorElement",
    "MixedElement",
    "TensorElement",
)
REQUIRED_BASIX_UFL_NAMES = (
    "element",
    "mixed_element",
    "blocked_element",
)


def main() -> int:
    # (1) Removed-from-ufl assertions.
    removed = {n: hasattr(ufl, n) for n in REMOVED_UFL_NAMES}
    print(f"removed_in_ufl={removed}")
    if any(removed.values()):
        print("FAIL: ufl still exposes removed element classes",
              file=sys.stderr)
        return 2

    # (2) basix.ufl replacement names present.
    replacement = {n: hasattr(basix.ufl, n)
                   for n in REQUIRED_BASIX_UFL_NAMES}
    print(f"basix_ufl_present={replacement}")
    if not all(replacement.values()):
        print("FAIL: basix.ufl missing required factories",
              file=sys.stderr)
        return 2

    # (3) Smoke: construct a Taylor-Hood mixed element on a
    # tiny mesh and build a function space. The mixed
    # element must compile through dolfinx.fem.functionspace.
    domain = mesh.create_unit_square(
        MPI.COMM_WORLD, 2, 2, mesh.CellType.triangle)
    gdim = domain.geometry.dim
    P2 = basix.ufl.element(
        "Lagrange", domain.basix_cell(), 2, shape=(gdim,))
    P1 = basix.ufl.element(
        "Lagrange", domain.basix_cell(), 1)
    TH = basix.ufl.mixed_element([P2, P1])
    W = fem.functionspace(domain, TH)
    # dim(W) = 2*n_P2_dofs + n_P1_dofs (Taylor-Hood)
    dim_W = W.dofmap.index_map.size_global * W.dofmap.index_map_bs
    print(f"taylor_hood_dim={dim_W}")
    if dim_W <= 0:
        print("FAIL: empty Taylor-Hood space",
              file=sys.stderr)
        return 2

    print(f"dolfinx_version={dolfinx.__version__}")
    print(f"ufl_version={ufl.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
