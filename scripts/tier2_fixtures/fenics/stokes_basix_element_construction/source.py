"""Tier-2: fenics stokes element construction (Taylor-Hood, MINI, equal-order).

Verifies the basix.ufl API strings from the stokes catalog actually
construct valid mixed elements in dolfinx 0.10:

  Taylor-Hood (inf-sup stable):
    P2v = basix.ufl.element('Lagrange', cell, 2, shape=(gdim,))
    P1  = basix.ufl.element('Lagrange', cell, 1)
    TH  = basix.ufl.mixed_element([P2v, P1])

  MINI (inf-sup stable):
    P1v = basix.ufl.element('Lagrange', cell, 1, shape=(gdim,))
    B   = basix.ufl.element('Bubble',   cell, gdim+1, shape=(gdim,))
    V_el = basix.ufl.enriched_element([P1v, B])
    MINI = basix.ufl.mixed_element([V_el, P1])

  Equal-order P1/P1 (constructs OK but UNSTABLE for Stokes):
    P1v / P1p / mixed_element([P1v, P1p])

This fixture confirms all three constructions return valid
FunctionSpace objects in dolfinx 0.10. The instability of P1/P1
is a separate physics-level claim (not exercised here, only
constructability is).
"""
from __future__ import annotations

import sys

import basix.ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI


def main() -> int:
    msh = dmesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    cell = msh.basix_cell()
    gdim = msh.geometry.dim
    print(f"cell={cell}")
    print(f"gdim={gdim}")

    # Taylor-Hood
    P2v = basix.ufl.element("Lagrange", cell, 2, shape=(gdim,))
    P1 = basix.ufl.element("Lagrange", cell, 1)
    TH = basix.ufl.mixed_element([P2v, P1])
    W_th = fem.functionspace(msh, TH)
    dim_th = (W_th.dofmap.index_map.size_global
              * W_th.dofmap.index_map_bs)
    print(f"taylor_hood_dim={dim_th}")
    print(f"taylor_hood_element_kind={type(TH).__name__}")

    # MINI
    P1v = basix.ufl.element("Lagrange", cell, 1, shape=(gdim,))
    B = basix.ufl.element("Bubble", cell, gdim + 1, shape=(gdim,))
    V_el = basix.ufl.enriched_element([P1v, B])
    MINI = basix.ufl.mixed_element([V_el, P1])
    W_mini = fem.functionspace(msh, MINI)
    dim_mini = (W_mini.dofmap.index_map.size_global
                * W_mini.dofmap.index_map_bs)
    print(f"mini_dim={dim_mini}")
    print(f"enriched_kind={type(V_el).__name__}")
    print(f"bubble_degree={gdim + 1}")

    # Equal-order P1/P1 — constructs but unstable for Stokes
    P1v2 = basix.ufl.element("Lagrange", cell, 1, shape=(gdim,))
    P1p = basix.ufl.element("Lagrange", cell, 1)
    EQ = basix.ufl.mixed_element([P1v2, P1p])
    W_eq = fem.functionspace(msh, EQ)
    dim_eq = (W_eq.dofmap.index_map.size_global
              * W_eq.dofmap.index_map_bs)
    print(f"equal_order_p1p1_dim={dim_eq}")

    # TH (187) > MINI (139) > P1/P1 (75) — strict ordering for a
    # 4x4 unit square triangulation in dolfinx 0.10.
    ordering_ok = dim_th > dim_mini > dim_eq
    print(f"dim_ordering_th_gt_mini_gt_eq={ordering_ok}")

    ok = (dim_th > 0 and dim_mini > 0 and dim_eq > 0 and ordering_ok)
    if ok:
        return 0
    print("ERROR: element construction or ordering failed",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
