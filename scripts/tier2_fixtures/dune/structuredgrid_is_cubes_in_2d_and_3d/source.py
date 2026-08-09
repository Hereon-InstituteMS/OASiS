"""Tier-2: structuredGrid makes CUBES, in 2D and in 3D.

  poisson#5       structuredGrid gives quadrilaterals in 2D and
                  hexahedra in 3D, never simplices; count the CELLS to
                  tell the two apart, because the dof count does not.
  poisson_mms#5   3D structuredGrid is hexahedral, so a Lagrange space
                  of order k has exactly (n*k+1)^3 scalar dofs — which
                  is why cost grows with the cube of n*k.

Both are the same measurement on the same two grid objects, so they
share a fixture. No weak form is built, so nothing here JIT-compiles a
scheme; only the grid and space modules are needed.

Verified by execution against dune-fem 2.12.0.2.

MUTATION CONTROL. T2_MUTATE=1 builds the 2D grid under test with
aluConformGrid on the same cartesianDomain — the pathology removed,
because that grid really is simplicial. structured_2d_types then reads
['triangle'] and structured_2d_cells 32, so
"structured_2d_types=['quadrilateral']" and 'structured_2d_cells=16'
are no longer printed and a FAIL: line appears. The ALU grid is one the
fixture already builds.
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dune.grid import structuredGrid, cartesianDomain          # noqa: E402
from dune.fem.space import lagrange                            # noqa: E402
from dune.alugrid import aluConformGrid                        # noqa: E402


def main() -> int:
    fail: list[str] = []

    # ── 2D: cubes, and the SAME dof count as the simplex grid ────────
    if MUTATE:
        print("mutation=the_grid_under_test_is_a_simplex_alugrid")
        cube2 = aluConformGrid(cartesianDomain([0, 0], [1, 1], [4, 4]))
    else:
        cube2 = structuredGrid([0, 0], [1, 1], [4, 4])
    types2 = sorted({str(e.type) for e in cube2.elements})
    print(f"structured_2d_cells={cube2.size(0)}")
    print(f"structured_2d_types={types2}")
    if cube2.size(0) != 16 or types2 != ["quadrilateral"]:
        fail.append(f"structuredGrid([0,0],[1,1],[4,4]) gave "
                    f"{cube2.size(0)} cells of type {types2}; the "
                    f"claim is 16 quadrilaterals")

    simplex2 = aluConformGrid(cartesianDomain([0, 0], [1, 1], [4, 4]))
    types2s = sorted({str(e.type) for e in simplex2.elements})
    print(f"alu_2d_cells={simplex2.size(0)}")
    print(f"alu_2d_types={types2s}")
    if simplex2.size(0) != 32 or types2s != ["triangle"]:
        fail.append(f"aluConformGrid on the same domain gave "
                    f"{simplex2.size(0)} cells of type {types2s}; the "
                    f"claim is 32 triangles")

    # The dof count CANNOT distinguish them — same vertices.
    n_cube = lagrange(cube2, order=1).size
    n_simp = lagrange(simplex2, order=1).size
    print(f"p1_dofs_cube_vs_simplex={n_cube},{n_simp}")
    print(f"dof_count_cannot_tell_them_apart={n_cube == n_simp}")
    if n_cube != n_simp:
        fail.append(f"P1 dof counts differ ({n_cube} vs {n_simp}); the "
                    f"claim that only the CELL count separates the two "
                    f"grids no longer holds")

    # ── 3D: hexahedra, and (n*k+1)^3 dofs ───────────────────────────
    cube3 = structuredGrid([0] * 3, [1] * 3, [4] * 3)
    types3 = sorted({str(e.type) for e in cube3.elements})
    print(f"structured_3d_cells={cube3.size(0)}")
    print(f"structured_3d_types={types3}")
    if cube3.size(0) != 64 or types3 != ["hexahedron"]:
        fail.append(f"3D structuredGrid gave {cube3.size(0)} cells of "
                    f"type {types3}; the claim is 64 hexahedra")

    n = 4
    for order in (1, 2):
        got = lagrange(cube3, order=order).size
        want = (n * order + 1) ** 3
        print(f"dofs_3d_order{order}={got} expected={want} "
              f"matches={got == want}")
        if got != want:
            fail.append(f"3D order-{order} space has {got} dofs, not "
                        f"(n*k+1)^3 = {want}; the cubic cost law in "
                        f"the claim is measured off this identity")

    if not fail:
        print("dune_structuredgrid_cell_shapes_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
