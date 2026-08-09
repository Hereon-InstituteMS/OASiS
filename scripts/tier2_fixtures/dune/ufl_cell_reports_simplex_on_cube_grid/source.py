"""Tier-2: a DUNE-fem space reports a SIMPLEX UFL cell on a CUBE grid.

dune/ufl/__init__.py::_cell(dimWorld, dimDomain) maps the dimension
straight onto ufl.interval / ufl.triangle / ufl.tetrahedron — the UFL
cell carries no information about what the grid is really made of. On a
YaspGrid built by dune.grid.structuredGrid, whose cells are
quadrilaterals (2D) or hexahedra (3D), the space still prints
"<Lagrange1 on a triangle>" / "on a tetrahedron".

This matters because a model transferring dolfinx habits reads
mesh.ufl_cell() to decide an element family, a quadrature degree or a
basix element. Here that read is a lie, and it is a QUIET one: nothing
raises, the form assembles, and the wrong conclusion is drawn about the
discretisation.

The ELEMENT COUNT and geometry type are the honest witnesses — the dof
count is NOT. Continuous Lagrange over the same cartesianDomain has 25
dofs at order 1 whether the grid is 16 quadrilaterals or 32 triangles,
because both carry the same 25 vertices (measured: b5 convergence study,
"CUBE k=1 n=4 dofs=25" and "SIMPLEX k=1 n=4 dofs=25"). So this fixture
does not use dof counts to discriminate cube from simplex; it only
records space.size as context.

The fixture asserts:
  (a) the grid really is made of quadrilaterals / hexahedra
      (the per-element geometry types), with 16 cells in 2D
  (b) the space nevertheless reports triangle / tetrahedron
  (c) an ALUGrid built from the SAME cartesianDomain does produce
      simplices, with exactly twice the element count in 2D

Verified by execution against dune-fem 2.12.0.2 on 2026-08-03.

MUTATION CONTROL. T2_MUTATE=1 builds the 2D grid under test with
aluConformGrid on the same cartesianDomain, so the reported UFL cell
'triangle' is TRUE rather than a lie — the pathology removed.
grid2d_element_types then reads ['triangle'], so
"grid2d_element_types=['quadrilateral']" is no longer printed and a
FAIL: line appears. The ALU grid is one the fixture already builds.
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dune.grid import structuredGrid, cartesianDomain      # noqa: E402
from dune.alugrid import aluConformGrid                    # noqa: E402
from dune.fem.space import lagrange                        # noqa: E402

N = 4


def main() -> int:
    if MUTATE:
        print("mutation=the_2d_grid_under_test_is_a_simplex_alugrid_"
              "so_the_reported_ufl_cell_is_true")
        gv2 = aluConformGrid(cartesianDomain([0, 0], [1, 1], [N, N]),
                             dimgrid=2)
    else:
        gv2 = structuredGrid([0, 0], [1, 1], [N, N])
    sp2 = lagrange(gv2, order=1)
    types2 = sorted({str(e.type) for e in gv2.elements})

    gv3 = structuredGrid([0, 0, 0], [1, 1, 1], [2, 2, 2])
    sp3 = lagrange(gv3, order=1)
    types3 = sorted({str(e.type) for e in gv3.elements})

    print(f"grid2d_element_types={types2}")
    print(f"grid2d_n_elements={gv2.size(0)}")
    print(f"space2d_ufl_cell={sp2.cell()}")
    print(f"space2d_ufl_element={sp2.ufl_element()}")
    # Context only — NOT a cube-vs-simplex discriminator: continuous
    # Lagrange over the same domain has the same dof count on either
    # grid (they share the vertices).
    print(f"space2d_size={sp2.size} vertex_count={gv2.size(2)}")
    print(f"grid3d_element_types={types3}")
    print(f"space3d_ufl_cell={sp3.cell()}")

    gsim = aluConformGrid(cartesianDomain([0, 0], [1, 1], [N, N]),
                          dimgrid=2)
    types_sim = sorted({str(e.type) for e in gsim.elements})
    print(f"alugrid_element_types={types_sim}")
    print(f"alugrid_n_elements={gsim.size(0)}")

    fail = []
    if types2 != ["quadrilateral"]:
        fail.append(f"2D structuredGrid cells {types2} != "
                    f"['quadrilateral']")
    if types3 != ["hexahedron"]:
        fail.append(f"3D structuredGrid cells {types3} != "
                    f"['hexahedron']")
    if str(sp2.cell()) != "triangle":
        fail.append(f"2D space UFL cell is {sp2.cell()} — the "
                    f"dimension->simplex mapping in "
                    f"dune/ufl/__init__.py::_cell has changed; the "
                    f"catalog claim needs re-checking")
    if str(sp3.cell()) != "tetrahedron":
        fail.append(f"3D space UFL cell is {sp3.cell()} — see above")
    if sp2.size != gv2.size(2):
        fail.append(f"order-1 space.size {sp2.size} != vertex count "
                    f"{gv2.size(2)} — continuous Lagrange order 1 "
                    f"should have exactly one dof per vertex")
    if gv2.size(0) != N * N:
        fail.append(f"cube grid element count {gv2.size(0)} != "
                    f"{N * N}")
    if types_sim != ["triangle"]:
        fail.append(f"aluConformGrid cells {types_sim} != "
                    f"['triangle']")
    if gsim.size(0) != 2 * N * N:
        fail.append(f"aluConformGrid element count {gsim.size(0)} != "
                    f"2*{N * N}")

    if not fail:
        print("dune_ufl_cell_is_always_simplex=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
