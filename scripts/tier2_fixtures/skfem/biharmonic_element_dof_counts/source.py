"""Tier-2: skfem biharmonic element DOF counts match catalog claims.

Pitfalls (skfem biharmonic#0-2): Morley, Argyris, and BFS are
the three plate-element classes for biharmonic problems in
skfem. The catalog claims:

  - Morley: 6 DOFs per tri (3 vertex + 3 edge-midpoint
    normal-derivative DOFs)
  - Argyris: 21 DOFs per tri (5th-degree polynomial, C^1
    continuity)
  - BFS (Bogner-Fox-Schmit): 16 DOFs per quad (C^1)

All three element classes must be registered in skfem and
Basis(mesh, elem).Nbfun must match.

Mutation control (INVERSE -- this fixture contains no pathology to
remove; it is a pure catalog-claim gate on the right variant only):
T2_MUTATE=1 swaps each plate element for the C^0 Lagrange element a
user might reach for instead (Morley -> ElementTriP1, Argyris ->
ElementTriP2, BFS -> ElementQuad1), which proves the printed numbers
are read out of skfem rather than restated from the catalog.  Re-run
with T2_MUTATE=1 python source.py.
"""
from __future__ import annotations

import os
import sys

import skfem

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh_tri = skfem.MeshTri()
    mesh_quad = skfem.MeshQuad()
    # Under T2_MUTATE the C^0 stand-ins go in at the element argument.
    tri_plate_1 = (skfem.ElementTriMorley() if not MUTATE
                   else skfem.ElementTriP1())
    tri_plate_2 = (skfem.ElementTriArgyris() if not MUTATE
                   else skfem.ElementTriP2())
    quad_plate = (skfem.ElementQuadBFS() if not MUTATE
                  else skfem.ElementQuad1())
    morley = skfem.Basis(mesh_tri, tri_plate_1)
    argyris = skfem.Basis(mesh_tri, tri_plate_2)
    bfs = skfem.Basis(mesh_quad, quad_plate)
    print(f"Morley_Nbfun={morley.Nbfun}")
    print(f"Argyris_Nbfun={argyris.Nbfun}")
    print(f"BFS_Nbfun={bfs.Nbfun}")
    if (morley.Nbfun == 6
            and argyris.Nbfun == 21
            and bfs.Nbfun == 16):
        return 0
    print("ERROR: Nbfun did not match catalog DOF claims",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
