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
"""
from __future__ import annotations

import sys

import skfem


def main() -> int:
    mesh_tri = skfem.MeshTri()
    mesh_quad = skfem.MeshQuad()
    morley = skfem.Basis(mesh_tri, skfem.ElementTriMorley())
    argyris = skfem.Basis(mesh_tri, skfem.ElementTriArgyris())
    bfs = skfem.Basis(mesh_quad, skfem.ElementQuadBFS())
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
