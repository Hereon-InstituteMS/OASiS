"""Tier-2: ElementQuadBFS has 16 DOFs per quad and is quad-only.

Claim: skfem biharmonic#2 -- ElementQuadBFS (Bogner-Fox-Schmit) is C^1 on quads
with 16 DOFs per element: value, du/dx, du/dy and d2u/dxdy at each of 4 vertices.

Wrong variant (a): using the C^0 ElementQuad1 where a C^1 plate element is
needed. Its 4 DOFs per element cannot carry a derivative DOF, and a biharmonic
ddot(u.hess, v.hess) form over it raises inside numpy.einsum.

Wrong variant (b): putting ElementQuadBFS on a triangular mesh. The element is
defined on quads only and skfem rejects the pairing.

Mutation control: T2_MUTATE=1 applies the documented fix at pathology (b) --
the mesh ElementQuadBFS is paired with becomes a MeshQuad instead of a
MeshTri -- so the pairing is accepted and the rejection evidence disappears.
Re-run with T2_MUTATE=1 python source.py.
"""
from __future__ import annotations

import os
import sys

from skfem import (
    Basis,
    BilinearForm,
    ElementQuad1,
    ElementQuadBFS,
    MeshQuad,
    MeshTri,
)
from skfem.helpers import ddot

MUTATE = os.environ.get("T2_MUTATE") == "1"


@BilinearForm
def bilaplace(u, v, w):
    return ddot(u.hess, v.hess)


def main() -> int:
    ok = True
    m = MeshQuad().refined(2)

    bfs = Basis(m, ElementQuadBFS())
    quad1 = Basis(m, ElementQuad1())
    print(f"bfs_nbfun={bfs.Nbfun}")
    print(f"quad1_nbfun={quad1.Nbfun}")
    print(f"bfs_is_four_times_quad1={bfs.Nbfun == 4 * quad1.Nbfun}")
    if bfs.Nbfun != 16:
        print(f"FAIL: BFS Nbfun is {bfs.Nbfun}, claim says 16", file=sys.stderr)
        ok = False

    # --- WRONG variant (a): C^0 quad element in a biharmonic form -------
    raised_c0 = ""
    try:
        bilaplace.assemble(quad1)
    except ValueError as exc:
        raised_c0 = str(exc)
    print(f"quad1_bilaplace_raises={bool(raised_c0)}")
    print(f"quad1_bilaplace_msg={raised_c0!r}")
    if not raised_c0:
        print("FAIL: the C^0 quad element assembled a biharmonic form without "
              "complaint", file=sys.stderr)
        ok = False

    # --- WRONG variant (b): BFS on a triangular mesh --------------------
    # The pathology: the quad-only element is paired with a MeshTri.
    # T2_MUTATE=1 applies the documented fix -- give it a quad mesh -- here.
    tri_mesh = (MeshTri().refined(1) if not MUTATE
                else MeshQuad().refined(1))
    raised_tri = ""
    try:
        Basis(tri_mesh, ElementQuadBFS())
    except Exception as exc:
        raised_tri = f"{type(exc).__name__}: {exc}"
    print(f"bfs_on_meshtri_raises={bool(raised_tri)}")
    print(f"bfs_on_meshtri_msg={raised_tri!r}")
    if not raised_tri:
        print("FAIL: ElementQuadBFS was accepted on a MeshTri", file=sys.stderr)
        ok = False

    # --- RIGHT variant --------------------------------------------------
    K = bilaplace.assemble(bfs)
    good = K.nnz > 0 and abs(K).max() > 0.0
    print(f"bfs_bilaplace_assembles={good}")
    if not good:
        print("FAIL: BFS bilaplacian assembled empty", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
