"""Tier-2: ArnoldiSolver shift=0 on curl-curl raises.

Pitfall (NGSolve maxwell#5): ArnoldiSolver eigenvalue solver
uses shift-and-invert. With shift=0 on a curl-curl matrix that
has the gradient kernel as null space, the (A - 0*M) operator
stays singular and the UMFPACK factorisation inside the
inverse step fails:

  NgException: UmfpackInverse: Numeric factorization failed.
  UMFPACK V5.7.4 (Feb 1, 2016): WARNING: matrix is singular

Catalog-recommended fix: set shift near the expected
eigenvalue (k^2_estimate from analytic cavity formula).
"""
from __future__ import annotations

import sys
import traceback

from netgen.geom2d import unit_square
from ngsolve import HCurl, Mesh, BilinearForm, curl, dx, MultiVector, ArnoldiSolver


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    fes = HCurl(mesh, order=2, dirichlet="bottom|right|top|left",
                nograds=False)
    u, v = fes.TnT()
    a = BilinearForm(fes); a += curl(u) * curl(v) * dx; a.Assemble()
    m = BilinearForm(fes); m += u * v * dx; m.Assemble()
    vecs = MultiVector(a.mat.CreateColVector(), 4)
    try:
        ArnoldiSolver(a.mat, m.mat, fes.FreeDofs(),
                       list(vecs), shift=0.0)
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: ArnoldiSolver with shift=0 succeeded "
          "(catalog claim wrong)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
