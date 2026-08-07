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

Mutation control: T2_MUTATE=1 applies exactly that fix at the
pathology site -- the `shift` argument of ArnoldiSolver becomes a
non-zero value near the first cavity eigenvalue (2*pi^2) instead of
0.0 -- so (A - shift*M) is regular, UMFPACK factorises it and no
exception is raised. Re-run with `T2_MUTATE=1 python source.py`.
"""
from __future__ import annotations

import math
import os
import sys
import traceback

from netgen.geom2d import unit_square
from ngsolve import HCurl, Mesh, BilinearForm, curl, dx, MultiVector, ArnoldiSolver

MUTATE = os.environ.get("T2_MUTATE") == "1"
# 0.0 is the pathology; the catalog fix is a shift near the expected
# eigenvalue (k^2 = 2*pi^2 for the first unit-square cavity mode).
SHIFT = (2.0 * math.pi ** 2) if MUTATE else 0.0


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
                       list(vecs), shift=SHIFT)
    except Exception:
        traceback.print_exc()
        return 1
    print(f"ERROR: ArnoldiSolver with shift={SHIFT} succeeded "
          "(catalog claim wrong)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
