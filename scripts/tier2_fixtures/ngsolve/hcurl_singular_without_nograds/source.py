"""Tier-2: HCurl without nograds — curl-curl matrix is singular.

Pitfall (NGSolve maxwell#0): For magnetostatic / source problems
on HCurl with no nograds and no regularisation, the curl-curl
bilinear form has the gradient kernel as null space. The mass
matrix is also rank-deficient on the same kernel, so the
shifted-Helmholtz factorisation in BilinearForm.mat.Inverse
fails:

  NgException: UmfpackInverse: Numeric factorization failed.
  UMFPACK V5.7.4 (Feb 1, 2016): WARNING: matrix is singular

The catalog-recommended fix is HCurl(..., nograds=True) (removes
the gradient kernel) plus a 1e-8*u*v*dx regularisation, or a
non-zero shift far from spectrum.
"""
from __future__ import annotations

import sys
import traceback

from netgen.geom2d import unit_square
from ngsolve import HCurl, Mesh, BilinearForm, curl, dx


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    fes = HCurl(mesh, order=1, dirichlet="left|right|top|bottom",
                nograds=False)
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += curl(u) * curl(v) * dx  # no regularisation
    a.Assemble()
    try:
        a.mat.Inverse(fes.FreeDofs())
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: Inverse succeeded on the singular curl-curl "
          "matrix (catalog claim wrong)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
