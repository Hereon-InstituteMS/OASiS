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

Mutation control:  T2_MUTATE=1 applies the documented fix at the
pathology site -- the space is built with nograds=True, so the
gradient kernel is not in it and the SAME curl-curl form (still
unregularised) factorises.  Inverse() then returns, no traceback
is printed, and all three expectations 'NgException',
'UmfpackInverse' and 'matrix is singular' vanish from the output.
Measured on this run: nograds=False gives ndof=26 and a singular
factorisation; nograds=True gives ndof=13 and succeeds.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys
import traceback

from netgen.geom2d import unit_square
from ngsolve import HCurl, Mesh, BilinearForm, curl, dx

# Mutation control: under T2_MUTATE=1 the space is built with nograds=True --
# the documented fix -- so the gradient kernel that makes the curl-curl matrix
# singular is not in the space at all.
MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    fes = HCurl(mesh, order=1, dirichlet="left|right|top|bottom",
                nograds=MUTATE)
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
