"""Tier-2: NGSolve vector source on scalar LinearForm raises.

A common user error: a 2-D CoefficientFunction (e.g. body-force
gradient or a vector field) accidentally fed into a SCALAR
LinearForm raises NgException at Assemble:

    SymbolicLFI needs scalar-valued CoefficientFunction

This is the LinearForm dual of the existing
`hcurl_grad_mismatch` and `h1_curl_operator_unknown` fixtures
(which target BilinearForm integrands). Together they cover
the integrand-shape mismatch family from both sides.
"""
from __future__ import annotations

import sys
import traceback

from netgen.geom2d import unit_square
from ngsolve import (
    CoefficientFunction,
    H1,
    LinearForm,
    Mesh,
    dx,
)


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    fes = H1(mesh, order=1)
    _, v = fes.TnT()
    f = LinearForm(fes)
    # Vector source on scalar form — wrong shape.
    f += CoefficientFunction((1.0, 2.0)) * v * dx
    try:
        f.Assemble()
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: Assemble accepted a vector source on a scalar "
          "LinearForm", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
