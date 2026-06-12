"""Tier-2: NGSolve HCurl with grad-grad bilinear form raises NgException.

A common Maxwell-newbie pitfall is to write `grad(u)*grad(v)*dx`
on an HCurl basis — this is the H1 elliptic form, not the H(curl)
form (`curl(u)*curl(v)*dx`). NGSolve's SymbolicBFI rejects the
non-scalar integrand:

    NgException: SymbolicBFI needs scalar-valued CoefficientFunction

The Tier-2 runner greps for that exact phrase.
"""
from __future__ import annotations

import sys
import traceback

from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    HCurl,
    Mesh,
    dx,
    grad,
)


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    fes = HCurl(mesh, order=1)
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += grad(u) * grad(v) * dx  # H1 form on HCurl basis: WRONG
    try:
        a.Assemble()
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: Assemble did not raise — pitfall claim no longer "
          "holds for this NGSolve version",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
