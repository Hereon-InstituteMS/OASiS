"""Tier-2: NGSolve curl() on H1 basis raises.

Dual to the existing `hcurl_grad_mismatch` fixture: applying
the H(curl) operator to an H1 basis is rejected just as
applying H1 grad() to HCurl is rejected. The exact text is
'Operator "curl" does not exist for H1HighOrderFESpace!'.

Verifying both directions covers the typical Maxwell-newbie
confusion of operators with spaces.
"""
from __future__ import annotations

import sys
import traceback

from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    H1,
    Mesh,
    curl,
    dx,
)


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    fes = H1(mesh, order=1)
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += curl(u) * curl(v) * dx  # curl on scalar H1: WRONG
    try:
        a.Assemble()
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: Assemble accepted curl on H1 basis",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
