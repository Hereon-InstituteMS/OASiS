"""Tier-2: NGSolve curl() on H1 basis raises.

Dual to the existing `hcurl_grad_mismatch` fixture: applying
the H(curl) operator to an H1 basis is rejected just as
applying H1 grad() to HCurl is rejected. The exact text is
'Operator "curl" does not exist for H1HighOrderFESpace!'.

Verifying both directions covers the typical Maxwell-newbie
confusion of operators with spaces.

Mutation control: T2_MUTATE=1 applies the documented fix at the
pathology site -- the trial/test space becomes HCurl instead of H1,
so curl() is now paired with the space that defines it.  The form
then assembles, and 'NgException' / 'does not exist for H1' are
absent from the output, so the fixture goes red.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys
import traceback

from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    H1,
    HCurl,
    Mesh,
    curl,
    dx,
)

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    # The pathology: curl() asked of a scalar H1 space.  Under mutation the
    # documented fix is applied -- the space that defines curl is used instead.
    fes = H1(mesh, order=1) if not MUTATE else HCurl(mesh, order=1)
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
