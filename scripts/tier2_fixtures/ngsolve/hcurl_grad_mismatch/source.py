"""Tier-2: NGSolve HCurl with grad-grad bilinear form raises NgException.

A common Maxwell-newbie pitfall is to write `grad(u)*grad(v)*dx`
on an HCurl basis — this is the H1 elliptic form, not the H(curl)
form (`curl(u)*curl(v)*dx`). NGSolve's SymbolicBFI rejects the
non-scalar integrand:

    NgException: SymbolicBFI needs scalar-valued CoefficientFunction

The Tier-2 runner greps for that exact phrase.

Mutation control: T2_MUTATE=1 applies the documented fix at the
pathology site -- the integrand becomes curl(u)*curl(v)*dx, the
H(curl) form the docstring names, instead of grad(u)*grad(v)*dx.
The integrand is then scalar-valued, Assemble() succeeds, and
'NgException' / 'SymbolicBFI' / 'scalar-valued' are absent from the
output, so the fixture goes red.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys
import traceback

from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    HCurl,
    Mesh,
    curl,
    dx,
    grad,
)

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    fes = HCurl(mesh, order=1)
    u, v = fes.TnT()
    a = BilinearForm(fes)
    # The pathology: the H1 elliptic form on an HCurl basis.  Under mutation
    # the documented fix is applied -- the H(curl) form curl(u)*curl(v)*dx.
    if not MUTATE:
        a += grad(u) * grad(v) * dx  # H1 form on HCurl basis: WRONG
    else:
        a += curl(u) * curl(v) * dx  # the documented fix
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
