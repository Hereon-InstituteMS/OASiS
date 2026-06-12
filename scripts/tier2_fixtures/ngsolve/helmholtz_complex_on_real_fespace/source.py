"""Tier-2: Helmholtz form with complex coefficient on a real FESpace
raises NgException at BilinearForm.Assemble.

Pitfall (NGSolve helmholtz#0): the catalog claim is that
complex=True must be set on the FESpace when the form contains
a complex coefficient (e.g. for an absorbing boundary
1j*k*u*v*ds, or 1j*k*u*v*dx). On a real-only FESpace
(complex=False, the default), BilinearForm.Assemble hits the
NGSolve type-check 'real Evaluate called for complex ScaleCF
in Assemble BilinearForm "biform_from_py"' from the BFI
assembler.
"""
from __future__ import annotations

import sys
import traceback

from netgen.geom2d import unit_square
from ngsolve import H1, Mesh, BilinearForm, grad, dx


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    fes = H1(mesh, order=1, complex=False)
    u, v = fes.TnT()
    k = 5.0
    a = BilinearForm(fes)
    # Add a complex-coefficient term — should be rejected at Assemble.
    a += grad(u) * grad(v) * dx + 1j * k * u * v * dx
    try:
        a.Assemble()
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: BilinearForm.Assemble accepted complex "
          "coefficient on real FESpace (catalog claim wrong)",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
