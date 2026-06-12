"""Tier-2: NGSolve solvers.Newton uses `maxit` (singular), not `maxits`.

Catalog falsification: the existing NGSolve hyperelasticity
catalog (and several generator templates) call
  solvers.Newton(a, gfu, maxits=20, ...)
where the real ngsolve.solvers.Newton signature is

  Newton(a, u, freedofs=None, maxit=100, maxerr=1e-11,
         inverse='', dirichletvalues=None, dampfactor=1,
         printing=True, callback=None)

The kwarg is `maxit` (singular) with default 100. Calling with
`maxits=20` raises
  TypeError: Newton() got an unexpected keyword argument 'maxits'
The catalog template would crash an LLM-generated user script on
the very first solve.

Five occurrences existed in NGSolve generators before
2026-06-01:
  src/backends/ngsolve/generators/hyperelasticity.py:38
  src/backends/ngsolve/generators/hyperelasticity.py:88
  src/backends/ngsolve/generators/hyperelasticity.py:102
  src/backends/ngsolve/generators/advanced.py:654
  src/backends/ngsolve/generators/advanced.py:738
All five rewritten to use `maxit` in the same commit.

This fixture verifies both ends of the invariant:
  * Newton(maxit=20)  → no TypeError, returns (iters, conv).
  * Newton(maxits=20) → TypeError with the specific kwarg
                        error string.
"""
from __future__ import annotations

import sys

from netgen.geom2d import unit_square
import ngsolve.solvers as solvers
from ngsolve import (
    BilinearForm,
    GridFunction,
    H1,
    Mesh,
    dx,
)


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))
    fes = H1(mesh, order=1, dirichlet=".*")
    u, v = fes.TnT()

    # A trivial well-posed bilinear form so Newton can converge.
    a = BilinearForm(fes)
    a += u * v * dx
    gfu = GridFunction(fes)

    # (1) Correct kwarg: maxit
    try:
        result = solvers.Newton(a, gfu, maxit=20, printing=False)
        n_iters = result[0] if isinstance(result, tuple) else 0
        print(f"newton_maxit_ok=True")
        print(f"newton_iters={n_iters}")
    except TypeError as exc:
        print(f"newton_maxit_ok=False: {exc}", file=sys.stderr)
        return 2  # FAIL — expected pass

    # (2) Wrong kwarg: maxits → TypeError
    gfu2 = GridFunction(fes)
    caught = False
    msg = ""
    try:
        solvers.Newton(a, gfu2, maxits=20, printing=False)
    except TypeError as exc:
        msg = str(exc)
        caught = "unexpected keyword argument 'maxits'" in msg
    print(f"newton_maxits_raises_typeerror={caught}")
    print(f"newton_maxits_diag_has_maxits={'maxits' in msg}")

    if caught:
        return 0
    print("FAIL: maxits did NOT raise the expected TypeError",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
