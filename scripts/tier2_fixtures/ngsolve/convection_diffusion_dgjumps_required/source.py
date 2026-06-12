"""Tier-2: NGSolve L2(dgjumps=True) is required for DG skeleton integrals.

Catalog claim under audit (NGSolve convection_diffusion pitfall #0):
  'MUST set dgjumps=True on L2 space to allocate coupling entries'

Empirical observation:
  * fes = L2(mesh, order=2)                        # default dgjumps=False
  * fes_dg = L2(mesh, order=2, dgjumps=True)

Both spaces have the SAME ndof on the same mesh (the default vs
dgjumps L2 spaces have identical DOF layouts). What differs is the
sparse-matrix coupling pattern allocated when a BilinearForm is
assembled with an interior-facet integral.

When BilinearForm.Assemble() is called with a dx(skeleton=True)
integral on a default-L2 space, NGSolve raises

    NgException: SparseMatrixTM::AddElementMatrix: illegal dnums
    in Assemble BilinearForm 'biform_from_py'

because the sparsity pattern allocated for the default L2 space
has no slots for inter-element coupling DOFs. With dgjumps=True
the assembly succeeds and produces a non-trivial sparse matrix.

This fixture asserts both ends:
  * Default L2 + skeleton integral → NgException with the literal
    'illegal dnums' string.
  * dgjumps=True L2 + same integral → assembly succeeds, matrix
    has nze > 0.
  * Both spaces have the same ndof.
"""
from __future__ import annotations

import sys

from netgen.geom2d import unit_square
from ngsolve import (
    BilinearForm,
    L2,
    Mesh,
    dx,
)


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.5))

    fes_default = L2(mesh, order=2)
    fes_dg = L2(mesh, order=2, dgjumps=True)
    print(f"ndof_default={fes_default.ndof}")
    print(f"ndof_dgjumps={fes_dg.ndof}")
    print(f"ndof_equal={fes_default.ndof == fes_dg.ndof}")

    # Without dgjumps: skeleton integral fails at Assemble.
    u, v = fes_default.TnT()
    a_no = BilinearForm(fes_default)
    a_no += u * v * dx
    a_no += (u - u.Other()) * (v - v.Other()) * dx(skeleton=True)
    raised_no = False
    msg = ""
    try:
        a_no.Assemble()
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        raised_no = "illegal dnums" in msg
    print(f"default_assemble_raises={raised_no}")
    print(f"default_diag_has_illegal_dnums={'illegal dnums' in msg}")

    # With dgjumps: skeleton integral assembles.
    u2, v2 = fes_dg.TnT()
    a_yes = BilinearForm(fes_dg)
    a_yes += u2 * v2 * dx
    a_yes += (u2 - u2.Other()) * (v2 - v2.Other()) * dx(skeleton=True)
    try:
        a_yes.Assemble()
        nze = a_yes.mat.nze
        ok_yes = nze > 0
        print(f"dgjumps_assemble_ok=True")
        print(f"dgjumps_nze_positive={ok_yes}")
    except Exception as exc:  # noqa: BLE001
        print(f"dgjumps_assemble_ok=False: {exc}",
              file=sys.stderr)
        return 2

    ok = (raised_no
          and fes_default.ndof == fes_dg.ndof
          and ok_yes)
    if ok:
        return 0
    print("FAIL: dgjumps invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
