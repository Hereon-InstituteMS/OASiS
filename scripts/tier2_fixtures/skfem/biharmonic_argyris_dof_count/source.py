"""Tier-2: ElementTriArgyris has 21 DOFs and admits a biharmonic form.

Claim: skfem biharmonic#1 — ElementTriArgyris is C^1 with 21 DOFs per triangle
(6 per vertex x 3 vertices = 18, plus 3 edge-midpoint normal derivatives).

Wrong variant: reaching for a C^0 Lagrange element (ElementTriP1 / ElementTriP2)
in a biharmonic form ddot(u.hess, v.hess). Second derivatives of a C^0 basis are
not usable there and skfem 12.0.1 does not silently return zero -- numpy.einsum
inside skfem.helpers.ddot raises

    ValueError: einstein sum subscripts string contains too many subscripts
                for operand 0

Right variant: ElementTriArgyris (21 DOFs) and ElementTriMorley (6 DOFs) both
assemble a non-trivial bilaplacian.
"""
from __future__ import annotations

import sys

from skfem import (
    Basis,
    BilinearForm,
    ElementTriArgyris,
    ElementTriMorley,
    ElementTriP1,
    ElementTriP2,
    MeshTri,
)
from skfem.helpers import ddot


@BilinearForm
def bilaplace(u, v, w):
    return ddot(u.hess, v.hess)


def main() -> int:
    ok = True
    m = MeshTri().refined(2)

    counts = {}
    for label, element in (("argyris", ElementTriArgyris()),
                           ("morley", ElementTriMorley()),
                           ("p1", ElementTriP1()),
                           ("p2", ElementTriP2())):
        basis = Basis(m, element)
        counts[label] = basis.Nbfun
        print(f"{label}_nbfun={basis.Nbfun}")

    if counts["argyris"] != 21:
        print(f"FAIL: Argyris Nbfun is {counts['argyris']}, claim says 21",
              file=sys.stderr)
        ok = False

    # --- WRONG variant: C^0 element in a biharmonic form ----------------
    for label, element in (("p1", ElementTriP1()), ("p2", ElementTriP2())):
        basis = Basis(m, element)
        raised = ""
        try:
            bilaplace.assemble(basis)
        except ValueError as exc:
            raised = str(exc)
        print(f"{label}_bilaplace_raises_valueerror={bool(raised)}")
        print(f"{label}_bilaplace_msg={raised!r}")
        if "too many subscripts" not in raised:
            print(f"FAIL: {label} biharmonic assembly did not raise the "
                  f"einsum subscript error; got {raised!r}", file=sys.stderr)
            ok = False

    # --- RIGHT variant: the C^1 elements assemble -----------------------
    for label, element in (("argyris", ElementTriArgyris()),
                           ("morley", ElementTriMorley())):
        basis = Basis(m, element)
        K = bilaplace.assemble(basis)
        good = K.nnz > 0 and abs(K).max() > 0.0
        print(f"{label}_bilaplace_assembles={good}")
        print(f"{label}_bilaplace_nnz_positive={K.nnz > 0}")
        if not good:
            print(f"FAIL: {label} bilaplacian assembled to an empty/zero matrix",
                  file=sys.stderr)
            ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
