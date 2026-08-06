"""Tier-2: a DG facet form on a plain Basis -- what skfem 12.0.1 ACTUALLY says.

Claim: skfem dg_methods#1 -- InteriorFacetBasis assembles over interior mesh
facets, needed for DG flux terms.  Using a plain Basis with an
InteriorFacetBasis-style form (involving u and u.other or jump terms) "raises
`AttributeError: 'Basis' object has no attribute 'normals'` or yields a matrix
with no facet-coupling entries".

PARTIAL FALSIFICATION (measured here, skfem 12.0.1):
  * an AttributeError IS raised, but the message is
        "Attribute 'n' not found in 'w'."
    It is raised by FormExtraParams (the ``w`` dict), NOT by Basis, and the
    word "normals" never appears.  The catalog's quoted wording is wrong; this
    fixture pins the real wording and asserts the claimed one is absent.
  * the second half of the claim IS true: a plain Basis (CellBasis) assembly
    of the same volume form produces a matrix with ZERO entries coupling DOFs
    of different elements, while the InteriorFacetBasis side=0/side=1 pair
    produces 704 such entries on MeshTri().refined(3).
  * ``u.other`` does not exist either: touching it raises
        "'DiscreteField' object has no attribute 'other'".

Wrong variant: assemble a facet form that reads ``w.n`` over a plain
``Basis``, and assemble a jump-style form over a plain Basis, then count
inter-element coupling.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import (
    Basis,
    BilinearForm,
    ElementDG,
    ElementTriP1,
    InteriorFacetBasis,
    MeshTri,
    asm,
)

B = np.array([1.0, 0.5])

CLAIMED_MSG = "'Basis' object has no attribute 'normals'"


@BilinearForm
def facet_form_using_normals(u, v, w):
    """A DG flux form: it needs the facet normal w.n."""
    bn = B[0] * w.n[0] + B[1] * w.n[1]
    return bn * u * v


@BilinearForm
def volume_form(u, v, w):
    return (B[0] * u.grad[0] + B[1] * u.grad[1]) * v


@BilinearForm
def upwind_flux(u, v, w):
    bn = B[0] * w.n[0] + B[1] * w.n[1]
    sv = (-1.0) ** w.idx[1]
    ju = (-1.0) ** w.idx[0] * u
    return np.minimum(sv * bn, 0.0) * (-sv) * ju * v


@BilinearForm
def form_touching_other(u, v, w):
    return (u - u.other) * v


def count_interelement(M, dof2el):
    C = M.tocoo()
    return int(np.sum((dof2el[C.row] != dof2el[C.col])
                      & (np.abs(C.data) > 1e-14)))


def main() -> int:
    ok = True
    m = MeshTri().refined(3)
    e = ElementDG(ElementTriP1())
    ib = Basis(m, e)                       # plain CellBasis -- the wrong choice
    i0 = InteriorFacetBasis(m, e, side=0)
    i1 = InteriorFacetBasis(m, e, side=1)

    dof2el = np.empty(ib.N, dtype=int)
    for k in range(m.nelements):
        dof2el[ib.element_dofs[:, k]] = k

    # --- WRONG variant 1: facet form (needs w.n) on a plain Basis ----------
    try:
        asm(facet_form_using_normals, ib)
    except AttributeError as exc:
        msg = str(exc)
        kind = type(exc).__name__
    else:
        msg, kind = "", ""
    print(f"plain_basis_raises_attributeerror={kind == 'AttributeError'}")
    print(f"real_message={msg!r}")
    print(f"claimed_normals_wording_present={CLAIMED_MSG in msg}")
    print(f"word_normals_in_message={'normals' in msg}")
    if kind != "AttributeError":
        print("FAIL: a plain Basis accepted a form that reads w.n", file=sys.stderr)
        ok = False
    if "Attribute 'n' not found in 'w'." not in msg:
        print("FAIL: the observed AttributeError wording changed", file=sys.stderr)
        ok = False
    if CLAIMED_MSG in msg:
        print("FAIL: the catalog wording now appears -- update the falsification",
              file=sys.stderr)
        ok = False

    # --- the same for u.other, which the catalog also cites ---------------
    try:
        asm(form_touching_other, [i0, i1], [i0, i1])
    except AttributeError as exc:
        other_msg = str(exc)
    else:
        other_msg = ""
    print(f"u_dot_other_raises={bool(other_msg)}")
    print(f"u_dot_other_message={other_msg!r}")
    if "'DiscreteField' object has no attribute 'other'" not in other_msg:
        print("FAIL: u.other no longer raises the observed AttributeError",
              file=sys.stderr)
        ok = False

    # --- WRONG variant 2: no facet coupling from a plain Basis -------------
    A_plain = asm(volume_form, ib)
    n_plain = count_interelement(A_plain, dof2el)
    print(f"plain_basis_nnz={A_plain.nnz}")
    print(f"plain_basis_interelement_entries={n_plain}")
    if n_plain != 0:
        print("FAIL: a plain Basis assembly produced inter-element coupling",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant: InteriorFacetBasis pair DOES couple elements -------
    F = asm(upwind_flux, [i0, i1], [i0, i1])
    n_facet = count_interelement(F, dof2el)
    print(f"interior_facet_basis_interelement_entries={n_facet}")
    print(f"interior_facet_basis_couples_elements={n_facet > 0}")
    print(f"n_interior_facets={int(np.sum(m.f2t[1] != -1))}")
    print(f"interior_facet_basis_nelems={i0.nelems}")
    if n_facet <= 0:
        print("FAIL: InteriorFacetBasis produced no inter-element coupling",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
