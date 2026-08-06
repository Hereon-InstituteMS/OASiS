"""Tier-2: w.n needs a facet basis, and the error names w, not the Basis.

Claim: skfem convection_diffusion#1 -- for DG jump terms use InteriorFacetBasis.
A form that reads the outward normal as w.n assembled against a plain CellBasis
raises AttributeError("Attribute 'n' not found in 'w'."). The message names the
FormExtraParams dict w, not the Basis, so the retired signal
"'Basis' object has no attribute 'normals'" is not a string skfem 12.0.1 emits.

Wrong variant: assemble the normal-reading form on a cell Basis.
Right variants: FacetBasis (boundary facets) and InteriorFacetBasis (the rest),
whose facet counts are checked to partition the mesh's facets exactly.

Mutation control: T2_MUTATE=1 assembles the same normal-reading form on
FacetBasis instead of the plain cell Basis -- the documented fix. Nothing then
raises, so "Attribute 'n' not found in 'w'." and
cellbasis_normal_raises_attributeerror=True vanish and the fixture goes red.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

from skfem import (
    Basis,
    BilinearForm,
    ElementTriP1,
    FacetBasis,
    InteriorFacetBasis,
    MeshTri,
)
from skfem.helpers import dot

MUTATE = os.environ.get("T2_MUTATE") == "1"


@BilinearForm
def uses_normal(u, v, w):
    return dot(w.n, w.n) * u * v


def main() -> int:
    ok = True
    m = MeshTri().refined(2)

    # --- WRONG variant: the normal on a cell basis -----------------------
    msg = ""
    # Pathology: the normal-reading form is assembled on a plain cell Basis.
    # Mutation: the documented fix -- assemble it on a FacetBasis instead.
    wrong_basis = (FacetBasis(m, ElementTriP1()) if MUTATE
                   else Basis(m, ElementTriP1()))
    try:
        uses_normal.assemble(wrong_basis)
    except AttributeError as exc:
        msg = str(exc)
    print(f"cellbasis_normal_raises_attributeerror={bool(msg)}")
    print(f"cellbasis_normal_msg={msg!r}")
    names_w = "'w'" in msg and "Basis" not in msg
    print(f"message_names_w_not_basis={names_w}")
    prior = "'Basis' object has no attribute 'normals'"
    print(f"prior_signal_string_absent={prior not in msg}")
    if "Attribute 'n' not found in 'w'." not in msg or not names_w:
        print(f"FAIL: the cell-basis normal error changed; got {msg!r}",
              file=sys.stderr)
        ok = False

    # --- RIGHT variants ---------------------------------------------------
    n_facets = m.facets.shape[1]
    n_boundary = len(m.boundary_facets())
    facet_basis = FacetBasis(m, ElementTriP1())
    interior_basis = InteriorFacetBasis(m, ElementTriP1())
    n_facet_basis = len(facet_basis.find)
    n_interior = len(interior_basis.find)
    print(f"n_facets={n_facets}")
    print(f"n_boundary_facets={n_boundary}")
    print(f"n_facetbasis_facets={n_facet_basis}")
    print(f"n_interior_facets={n_interior}")
    partitions = (n_facet_basis == n_boundary
                  and n_facet_basis + n_interior == n_facets)
    print(f"facet_plus_interior_equals_total={partitions}")
    if not partitions:
        print(f"FAIL: {n_facet_basis} boundary + {n_interior} interior does not "
              f"partition {n_facets} facets", file=sys.stderr)
        ok = False

    # Both facet bases DO carry the normal.
    for label, b in (("facetbasis", facet_basis),
                     ("interiorfacetbasis", interior_basis)):
        raised = ""
        try:
            uses_normal.assemble(b)
        except Exception as exc:
            raised = f"{type(exc).__name__}: {exc}"
        print(f"{label}_assembles_normal_form={not raised}")
        if raised:
            print(f"FAIL: {label} could not assemble the normal form: {raised}",
                  file=sys.stderr)
            ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
