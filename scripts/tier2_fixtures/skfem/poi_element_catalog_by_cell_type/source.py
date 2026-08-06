"""Tier-2: the element catalog is cell-type specific and skfem enforces it.

Claim: skfem poisson#5 -- ElementTriP1/P2/P3 for triangles, ElementQuad1/Quad2
for quads, ElementTetP1/P2 for tets, ElementHex1/Hex2 for hexes, plus
ElementTriRT0 / ElementTriMini / ElementTetMini for mixed methods; and
type(ElementTriP1()).__bases__ shows the Element hierarchy.

Wrong variant: pairing an element with a mesh of a different cell type
(ElementTriP1 on a MeshQuad, ElementQuad1 on a MeshTri). skfem 12.0.1 raises
ValueError('Incompatible Mesh and Element.') at Basis construction.
"""
from __future__ import annotations

import sys

import skfem
from skfem import Basis, ElementQuad1, ElementTriP1, MeshQuad, MeshTri
from skfem.element import Element

NAMES = (
    "ElementTriP1", "ElementTriP2", "ElementTriP3",
    "ElementQuad1", "ElementQuad2",
    "ElementTetP1", "ElementTetP2",
    "ElementHex1", "ElementHex2",
    "ElementTriRT0", "ElementTriMini", "ElementTetMini",
)


def main() -> int:
    ok = True
    missing = [n for n in NAMES if not hasattr(skfem, n)]
    print(f"all_catalog_element_names_resolve={not missing}")
    print(f"missing_names={missing!r}")
    if missing:
        print(f"FAIL: the catalog names {missing!r} do not exist on skfem "
              f"{skfem.__version__}", file=sys.stderr)
        ok = False

    element = ElementTriP1()
    bases = [c.__name__ for c in type(element).__bases__]
    print(f"trip1_bases={bases!r}")
    print(f"trip1_is_element_instance={isinstance(element, Element)}")
    if not isinstance(element, Element):
        print("FAIL: ElementTriP1 is not an Element instance", file=sys.stderr)
        ok = False

    # --- WRONG variant: element / cell-type mismatch --------------------
    for label, mesh, el in (("trip1_on_meshquad", MeshQuad().refined(1),
                             ElementTriP1()),
                            ("quad1_on_meshtri", MeshTri().refined(1),
                             ElementQuad1())):
        raised = ""
        try:
            Basis(mesh, el)
        except ValueError as exc:
            raised = str(exc)
        print(f"{label}_raises={bool(raised)}")
        print(f"{label}_msg={raised!r}")
        if "Incompatible Mesh and Element" not in raised:
            print(f"FAIL: {label} did not raise the incompatibility error; "
                  f"got {raised!r}", file=sys.stderr)
            ok = False

    # --- RIGHT variant --------------------------------------------------
    print(f"trip1_on_meshtri_nbfun={Basis(MeshTri().refined(1), ElementTriP1()).Nbfun}")
    print(f"quad1_on_meshquad_nbfun={Basis(MeshQuad().refined(1), ElementQuad1()).Nbfun}")

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
