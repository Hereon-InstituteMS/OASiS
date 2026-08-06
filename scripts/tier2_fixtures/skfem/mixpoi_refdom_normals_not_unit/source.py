"""Tier-2: Refdom.normals are not unit length on slanted reference facets.

Claim: skfem mixed_poisson#6 -- RefTri.normals[1] = [1, 1] (norm sqrt(2)),
RefTet.normals[3] = [1, 1, 1] (norm sqrt(3)), RefWedge.normals[1] = [1, 1, 0]
(norm sqrt(2)); RefLine/RefQuad/RefHex are all unit; RefWedge.brefdom is None so
FacetBasis on wedges is unsupported. Indexing these arrays directly for a flux
integral gives an O(1) scale error on diagonal facets.

Wrong variant: assuming all reference normals are unit and using them as-is.
Right variant: FacetBasis's own normals, which ARE unit -- and the claim gets
that attribute's NAME wrong: it is `normals`, not `normal`.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import ElementTriRT0, FacetBasis, MeshTri
from skfem.refdom import RefHex, RefLine, RefQuad, RefTet, RefTri, RefWedge

SQRT2 = float(np.sqrt(2.0))
SQRT3 = float(np.sqrt(3.0))


def norms(refdom) -> np.ndarray:
    return np.linalg.norm(np.array(refdom.normals), axis=1)


def main() -> int:
    ok = True
    tri, tet, wedge = norms(RefTri), norms(RefTet), norms(RefWedge)
    print(f"reftri_norms={np.round(tri, 6).tolist()}")
    print(f"reftet_norms={np.round(tet, 6).tolist()}")
    print(f"refwedge_norms={np.round(wedge, 6).tolist()}")

    checks = {
        "reftri_hypotenuse_norm_is_sqrt2": abs(tri[1] - SQRT2) < 1e-12,
        "reftet_slanted_norm_is_sqrt3": abs(tet[3] - SQRT3) < 1e-12,
        "refwedge_diagonal_norm_is_sqrt2": abs(wedge[1] - SQRT2) < 1e-12,
        "line_quad_hex_all_unit": all(
            np.allclose(norms(r), 1.0) for r in (RefLine, RefQuad, RefHex)),
        "refwedge_brefdom_is_none": RefWedge.brefdom is None,
    }
    for key, value in checks.items():
        print(f"{key}={value}")
        if not value:
            print(f"FAIL: {key} does not hold", file=sys.stderr)
            ok = False

    # An assumption of all-unit normals is wrong for tri and tet.
    print(f"reftri_all_unit_assumption_holds={bool(np.allclose(tri, 1.0))}")
    print(f"reftet_all_unit_assumption_holds={bool(np.allclose(tet, 1.0))}")
    if np.allclose(tri, 1.0) or np.allclose(tet, 1.0):
        print("FAIL: the reference normals are now all unit, so the claim is "
              "stale", file=sys.stderr)
        ok = False

    # --- the claim's workaround, with the attribute name corrected -----
    m = MeshTri().refined(2).with_boundaries(
        {"right": lambda x: x[0] > 1.0 - 1e-10})
    fb = FacetBasis(m, ElementTriRT0(), facets=m.boundaries["right"])
    print(f"facetbasis_has_normal_singular={hasattr(fb, 'normal')}")
    print(f"facetbasis_has_normals_plural={hasattr(fb, 'normals')}")
    unit = bool(np.allclose(
        np.linalg.norm(np.asarray(fb.normals), axis=0), 1.0))
    print(f"facetbasis_normals_are_unit={unit}")
    if hasattr(fb, "normal") or not unit:
        print(f"FAIL: FacetBasis.normal exists ({hasattr(fb, 'normal')}) or "
              f"normals are not unit ({unit})", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
