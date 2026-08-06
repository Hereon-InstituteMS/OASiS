"""Tier-2: MeshQuad.to_simplex() was renamed to MeshQuad.to_meshtri().

Claim: skfem hyperelasticity#7 -- skfem 12 renamed MeshQuad.to_simplex() to
MeshQuad.to_meshtri(); legacy templates calling .to_simplex() raise
AttributeError "'MeshQuad1' object has no attribute 'to_simplex'"; the
returned MeshTri splits each quad into two triangles.  Signal:
hasattr(..., 'to_meshtri') is True and hasattr(..., 'to_simplex') is False.

Measured on skfem 12.0.1: exactly as claimed.  The AttributeError message is
reproduced verbatim including the concrete class name MeshQuad1 (not
"MeshQuad"), the element count doubles under to_meshtri, the vertex set is
preserved unchanged, and the returned object is a MeshTri1.  A Poisson
stiffness matrix assembles on the converted mesh, so the conversion is not
merely structural.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology site
-- the legacy `mq.to_simplex()` call becomes `mq.to_meshtri()`, the exact edit
this entry tells a user to make.  The call then succeeds, so no AttributeError
is captured and to_simplex_raises_AttributeError=True, the verbatim message
"object has no attribute 'to_simplex'" and message_names_MeshQuad1=True all
disappear.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import skfem
from skfem import Basis, ElementQuad1, ElementTriP1, MeshQuad, MeshTri
from skfem.models.poisson import laplace

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    ok = True
    print(f"skfem_version={skfem.__version__}")

    mq = MeshQuad.init_tensor(np.linspace(0.0, 1.0, 5),
                              np.linspace(0.0, 1.0, 5))
    print(f"quad_class={type(mq).__name__}")
    print(f"quad_elements={mq.t.shape[1]} quad_vertices={mq.p.shape[1]}")

    # --- the two hasattr probes the claim names -----------------------
    has_new = hasattr(mq, "to_meshtri")
    has_old = hasattr(mq, "to_simplex")
    print(f"hasattr_to_meshtri={has_new}")
    print(f"hasattr_to_simplex={has_old}")
    if not has_new or has_old:
        print("FAIL: the rename is not what the claim describes",
              file=sys.stderr)
        ok = False

    # --- the legacy call really raises --------------------------------
    err = None
    try:
        # PATHOLOGY: the legacy spelling.  T2_MUTATE=1 applies the documented
        # rename fix (.to_simplex -> .to_meshtri) at this call site.
        mq.to_meshtri() if MUTATE else mq.to_simplex()
    except AttributeError as e:
        err = e
    print(f"to_simplex_raises_AttributeError={isinstance(err, AttributeError)}")
    print(f"to_simplex_message={str(err)!r}")
    print(f"message_names_MeshQuad1={'MeshQuad1' in str(err)}")
    if err is None:
        print("FAIL: to_simplex() did not raise", file=sys.stderr)
        ok = False

    # --- what the modern call returns ---------------------------------
    mt = mq.to_meshtri()
    print(f"to_meshtri_class={type(mt).__name__}")
    print(f"to_meshtri_is_MeshTri={isinstance(mt, MeshTri)}")
    print(f"tri_elements={mt.t.shape[1]} tri_vertices={mt.p.shape[1]}")
    doubled = mt.t.shape[1] == 2 * mq.t.shape[1]
    print(f"each_quad_split_into_two_triangles={doubled}")
    print(f"vertices_preserved={mt.p.shape[1] == mq.p.shape[1]}")
    if not doubled or not isinstance(mt, MeshTri):
        print("FAIL: to_meshtri did not split each quad into two triangles",
              file=sys.stderr)
        ok = False

    # --- and the converted mesh is usable -----------------------------
    Kq = laplace.assemble(Basis(mq, ElementQuad1()))
    Kt = laplace.assemble(Basis(mt, ElementTriP1()))
    print(f"quad_stiffness_shape={Kq.shape} nnz={Kq.nnz}")
    print(f"tri_stiffness_shape={Kt.shape} nnz={Kt.nnz}")
    print(f"converted_mesh_assembles={Kt.shape == (mt.p.shape[1],) * 2}")
    # The split loses the quad's two-diagonal coupling, so the triangulated
    # mesh is SPARSER on the same vertex set even though it has twice the
    # elements -- a clean structural fingerprint that the conversion really
    # happened rather than relabelling the same connectivity.
    print(f"same_vertex_count_but_sparser={Kt.shape == Kq.shape and Kt.nnz < Kq.nnz}")
    if Kt.nnz == 0:
        print("FAIL: the converted mesh did not assemble", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
