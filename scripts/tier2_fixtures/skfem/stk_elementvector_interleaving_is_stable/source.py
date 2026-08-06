"""Tier-2: ElementVector DOF ordering is interleaved and does not vary.

Claim: skfem stokes#5 -- corrected 2026-08-06.  The entry previously said
"ElementVector DOF ordering is unreliable for mixed systems ... the actual
layout depends on element type", and recommended building two separate scalar
bases instead.  It was marked inherited and never falsified.

Measured on skfem 12.0.1:

  * THE CLAIM IS FALSE.  The layout is strictly interleaved and IDENTICAL for
    every element tested -- ElementTriP1, ElementTriP2, ElementTriP3 and the
    tetrahedral P1/P2 -- with component 1 on the even global indices,
    component 2 on the odd ones, and dof(u^2) = dof(u^1) + 1 at every entry.
    basis.doflocs[:, k::dim] reproduces the scalar basis doflocs for each
    component k, and basis.N is exactly dim times the scalar count.
  * IT IS NOT MERELY "documented as interleaved": the fixture checks the
    actual arrays, not the documentation, at three polynomial orders and in
    both 2D and 3D.
  * the recommendation is not wrong, only mis-motivated: two scalar bases do
    give a blocked layout, but that is a layout preference, not a
    correctness fix.
  * THE REAL TRAP IS THE OPPOSITE ONE, and it is silent: slicing a vector
    solution as [:N] and [N:] -- the blocked convention skfem does NOT use --
    returns two arrays of the right length that are NOT the components.  The
    fixture shows the mix-up produces a field that differs from the true
    component while raising nothing.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
from skfem import (
    Basis,
    ElementTetP1,
    ElementTetP2,
    ElementTriP1,
    ElementTriP2,
    ElementTriP3,
    ElementVector,
    MeshTet,
    MeshTri,
)

CASES = (
    ("triP1", MeshTri, ElementTriP1, 2),
    ("triP2", MeshTri, ElementTriP2, 2),
    ("triP3", MeshTri, ElementTriP3, 2),
    ("tetP1", MeshTet, ElementTetP1, 3),
    ("tetP2", MeshTet, ElementTetP2, 3),
)


def main() -> int:
    ok = True
    interleaved_all, blocked_any, offset_all = [], [], []
    for tag, meshcls, elemcls, dim in CASES:
        m = meshcls().refined(2)
        e = elemcls()
        bv = Basis(m, ElementVector(e))
        bs = Basis(m, e)
        print(f"{tag}_dim={dim} vector_N={bv.N} scalar_N={bs.N} "
              f"ratio={bv.N // bs.N}")
        print(f"{tag}_N_is_dim_times_scalar={bv.N == dim * bs.N}")

        interleaved = all(
            np.allclose(bv.doflocs[:, k::dim], bs.doflocs) for k in range(dim))
        blocked = all(
            np.allclose(bv.doflocs[:, k * bs.N:(k + 1) * bs.N], bs.doflocs)
            for k in range(dim))
        interleaved_all.append(interleaved)
        blocked_any.append(blocked)
        print(f"{tag}_layout_is_interleaved={interleaved}")
        print(f"{tag}_layout_is_blocked={blocked}")

        dofs = bv.get_dofs(lambda x: x[0] < 1e-10)
        keys = sorted(dofs.nodal.keys())
        print(f"{tag}_nodal_keys={keys}")
        comp = [np.sort(np.asarray(dofs.nodal[f"u^{k + 1}"]))
                for k in range(dim)]
        consecutive = all(np.array_equal(comp[k], comp[0] + k)
                          for k in range(dim))
        modclean = all(bool((comp[k] % dim == comp[0][0] % dim + k).all()
                            if len(comp[k]) else True) for k in range(dim))
        offset_all.append(consecutive)
        print(f"{tag}_components_are_consecutive={consecutive}")
        print(f"{tag}_component_stride_is_dim={modclean}")
        if not (bv.N == dim * bs.N and interleaved and consecutive):
            print(f"FAIL: {tag} did not have the interleaved layout",
                  file=sys.stderr)
            ok = False

    print(f"interleaved_for_every_element={all(interleaved_all)}")
    print(f"blocked_for_no_element={not any(blocked_any)}")
    print(f"components_consecutive_for_every_element={all(offset_all)}")
    print(f"layout_depends_on_element_type="
          f"{len(set(interleaved_all)) > 1}")
    if not all(interleaved_all):
        print("FAIL: the layout was not interleaved everywhere",
              file=sys.stderr)
        ok = False
    if any(blocked_any):
        print("FAIL: some element used the blocked layout, so the layout "
              "really does vary", file=sys.stderr)
        ok = False

    # --- the silent trap: slicing as if the layout were blocked -----------
    m = MeshTri().refined(3)
    e = ElementTriP2()
    bv = Basis(m, ElementVector(e))
    bs = Basis(m, e)
    u = bv.zeros()
    u[0::2] = np.sin(np.pi * bs.doflocs[0])          # component 1
    u[1::2] = np.cos(np.pi * bs.doflocs[1])          # component 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        blocked_guess = u[:bs.N]
        true_comp = u[0::2]
        msgs = sorted({str(c.message) for c in caught})
    print(f"blocked_slice_length={len(blocked_guess)} "
          f"true_component_length={len(true_comp)}")
    print(f"blocked_slice_has_the_right_length="
          f"{len(blocked_guess) == len(true_comp)}")
    dev = float(np.abs(blocked_guess - true_comp).max())
    print(f"blocked_slice_max_deviation={dev:.4e}")
    print(f"blocked_slice_is_silently_wrong={dev > 1e-6 and not msgs}")
    print(f"blocked_slice_warnings={msgs!r}")
    if dev <= 1e-6:
        print("FAIL: the blocked slice happened to equal the component",
              file=sys.stderr)
        ok = False
    if len(blocked_guess) != len(true_comp):
        print("FAIL: the blocked slice had the wrong length, so a length "
              "check would catch it", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
