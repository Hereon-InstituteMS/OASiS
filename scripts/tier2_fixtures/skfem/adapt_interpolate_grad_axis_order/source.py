"""Tier-2: Basis.interpolate(u).grad axis order, and the silent wrong slice.

Claim: skfem adaptive_poisson#2 -- Basis.interpolate(u).grad returns shape
(spatial_dim, n_elements, n_qpoints); for P1 the gradient is constant per
element so grad_u[:, :, 0] extracts a per-element gradient vector cheaply.
"Confusing the axis order (e.g. grad_u[:, 0, :] for 'gradient of element 0')
produces silently-wrong jump terms.  Signal: ... ValueError 'operands could
not be broadcast together' if you try to dot the wrong-shape tensor with the
facet normal."

Measured on skfem 12.0.1, MeshTri().refined(2) (32 elements, 3 quadrature
points per element), ElementTriP1:

  * the axis order is confirmed: grad has shape (2, 32, 3), grad[:, :, 0] is
    (2, 32) and grad[:, 0, :] is (2, 3).
  * P1 constancy is confirmed exactly: every quadrature point in an element
    carries the same gradient to machine precision, so the [:, :, 0] slice
    loses nothing.
  * "SILENTLY-WRONG" IS RIGHT; THE ValueError IS NOT.  Contracting the
    wrong slice against a 2-vector normal does NOT raise -- both slices have
    the spatial dimension first, so the contraction is legal either way and
    returns a length-3 array (one per quadrature point) instead of the
    length-32 array (one per element) that was intended.  No broadcast error
    appears, because nothing is broadcast wrongly; the WRONG AXIS is simply
    contracted correctly.
  * a broadcast ValueError only appears in the unrelated case of contracting
    against an array whose length is neither 2 nor a trailing-axis length,
    so a try/except around the dot is not a guard.  The guard is a SHAPE
    assertion: the result must be n_elements long.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri


def main() -> int:
    ok = True
    m = MeshTri().refined(2)
    ib = Basis(m, ElementTriP1())
    u = ib.project(lambda x: x[0] ** 2 + 3.0 * x[1])
    g = np.asarray(ib.interpolate(u).grad)
    ne = m.t.shape[1]
    print(f"elements={ne} dofs_N={ib.N}")
    print(f"grad_shape={g.shape}")
    print(f"grad_is_dim_by_nelem_by_nqp="
          f"{g.shape[0] == 2 and g.shape[1] == ne}")
    nqp = g.shape[2]
    print(f"n_quadrature_points={nqp}")
    print(f"nqp_differs_from_nelem={nqp != ne}")
    if g.shape[0] != 2 or g.shape[1] != ne:
        print("FAIL: grad is not (dim, n_elements, n_qpoints)",
              file=sys.stderr)
        ok = False

    per_elem = g[:, :, 0]
    per_qp_of_elem0 = g[:, 0, :]
    print(f"slice_colon_colon_zero_shape={per_elem.shape}")
    print(f"slice_colon_zero_colon_shape={per_qp_of_elem0.shape}")
    spread = float(np.abs(g - g[:, :, [0]]).max())
    print(f"p1_gradient_variation_within_element={spread:.3e}")
    print(f"p1_gradient_is_constant_per_element={spread < 1e-13}")
    if spread >= 1e-13:
        print("FAIL: the P1 gradient varied inside an element",
              file=sys.stderr)
        ok = False

    # --- contracting each slice with a 2-vector normal -------------------
    n = np.array([0.6, 0.8])
    err_right = err_wrong = None
    try:
        right = np.einsum("i...,i->...", per_elem, n)
    except ValueError as e:
        err_right = e
        right = None
    try:
        wrong = np.einsum("i...,i->...", per_qp_of_elem0, n)
    except ValueError as e:
        err_wrong = e
        wrong = None
    print(f"correct_slice_raises={err_right is not None}")
    print(f"wrong_slice_raises={err_wrong is not None}")
    print(f"correct_contraction_shape={np.shape(right)}")
    print(f"wrong_contraction_shape={np.shape(wrong)}")
    print(f"correct_result_is_one_per_element={np.shape(right) == (ne,)}")
    print(f"wrong_result_is_one_per_quadrature_point="
          f"{np.shape(wrong) == (nqp,)}")
    print(f"wrong_slice_is_silently_wrong="
          f"{err_wrong is None and np.shape(wrong) != (ne,)}")
    if err_wrong is not None:
        print(f"FAIL: the wrong slice DID raise {type(err_wrong).__name__}, "
              f"so the quoted ValueError fires after all", file=sys.stderr)
        ok = False
    if np.shape(right) != (ne,):
        print("FAIL: the correct slice did not give one value per element",
              file=sys.stderr)
        ok = False

    # --- where a broadcast ValueError actually comes from ----------------
    bad = None
    try:
        np.einsum("i...,i->...", per_elem, np.array([1.0, 2.0, 3.0]))
    except ValueError as e:
        bad = e
    print(f"mismatched_length_raises={bad is not None}")
    print(f"mismatched_length_message={str(bad)[:70]!r}")
    print(f"a_try_except_only_catches_the_unrelated_case="
          f"{bad is not None and err_wrong is None}")
    if bad is None:
        print("FAIL: a genuinely mismatched contraction did not raise",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
