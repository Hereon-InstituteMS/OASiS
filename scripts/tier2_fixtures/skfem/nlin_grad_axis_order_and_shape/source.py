"""Tier-2: .grad is (spatial_dim, n_elements, n_quadrature_points).

Claim: skfem nonlinear#3 -- basis.interpolate(u).grad has shape (spatial_dim,
n_elements, n_quad) in 2-D; .grad[0] is the x-component over all elements and
quadrature points; for MeshTri().refined(2) with P1 that is (2, 32, 3) and all
values are zero for a constant function.

Wrong variant: reading .grad[:, 0, :] as "the gradient of element 0", the axis
confusion this claim exists to prevent. The fixture shows the two slices have
different shapes, so the mistake is detectable.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri


def main() -> int:
    ok = True
    m = MeshTri().refined(2)
    basis = Basis(m, ElementTriP1())
    n_elements = m.t.shape[1]
    field = basis.interpolate(np.full(basis.N, 7.0))
    g = np.asarray(field.grad)
    print(f"grad_shape={g.shape}")
    print(f"n_elements={n_elements}")
    print(f"grad_axis0_is_spatial_dim={g.shape[0] == m.dim()}")
    print(f"grad_axis1_is_n_elements={g.shape[1] == n_elements}")
    if g.shape[0] != m.dim() or g.shape[1] != n_elements:
        print(f"FAIL: grad shape {g.shape!r} is not "
              f"(dim, n_elements, n_qp) with dim={m.dim()}, "
              f"n_elements={n_elements}", file=sys.stderr)
        ok = False

    print(f"constant_grad_all_zero={bool(np.allclose(g, 0.0))}")
    if not np.allclose(g, 0.0):
        print("FAIL: a constant field has non-zero grad", file=sys.stderr)
        ok = False

    # --- RIGHT variant: component slice --------------------------------
    comp = g[0]
    right_shape = comp.shape == (n_elements, g.shape[2])
    print(f"component_slice_shape={comp.shape}")
    print(f"component_slice_shape_is_elements_by_qp={right_shape}")
    if not right_shape:
        print(f"FAIL: .grad[0] has shape {comp.shape!r}", file=sys.stderr)
        ok = False

    # --- WRONG variant: transposed read -------------------------------
    per_element = g[:, 0, :]
    differs = per_element.shape != comp.shape
    print(f"transposed_read_shape={per_element.shape}")
    print(f"transposed_read_has_wrong_shape={differs}")
    if not differs:
        print(f"FAIL: .grad[:, 0, :] and .grad[0] have the same shape "
              f"{comp.shape!r}, so the axis confusion is undetectable",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
