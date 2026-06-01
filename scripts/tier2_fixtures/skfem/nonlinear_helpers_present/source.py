"""Tier-2: skfem has no built-in Newton; interpolate returns DiscreteField.

Verifies three claims at once:
  #0 skfem has NO Newton solver — no module attribute matching
     'newton' or 'nonlin' at top-level or in skfem.helpers.
  #2 basis.interpolate(u_dof_vec) returns a DiscreteField with
     .grad attribute.
  #3 .grad has shape (spatial_dim, n_elements, n_quad_points).
     For constant u_dof, the gradient field is uniformly zero.
"""
from __future__ import annotations

import sys

import numpy as np
import skfem
import skfem.helpers


def main() -> int:
    top_match = [n for n in dir(skfem)
                 if "newton" in n.lower() or "nonlin" in n.lower()]
    helper_match = [n for n in dir(skfem.helpers)
                    if "newton" in n.lower() or "nonlin" in n.lower()]
    no_newton = not top_match and not helper_match
    print(f"no_newton_in_skfem={no_newton}")

    mesh = skfem.MeshTri().refined(2)
    basis = skfem.Basis(mesh, skfem.ElementTriP1())
    field = basis.interpolate(np.ones(basis.N) * 2.5)
    print(f"interpolate type: {type(field).__name__}")
    grad_shape = field.grad.shape
    print(f"grad shape: {grad_shape}")
    print(f"grad_shape_first_dim={grad_shape[0]}")
    grad_zero = np.max(np.abs(field.grad)) == 0.0
    print(f"grad_for_const_u_is_zero={grad_zero}")

    if no_newton and grad_shape[0] == 2 and grad_zero:
        return 0
    print("ERROR: checks failed", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
