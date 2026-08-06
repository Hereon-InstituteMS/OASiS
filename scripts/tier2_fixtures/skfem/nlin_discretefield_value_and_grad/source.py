"""Tier-2: interpolate() gives a DiscreteField with .value and .grad.

Claim: skfem nonlinear#2 -- basis.interpolate(u) returns a DiscreteField with
.value (at quadrature points) and .grad; for constant u the .grad array is
uniformly zero.

Wrong variant: referring to a field name inside a form that was never passed to
assemble. Right variant: pass it and read .value / .grad.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import Basis, ElementTriP1, LinearForm, MeshTri


@LinearForm
def needs_field(v, w):
    return w["u"].value * v


def main() -> int:
    ok = True
    basis = Basis(MeshTri().refined(2), ElementTriP1())

    field = basis.interpolate(np.full(basis.N, 2.5))
    print(f"interpolate_type_name={type(field).__name__}")
    print(f"interpolate_module={type(field).__module__}")
    print(f"has_value_attribute={hasattr(field, 'value')}")
    print(f"has_grad_attribute={hasattr(field, 'grad')}")
    if type(field).__name__ != "DiscreteField" or not (
            hasattr(field, "value") and hasattr(field, "grad")):
        print(f"FAIL: interpolate returned {type(field).__name__} with "
              f"value={hasattr(field, 'value')} grad={hasattr(field, 'grad')}",
              file=sys.stderr)
        ok = False

    zero = bool(np.allclose(np.asarray(field.grad), 0.0))
    print(f"constant_field_grad_all_zero={zero}")
    print(f"constant_field_value_all_2p5="
          f"{bool(np.allclose(np.asarray(field.value), 2.5))}")
    if not zero:
        print(f"FAIL: a constant field has non-zero grad, max "
              f"{np.abs(np.asarray(field.grad)).max()!r}", file=sys.stderr)
        ok = False

    # A linear field's gradient is its slope everywhere -- checks .grad is the
    # real derivative, not an incidental zero array.
    linear = basis.interpolate(basis.project(lambda x: 3.0 * x[0] - x[1]))
    g = np.asarray(linear.grad)
    slope_ok = (np.allclose(g[0], 3.0, atol=1e-10)
                and np.allclose(g[1], -1.0, atol=1e-10))
    print(f"linear_field_grad_matches_slope={slope_ok}")
    if not slope_ok:
        print(f"FAIL: linear field grad is {g[0].mean()!r}, {g[1].mean()!r}, "
              f"expected 3 and -1", file=sys.stderr)
        ok = False

    # --- WRONG variant: never pass the field ----------------------------
    msg = ""
    try:
        needs_field.assemble(basis)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
    print(f"missing_field_raises={bool(msg)}")
    print(f"missing_field_msg={msg[:160]!r}")
    if not msg:
        print("FAIL: assembling a form that reads w['u'] without passing u "
              "succeeded", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
