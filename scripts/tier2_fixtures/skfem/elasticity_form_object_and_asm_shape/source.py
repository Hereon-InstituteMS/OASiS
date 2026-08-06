"""Tier-2: linear_elasticity(lam, mu) is a BilinearForm, ready for asm().

Claim: skfem linear_elasticity#2 (previously "claim inherited -- not yet
empirically separated") -- linear_elasticity(lam, mu) returns a BilinearForm
directly; pass it to skfem.asm(form, basis) and get an (basis.N, basis.N) sparse
matrix.

Wrong variant (a): treating the return value as an INTEGRAND and calling it with
(u, v, w) as if it were the decorated kernel.
Wrong variant (b): assembling it against a SCALAR basis. Elasticity needs a
vector space; skfem does not silently produce a scalar operator.
"""
from __future__ import annotations

import sys

import scipy.sparse as sp
from skfem import (
    Basis,
    ElementVector,
    ElementQuad1,
    MeshQuad,
    asm,
)
from skfem.assembly.form import BilinearForm
from skfem.models.elasticity import lame_parameters, linear_elasticity


def main() -> int:
    ok = True
    lam, mu = lame_parameters(210e9, 0.3)
    form = linear_elasticity(lam, mu)

    is_form = type(form) is BilinearForm
    print(f"is_bilinearform_instance={is_form}")
    print(f"form_type_name={type(form).__name__}")
    if not is_form:
        print(f"FAIL: linear_elasticity returned {type(form)!r}, not a "
              f"BilinearForm", file=sys.stderr)
        ok = False

    m = MeshQuad().refined(2)
    vector_basis = Basis(m, ElementVector(ElementQuad1()))
    K = asm(form, vector_basis)
    shape_ok = K.shape == (vector_basis.N, vector_basis.N)
    sparse_ok = sp.issparse(K)
    print(f"basis_N={vector_basis.N}")
    print(f"asm_shape_equals_basis_N={shape_ok}")
    print(f"asm_returns_scipy_sparse={sparse_ok}")
    if not (shape_ok and sparse_ok):
        print(f"FAIL: asm returned {type(K)!r} of shape {K.shape!r} against "
              f"basis.N={vector_basis.N}", file=sys.stderr)
        ok = False

    # --- WRONG variant (a): call the form as if it were the integrand ---
    raised_call = ""
    try:
        form(1.0, 1.0, None)
    except Exception as exc:
        raised_call = f"{type(exc).__name__}"
    print(f"calling_form_as_integrand_raises={bool(raised_call)}")
    print(f"calling_form_as_integrand_exc={raised_call!r}")
    if not raised_call:
        print("FAIL: calling the BilinearForm as an integrand succeeded",
              file=sys.stderr)
        ok = False

    # --- WRONG variant (b): scalar basis ---------------------------------
    raised_scalar = ""
    try:
        asm(form, Basis(m, ElementQuad1()))
    except Exception as exc:
        raised_scalar = f"{type(exc).__name__}: {exc}"
    print(f"scalar_basis_raises={bool(raised_scalar)}")
    print(f"scalar_basis_msg={raised_scalar[:160]!r}")
    if not raised_scalar:
        print("FAIL: the elasticity form assembled against a SCALAR basis "
              "without complaint", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
