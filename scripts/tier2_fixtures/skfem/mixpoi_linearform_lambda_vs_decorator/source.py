"""Tier-2: LinearForm(lambda v, w: ...) works; only the ARITY is enforced.

Claim: skfem mixed_poisson#3 -- wrap RHS callables with the @LinearForm
decorator, not LinearForm(lambda v, w: ...), because the lambda form
"mis-resolves the kwargs adapter inside asm" and "may raise opaque shape errors
deep inside skfem.assembly.form.linear_form".

Measured here: no. The two spellings assemble to the same vector, including when
extra fields are passed through asm(..., g=...). The one thing skfem does
enforce is the signature: a lambda taking only v raises an ordinary Python
TypeError at the call site.

Wrong variant: the 1-argument lambda. Right variants: both 2-argument spellings.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import Basis, ElementTriP1, LinearForm, MeshTri, asm


@LinearForm
def decorated(v, w):
    return 1.0 * v


@LinearForm
def decorated_kwargs(v, w):
    return w["g"] * v


def main() -> int:
    ok = True
    basis = Basis(MeshTri().refined(2), ElementTriP1())

    # --- WRONG variant: wrong arity -------------------------------------
    msg = ""
    try:
        asm(LinearForm(lambda v: 1.0 * v), basis)
    except TypeError as exc:
        msg = str(exc)
    print(f"one_arg_lambda_raises_typeerror={bool(msg)}")
    print(f"one_arg_lambda_msg={msg!r}")
    if "takes 1 positional argument but 2 were given" not in msg:
        print(f"FAIL: the 1-argument lambda did not raise the arity TypeError; "
              f"got {msg!r}", file=sys.stderr)
        ok = False

    # --- The claim's "wrong" variant, which is not wrong ----------------
    raised = ""
    try:
        f_lambda = asm(LinearForm(lambda v, w: 1.0 * v), basis)
    except Exception as exc:
        raised = f"{type(exc).__name__}: {exc}"
        f_lambda = np.array([np.nan])
    print(f"lambda_form_did_not_raise={not raised}")
    print(f"lambda_form_exception={raised!r}")
    if raised:
        print(f"FAIL: the lambda LinearForm raised ({raised}), so the claim's "
              f"warning would be right and this fixture stale", file=sys.stderr)
        ok = False

    f_dec = asm(decorated, basis)
    agree = bool(np.allclose(f_lambda, f_dec, rtol=0, atol=1e-15))
    print(f"lambda_and_decorator_agree={agree}")
    print(f"both_sums_are_one={abs(f_dec.sum() - 1.0) < 1e-12}")
    if not agree:
        print(f"FAIL: lambda sum {f_lambda.sum()!r} vs decorator "
              f"{f_dec.sum()!r}", file=sys.stderr)
        ok = False

    # ... and the same with an extra field routed through asm kwargs.
    g = basis.interpolate(basis.project(lambda x: x[0]))
    f_lambda_kw = asm(LinearForm(lambda v, w: w["g"] * v), basis, g=g)
    f_dec_kw = asm(decorated_kwargs, basis, g=g)
    agree_kw = bool(np.allclose(f_lambda_kw, f_dec_kw, rtol=1e-14))
    print(f"lambda_and_decorator_agree_with_kwargs={agree_kw}")
    if not agree_kw:
        print(f"FAIL: with kwargs, lambda sum {f_lambda_kw.sum()!r} vs "
              f"decorator {f_dec_kw.sum()!r}", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
