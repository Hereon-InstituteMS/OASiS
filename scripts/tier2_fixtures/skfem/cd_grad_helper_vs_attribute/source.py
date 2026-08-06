"""Tier-2: grad(u) is fine; the @ operator is what breaks.

Claim: skfem convection_diffusion#0 -- BOTH u.grad and skfem.helpers.grad(u) are
correct inside a @BilinearForm, and the prior claim that grad(u) raises
"NameError: grad is not defined" is FALSE. What does bite is numpy's @ on the
(dim, n_elem, n_qp) DiscreteField arrays.

Wrong variant (a): grad(u) @ grad(v) -- raises the gufunc core-dimension error.
Wrong variant (b): a genuinely undefined free function -- raises a plain
NameError, which is the shape the retired claim mistook for the helper failing.
Right variants: skfem.helpers.grad and the .grad attribute, both checked against
the library's own laplace form.
"""
from __future__ import annotations

import sys

from skfem import Basis, BilinearForm, ElementTriP1, MeshTri
from skfem.helpers import dot, grad
from skfem.models.poisson import laplace


@BilinearForm
def via_helper(u, v, w):
    return dot(grad(u), grad(v))


@BilinearForm
def via_attribute(u, v, w):
    return u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1]


@BilinearForm
def via_matmul(u, v, w):
    return grad(u) @ grad(v)


@BilinearForm
def via_undefined_name(u, v, w):
    return curlgrad(u) * v            # noqa: F821 - deliberate probe


def main() -> int:
    ok = True
    basis = Basis(MeshTri().refined(2), ElementTriP1())
    reference = laplace.assemble(basis)
    helper = via_helper.assemble(basis)
    attribute = via_attribute.assemble(basis)

    nnz = (reference.nnz, helper.nnz, attribute.nnz)
    print(f"nnz={nnz}")
    print(f"nnz_all_105={nnz == (105, 105, 105)}")
    helper_diff = abs(helper - reference).max()
    attribute_diff = abs(attribute - reference).max()
    print(f"helper_max_abs_diff={helper_diff!r}")
    print(f"attribute_max_abs_diff={attribute_diff!r}")
    print(f"helper_matches_laplace_exactly={helper_diff == 0.0}")
    print(f"attribute_matches_laplace_exactly={attribute_diff == 0.0}")
    if nnz != (105, 105, 105) or helper_diff != 0.0 or attribute_diff != 0.0:
        print(f"FAIL: the two spellings do not reproduce laplace exactly "
              f"(nnz {nnz}, diffs {helper_diff!r} / {attribute_diff!r})",
              file=sys.stderr)
        ok = False

    # --- WRONG variant (a): the @ operator -------------------------------
    matmul_msg = ""
    try:
        via_matmul.assemble(basis)
    except ValueError as exc:
        matmul_msg = str(exc)
    print(f"matmul_raises_valueerror={bool(matmul_msg)}")
    print(f"matmul_msg={matmul_msg!r}")
    if "core dimension" not in matmul_msg:
        print(f"FAIL: the @ operator did not raise the gufunc core-dimension "
              f"error; got {matmul_msg!r}", file=sys.stderr)
        ok = False

    # --- WRONG variant (b): an actually undefined name -------------------
    undefined_msg = ""
    undefined_type = ""
    try:
        via_undefined_name.assemble(basis)
    except NameError as exc:
        undefined_type = type(exc).__name__
        undefined_msg = str(exc)
    print(f"undefined_name_msg={undefined_msg!r}")
    print(f"undefined_name_is_plain_nameerror={undefined_type == 'NameError'}")
    if undefined_type != "NameError":
        print(f"FAIL: an undefined free function did not raise NameError; got "
              f"{undefined_type!r}", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
