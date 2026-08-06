"""Tier-2: numpy's @ operator is wrong for skfem (d, d, nelem, nqp) tensors.

Claim: skfem hyperelasticity#9 -- the numpy @ matmul operator does NOT work on
skfem's (d, d, n_elem, n_quad)-shaped tensor fields inside a @BilinearForm;
matmul tries the last-two-dims convention and aborts with
"matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc
signature (n?,k),(k,m?)->(n?,m?) (size N is different from d)".  Fall back to
explicit component algebra or to skfem.helpers.

Measured on skfem 12.0.1 / numpy 1.26.4.  Two distinct behaviours, and the
second is the dangerous one:

  * WHEN n_elem != n_quad, `A @ B` raises ValueError with exactly the quoted
    message shape -- the N and d in the quoted text are the trailing two axis
    lengths, i.e. numpy contracted over (n_elem, n_quad) rather than over the
    tensor indices.
  * WHEN n_elem == n_quad the shapes happen to be conformable and `A @ B`
    raises NOTHING: it returns an array of the right shape holding a
    contraction over the WRONG axes.  The fixture builds such a case and shows
    the result differs from the true tensor product.  So "matmul aborts" is
    only half the story; on a mesh whose element count equals the quadrature
    count it fails silently.

skfem.helpers.mul / ddot and an explicit einsum both give the correct
contraction and agree with each other to machine precision.
"""
from __future__ import annotations

import sys

import numpy as np
import skfem
from skfem.helpers import ddot, mul, transpose


def tensor_field(d, nelem, nqp, seed):
    rng = np.random.default_rng(seed)
    A = np.zeros((d, d, nelem, nqp))
    for i in range(d):
        for j in range(d):
            A[i, j] = rng.standard_normal((nelem, nqp))
        A[i, i] += 2.0
    return A


def true_product(A, B):
    return np.einsum("ik...,kj...->ij...", A, B)


def main() -> int:
    ok = True
    print(f"skfem_version={skfem.__version__} numpy_version={np.__version__}")

    # --- the raising case: n_elem != n_quad ---------------------------
    A = tensor_field(2, 8, 3, 0)
    B = tensor_field(2, 8, 3, 1)
    print(f"tensor_shape={A.shape}")
    err = None
    try:
        A @ B
    except ValueError as e:
        err = e
    print(f"matmul_raises={err is not None}")
    print(f"matmul_exception_type={type(err).__name__}")
    print(f"matmul_message={str(err)!r}")
    quoted = ("matmul: Input operand 1 has a mismatch in its core dimension "
              "0, with gufunc signature (n?,k),(k,m?)->(n?,m?)")
    print(f"quoted_prefix_matches={quoted in str(err)}")
    if err is None or quoted not in str(err):
        print("FAIL: matmul did not abort with the documented message",
              file=sys.stderr)
        ok = False

    # --- the SILENT case: n_elem == n_quad ----------------------------
    As = tensor_field(2, 3, 3, 2)
    Bs = tensor_field(2, 3, 3, 3)
    silent_err = None
    try:
        got = As @ Bs
    except ValueError as e:
        silent_err = e
        got = None
    print(f"square_trailing_shape={As.shape}")
    print(f"matmul_raises_when_nelem_eq_nqp={silent_err is not None}")
    if got is not None:
        want = true_product(As, Bs)
        print(f"matmul_result_shape={got.shape} correct_shape={want.shape}")
        print(f"matmul_shape_looks_right={got.shape == want.shape}")
        dev = float(np.abs(got - want).max())
        print(f"matmul_max_abs_deviation_from_true_product={dev:.4e}")
        print(f"matmul_silently_wrong={dev > 1e-8}")
        if dev <= 1e-8:
            print("FAIL: matmul agreed with the true tensor product",
                  file=sys.stderr)
            ok = False
    else:
        print("FAIL: the n_elem == n_quad case raised, so there is no "
              "silent branch to document", file=sys.stderr)
        ok = False

    # --- the working fallbacks ----------------------------------------
    want = true_product(A, B)
    ein = np.einsum("ik...,kj...->ij...", A, B)
    helper = mul(A, B)
    print(f"einsum_matches_true_product={float(np.abs(ein - want).max()):.3e}")
    print(f"helpers_mul_matches_true_product="
          f"{float(np.abs(helper - want).max()):.3e}")
    print(f"helpers_mul_is_correct={float(np.abs(helper - want).max()) < 1e-12}")
    # helpers also cover the contractions a hyperelastic kernel needs
    print(f"helpers_ddot_shape={np.shape(ddot(A, B))}")
    print(f"helpers_transpose_is_index_swap="
          f"{float(np.abs(transpose(A) - np.swapaxes(A, 0, 1)).max()):.3e}")
    if float(np.abs(helper - want).max()) >= 1e-12:
        print("FAIL: skfem.helpers.mul did not give the tensor product",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
