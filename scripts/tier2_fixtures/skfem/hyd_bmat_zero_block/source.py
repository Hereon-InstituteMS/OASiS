"""Tier-2: what scipy.sparse.bmat really raises for a scalar zero block.

Claim: skfem hydraulic_resistance#1 -- bmat([[A, B.T], [B, None]]) is the
canonical Stokes block assembly; the None produces a correctly shaped zero
block.  "Forgetting to wrap in bmat (passing [[A, B.T], [B, 0]] with scalar 0)
raises TypeError.  Signal: TypeError 'no supported conversion for types:
(dtype('int64'),)' or 'unsupported operand type for *' from
scipy.sparse.bmat when 0 is used instead of None."

Measured on skfem 12.0.1 / scipy 1.15.3:

  * THE EXCEPTION TYPE AND THE MESSAGE ARE BOTH WRONG.  Passing a scalar 0
    raises a ValueError, not a TypeError, and the message is
    "scipy sparse array classes do not support instantiation from a scalar".
    Neither quoted string appears.  An `except TypeError` around the assembly
    does not catch it, and a gate grepping for either quoted phrase never
    matches.
  * the None form works and is the point: it yields a block of exactly the
    right shape with no stored entries of its own, and an explicit
    csr_matrix((nP, nP)) gives a matrix with an identical shape and non-zero
    count, so the two are interchangeable.
  * a scalar 0.0 (float) fails the same way as the int, so the dtype in the
    old quoted message was incidental too.
"""
from __future__ import annotations

import sys

import numpy as np
import scipy.sparse as sp


def main() -> int:
    ok = True
    print(f"scipy_version={sp.__name__} {__import__('scipy').__version__}")
    n_u, n_p = 6, 3
    A = sp.eye(n_u, format="csr")
    B = sp.csr_matrix(np.ones((n_p, n_u)))

    errors = {}
    for tag, zero in (("int", 0), ("float", 0.0)):
        err = None
        try:
            sp.bmat([[A, B.T], [B, zero]], format="csr")
        except Exception as e:                        # noqa: BLE001
            err = e
        errors[tag] = err
        print(f"scalar_{tag}_raises={err is not None}")
        print(f"scalar_{tag}_exception_type="
              f"{type(err).__name__ if err else None}")
        print(f"scalar_{tag}_message={str(err)!r}")
        if err is None:
            print(f"FAIL: a scalar {tag} zero block did not raise",
                  file=sys.stderr)
            ok = False

    e_int = errors["int"]
    print(f"is_a_TypeError={isinstance(e_int, TypeError)}")
    print(f"is_a_ValueError={isinstance(e_int, ValueError)}")
    q1 = "no supported conversion for types"
    q2 = "unsupported operand type"
    blob = " ".join(str(v) for v in errors.values())
    print(f"quoted_no_supported_conversion_present={q1 in blob}")
    print(f"quoted_unsupported_operand_present={q2 in blob}")
    print(f"actual_message_mentions_instantiation_from_a_scalar="
          f"{'instantiation from a scalar' in str(e_int)}")
    print(f"except_TypeError_would_catch_it="
          f"{isinstance(e_int, TypeError)}")
    if isinstance(e_int, TypeError):
        print("FAIL: it really is a TypeError, so the entry's type is right",
              file=sys.stderr)
        ok = False
    if q1 in blob or q2 in blob:
        print("FAIL: one of the quoted messages WAS produced",
              file=sys.stderr)
        ok = False

    # --- the None form, and the explicit alternative ----------------------
    K_none = sp.bmat([[A, B.T], [B, None]], format="csr")
    K_expl = sp.bmat([[A, B.T], [B, sp.csr_matrix((n_p, n_p))]],
                     format="csr")
    print(f"none_block_shape={K_none.shape} nnz={K_none.nnz}")
    print(f"explicit_zero_shape={K_expl.shape} nnz={K_expl.nnz}")
    print(f"none_gives_the_right_shape="
          f"{K_none.shape == (n_u + n_p, n_u + n_p)}")
    print(f"none_and_explicit_agree_on_shape="
          f"{K_none.shape == K_expl.shape}")
    print(f"none_and_explicit_agree_on_nnz={K_none.nnz == K_expl.nnz}")
    diff = (K_none - K_expl)
    diff.eliminate_zeros()
    print(f"none_and_explicit_difference_nnz={diff.nnz}")
    print(f"none_and_explicit_are_interchangeable={diff.nnz == 0}")
    if K_none.shape != (n_u + n_p, n_u + n_p) or diff.nnz != 0:
        print("FAIL: the None block is not equivalent to an explicit zero",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
