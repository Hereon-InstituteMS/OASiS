"""Tier-2: condense's x must be full length; short raises, LONG is silent.

Claim: skfem heat#5 -- condense(K, f, x=..., D=D)'s x must have length basis.N,
not len(D). A short array raises IndexError 'index <N> is out of bounds for axis
0 with size <constrained count>' from condense's internal x[D] indexing.

This fixture pins the arithmetic behind that confusing message: the reported
INDEX is a global DOF number drawn from D, and the reported SIZE is len(x). It
also records the asymmetry the claim omits -- a too-LONG x raises nothing and is
silently accepted, so only one of the two length mistakes is loud.

Mutation control: T2_MUTATE=1 applies the documented fix at the short-x site --
the probe array is built with basis.zeros() (length basis.N) instead of
length len(D), carrying the same prescribed values.  condense then accepts it,
so the IndexError never happens and
'out of bounds for axis 0 with size 10', 'short_x_raises_indexerror=True',
'reported_index_is_a_global_dof_number=True' and
'reported_size_equals_len_short_x=True' disappear from the output.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from skfem import Basis, ElementQuad1, MeshQuad, condense, solve
from skfem.models.poisson import laplace

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    ok = True
    m = MeshQuad.init_tensor(
        np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5),
    ).with_boundaries({
        "left": lambda x: x[0] < 1e-10,
        "right": lambda x: x[0] > 1.0 - 1e-10,
    })
    basis = Basis(m, ElementQuad1())
    K = laplace.assemble(basis)
    f = basis.zeros()
    left = basis.get_dofs("left").flatten()
    right = basis.get_dofs("right").flatten()
    D = np.concatenate([left, right])
    print(f"basis_N={basis.N}")
    print(f"len_D={len(D)}")

    # --- WRONG variant (a): short x -------------------------------------
    # Length len(D), the mistake.  Under mutation the documented fix is applied
    # here: the same prescribed values in an array of length basis.N.
    if not MUTATE:
        x_short = np.concatenate([np.full(len(left), 100.0),
                                  np.zeros(len(right))])
    else:
        x_short = basis.zeros()
        x_short[left] = 100.0
        x_short[right] = 0.0
    msg = ""
    try:
        solve(*condense(K, f, x=x_short, D=D))
    except IndexError as exc:
        msg = str(exc)
    print(f"short_x_len={len(x_short)}")
    print(f"short_x_raises_indexerror={bool(msg)}")
    print(f"short_x_msg={msg!r}")
    if "out of bounds" not in msg:
        print(f"FAIL: a short x did not raise the out-of-bounds IndexError; "
              f"got {msg!r}", file=sys.stderr)
        ok = False

    reported = [int(t) for t in msg.replace(",", " ").split()
                if t.isdigit()] if msg else []
    index_is_dof = bool(reported) and reported[0] in set(int(d) for d in D)
    size_is_len_x = len(reported) > 1 and reported[-1] == len(x_short)
    print(f"reported_index_is_a_global_dof_number={index_is_dof}")
    print(f"reported_size_equals_len_short_x={size_is_len_x}")
    if not (index_is_dof and size_is_len_x):
        print(f"FAIL: the message numbers {reported!r} are not "
              f"(global DOF, len(x)) as claimed", file=sys.stderr)
        ok = False

    # --- WRONG variant (b): too-long x, silently accepted ---------------
    x_long = np.zeros(basis.N + 7)
    long_raised = ""
    try:
        solve(*condense(K, f, x=x_long, D=D))
    except Exception as exc:
        long_raised = f"{type(exc).__name__}: {exc}"
    print(f"long_x_silently_accepted={not long_raised}")
    print(f"long_x_raised={long_raised!r}")
    if long_raised:
        print(f"FAIL: a too-long x now raises ({long_raised}), so the "
              f"asymmetry this fixture records is gone", file=sys.stderr)
        ok = False

    # --- RIGHT variant --------------------------------------------------
    x_full = basis.zeros()
    x_full[left] = 100.0
    x_full[right] = 0.0
    u = solve(*condense(K, f, x=x_full, D=D))
    good = (np.allclose(u[left], 100.0, atol=1e-9)
            and np.allclose(u[right], 0.0, atol=1e-9)
            and 0.0 <= u.min() and u.max() <= 100.0 + 1e-9)
    print(f"full_length_x_reproduces_prescribed_values={good}")
    if not good:
        print(f"FAIL: full-length x gave u[left]={u[left][:3]!r}, "
              f"u[right]={u[right][:3]!r}", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
