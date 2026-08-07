"""Tier-2: skfem DofsView non-subscriptable + condense x-shape requirement.

Two claims surfaced during the 2026-06-01 alignment audit, both
buried in the skfem heat Template comments but absent from the
catalog pitfalls list (so 'pitfalls'-topic knowledge queries
missed them):

  (1) ib.get_dofs() returns a DofsView object that does NOT
      support string subscripting. Code like
        dv = ib.get_dofs()
        left = dv['left']      # WRONG
      raises
        TypeError: 'DofsView' object is not subscriptable

      The correct path is to pass the boundary name positionally:
        left = ib.get_dofs('left')

  (2) condense(K, f, x=..., D=D)'s x argument must be a FULL-
      size vector (length == ib.N), NOT just the constrained-
      DOF values concatenated. Passing a short array (length =
      number of constrained DOFs) raises
        IndexError: index <N> is out of bounds for axis 0
                    with size <constrained count>
      because condense uses x as an indexed source via x[D].

Both are real LLM-traps because the natural attempt — treating
DofsView like a dict, or passing only the boundary values to
condense — produces a runtime error several lines below the
mistake, with a confusing index out-of-bounds rather than
something tied to BC setup.

This fixture asserts:
  * dv['left'] raises TypeError 'not subscriptable'.
  * ib.get_dofs('left') succeeds.
  * condense(K, f, x=short_array, D=D) raises IndexError
    with the literal 'out of bounds' substring.
  * condense(K, f, x=full_vector, D=D) succeeds and the
    resulting solution attains the prescribed boundary values
    at the constrained DOFs.

Mutation control: with T2_MUTATE=1 both documented fixes are
applied at their pathology sites — the DofsView probe becomes
the positional ib.get_dofs('left') instead of dv['left'], and
the condense probe is handed a FULL-size x (length ib.N)
instead of the short constrained-values array.  Neither call
raises any more, so
'dofsview_subscript_raises_typeerror=True',
'dofsview_msg_has_not_subscriptable=True',
'condense_short_x_raises_indexerror=True' and
'condense_short_msg_has_out_of_bounds=True' are all printed
=False and the fixture goes red.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
from skfem import (
    Basis,
    ElementQuad1,
    MeshQuad,
    condense,
    solve,
)
from skfem.models.poisson import laplace

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    m = MeshQuad.init_tensor(np.linspace(0, 1, 5),
                             np.linspace(0, 1, 5)).with_boundaries({
        "left": lambda x: x[0] < 1e-10,
        "right": lambda x: x[0] > 1.0 - 1e-10,
    })
    ib = Basis(m, ElementQuad1())
    print(f"basis_N={ib.N}")

    # (1) DofsView subscript trap
    dv = ib.get_dofs()
    print(f"get_dofs_type={type(dv).__name__}")
    raised_subscript = False
    msg = ""
    try:
        # documented fix under mutation: pass the name positionally
        _ = dv["left"] if not MUTATE else ib.get_dofs("left")
    except TypeError as exc:
        msg = str(exc)
        raised_subscript = "not subscriptable" in msg
    print(f"dofsview_subscript_raises_typeerror={raised_subscript}")
    print(f"dofsview_msg_has_not_subscriptable="
          f"{'not subscriptable' in msg}")

    # Correct path
    left = ib.get_dofs("left").flatten()
    right = ib.get_dofs("right").flatten()
    print(f"left_dof_count={len(left)}")

    # (2) condense x-shape requirement
    K = laplace.assemble(ib)
    f = ib.zeros()
    D = np.concatenate([left, right])

    # Wrong: short x (only constrained values)
    x_short = np.concatenate([
        np.full_like(left, 100.0, dtype=float),
        np.full_like(right, 0.0, dtype=float),
    ])
    # documented fix under mutation: x must be a FULL-size vector
    if MUTATE:
        x_probe = ib.zeros()
        x_probe[left] = 100.0
        x_probe[right] = 0.0
    else:
        x_probe = x_short
    print(f"x_short_len={len(x_probe)}")
    raised_short = False
    short_msg = ""
    try:
        solve(*condense(K, f, x=x_probe, D=D))
    except IndexError as exc:
        short_msg = str(exc)
        raised_short = "out of bounds" in short_msg
    print(f"condense_short_x_raises_indexerror={raised_short}")
    print(f"condense_short_msg_has_out_of_bounds="
          f"{'out of bounds' in short_msg}")

    # Correct: full-size x
    x_full = ib.zeros()
    x_full[left] = 100.0
    x_full[right] = 0.0
    u = solve(*condense(K, f, x=x_full, D=D))
    u_max = float(u.max())
    u_min = float(u.min())
    print(f"condense_full_x_solve_u_max={u_max:.6f}")
    print(f"condense_full_x_solve_u_min={u_min:.6f}")

    ok = (raised_subscript
          and raised_short
          and abs(u_max - 100.0) < 1e-6
          and abs(u_min - 0.0) < 1e-6)
    if ok:
        return 0
    print("FAIL: invariants not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
