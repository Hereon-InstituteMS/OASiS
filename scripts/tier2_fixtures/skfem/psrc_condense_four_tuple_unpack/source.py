"""Tier-2: condense returns a 4-tuple that only skfem.solve can consume.

Claim: skfem point_source#3 -- condense(K, f, D=D) returns 4 values
(K_c, f_c, x_c, I) that must be unpacked via solve(*condense(...)).  Passing
the 4-tuple to a scipy solver raises ValueError('The truth value of an array
with more than one element is ambiguous. Use a.any() or a.all()') from
linsolve.py, because the THIRD element gets bound to spsolve's use_umfpack
flag -- it is NOT a "too many positional arguments" TypeError.

Measured on skfem 12.0.1 / scipy 1.15.3, MeshTri().refined(4), ElementTriP1,
289 DOFs, a unit point load at the centre node:

  * condense returns exactly four objects, of types csr_matrix, ndarray,
    ndarray, ndarray.
  * scipy.sparse.linalg.spsolve(*condense(...)) raises the quoted ValueError
    verbatim, and the traceback comes from scipy's linsolve module.
  * neither of the alternative messages the older wording suggested appears:
    there is no "takes ... positional arguments" TypeError and no
    "unhashable type: 'tuple'".
  * skfem.solve(*condense(...)) consumes all four and returns a full-length
    ib.N vector whose Dirichlet entries carry the prescribed values -- which
    is the whole reason the extra two are there.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import scipy.sparse.linalg as spl
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace


def main() -> int:
    ok = True
    m = MeshTri().refined(4)
    ib = Basis(m, ElementTriP1())
    K = laplace.assemble(ib)
    D = ib.get_dofs().all()
    f = ib.zeros()
    src = int(np.argmin((ib.doflocs[0] - 0.5) ** 2
                        + (ib.doflocs[1] - 0.5) ** 2))
    f[src] = 1.0
    print(f"dofs_N={ib.N} boundary_dofs={len(D)}")

    packed = condense(K, f, D=D)
    print(f"condense_returns={len(packed)}")
    print(f"condense_types={[type(x).__name__ for x in packed]}")
    print(f"condense_returns_four={len(packed) == 4}")
    print(f"third_element_is_ndarray="
          f"{isinstance(packed[2], np.ndarray)}")
    print(f"third_element_length={len(packed[2])}")
    if len(packed) != 4:
        print(f"FAIL: condense returned {len(packed)} values, not 4",
              file=sys.stderr)
        ok = False

    err = None
    tb = ""
    try:
        spl.spsolve(*packed)
    except Exception as e:                           # noqa: BLE001
        err = e
        tb = "".join(traceback.format_tb(e.__traceback__))
    print(f"spsolve_star_condense_raises={err is not None}")
    print(f"spsolve_exception_type={type(err).__name__ if err else None}")
    print(f"spsolve_message={str(err)!r}")
    quoted = ("The truth value of an array with more than one element is "
              "ambiguous. Use a.any() or a.all()")
    print(f"quoted_valueerror_text_matches={quoted in str(err)}")
    print(f"traceback_comes_from_linsolve={'linsolve' in tb}")
    print(f"is_a_TypeError={isinstance(err, TypeError)}")
    print(f"mentions_positional_arguments="
          f"{'positional argument' in str(err)}")
    print(f"mentions_unhashable_tuple={'unhashable' in str(err)}")
    if not isinstance(err, ValueError) or quoted not in str(err):
        print(f"FAIL: spsolve(*condense(...)) did not raise the documented "
              f"ValueError; got {type(err).__name__}: {err}", file=sys.stderr)
        ok = False
    if isinstance(err, TypeError) or "positional argument" in str(err):
        print("FAIL: the failure WAS the positional-argument TypeError the "
              "older wording quoted", file=sys.stderr)
        ok = False

    # --- what the four values are for -------------------------------------
    xbc = ib.zeros()
    xbc[D] = 0.25
    u = solve(*condense(K, f, x=xbc, D=D))
    print(f"solve_star_condense_length={len(u)}")
    print(f"solve_returns_full_length_vector={len(u) == ib.N}")
    print(f"dirichlet_entries_carry_prescribed_value="
          f"{bool(np.allclose(u[D], 0.25))}")
    print(f"interior_is_not_the_prescribed_value="
          f"{bool(np.abs(u[src] - 0.25) > 1e-3)}")
    if len(u) != ib.N or not np.allclose(u[D], 0.25):
        print("FAIL: skfem.solve did not honour the condensed data",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
