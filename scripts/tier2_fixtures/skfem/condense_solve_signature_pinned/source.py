"""Tier-2: skfem condense + solve signatures pinned.

The catalog text in src/backends/skfem/generators/*.py
makes specific claims about which keyword arguments
condense() and solve() accept (e.g. D=, I=, x=, expand=).
Skfem has renamed/reordered these in past majors —
catalog drift here breaks every generator at runtime.

This fixture binds the kwarg-name set as a regression
contract:

  condense(A, b=None, x=None, I=None, D=None, expand=True)
  solve(A, b, x=None, I=None, solver=None, **kwargs)

If skfem 13.x drops or renames a kwarg, the gate fails
within hours.
"""
from __future__ import annotations

import inspect
import sys

import skfem


REQUIRED_CONDENSE_PARAMS = {
    "A", "b", "x", "I", "D", "expand"}
REQUIRED_SOLVE_PARAMS = {"A", "b", "x", "I", "solver"}


def main() -> int:
    print(f"skfem_version={skfem.__version__}")

    sig_c = inspect.signature(skfem.condense)
    sig_s = inspect.signature(skfem.solve)
    cparams = set(sig_c.parameters)
    sparams = set(sig_s.parameters)
    print(f"condense_params={sorted(cparams)}")
    print(f"solve_params={sorted(sparams)}")

    missing_c = REQUIRED_CONDENSE_PARAMS - cparams
    missing_s = REQUIRED_SOLVE_PARAMS - sparams
    print(f"condense_missing_required={sorted(missing_c)}")
    print(f"solve_missing_required={sorted(missing_s)}")

    # condense() must accept DofsView as I=/D=
    # (catalog pitfall claim).
    m = skfem.MeshTri()
    b = skfem.Basis(m, skfem.ElementTriP1())
    dofs = b.get_dofs()
    print(f"DofsView_type={type(dofs).__name__}")
    print(f"DofsView_is_DofsView="
          f"{type(dofs).__name__ == 'DofsView'}")

    # condense() with x kwarg returns extended tuple
    # (catalog claim).
    import numpy as np
    from scipy.sparse import csr_matrix
    A = csr_matrix(np.eye(b.N))
    bb = np.ones(b.N)
    x = np.zeros(b.N)
    res = skfem.condense(A, bb, x=x, D=dofs)
    res_tuple_len = len(res) if isinstance(res, tuple) else 1
    print(f"condense_with_x_returns_tuple_len={res_tuple_len}")

    ok = (
        not missing_c
        and not missing_s
        and type(dofs).__name__ == "DofsView"
        and res_tuple_len >= 3
    )
    if ok:
        return 0
    print("FAIL: condense/solve signature drift",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
