"""Tier-2 fixture: skfem boundary lookup without with_boundaries.

The pitfall: in scikit-fem >= 8, `Basis.get_dofs("left")` requires
the mesh to have been pre-tagged via
``MeshQuad.with_boundaries({"left": lambda x: x[0] < tol})``.
Without the tag, the lookup raises ValueError.

This fixture intentionally omits with_boundaries and confirms
the Signal text ('ValueError', "Boundary 'left' not found")
appears in stderr — the Tier-2 runner greps for these
substrings.

Mutation control: T2_MUTATE=1 applies the documented fix at the
pathology site — the mesh is built with
with_boundaries({"left": lambda x: x[0] < tol}) — so
get_dofs("left") succeeds and both Signal strings vanish from the
output.  Re-run with T2_MUTATE=1 python source.py.
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np
from skfem import MeshQuad, Basis, ElementQuad1

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    m = MeshQuad.init_tensor(
        np.linspace(0, 1, 5), np.linspace(0, 1, 5)
    )
    # Deliberately NO with_boundaries call -- that omission is the
    # pathology.  T2_MUTATE=1 applies the documented tag here.
    if MUTATE:
        m = m.with_boundaries({"left": lambda x: x[0] < 1e-10})
    ib = Basis(m, ElementQuad1())
    try:
        _ = ib.get_dofs("left").flatten()
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: this fixture is supposed to raise but did not",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
