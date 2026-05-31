"""Tier-2 fixture: skfem boundary lookup without with_boundaries.

The pitfall: in scikit-fem >= 8, `Basis.get_dofs("left")` requires
the mesh to have been pre-tagged via
``MeshQuad.with_boundaries({"left": lambda x: x[0] < tol})``.
Without the tag, the lookup raises ValueError.

This fixture intentionally omits with_boundaries and confirms
the Signal text ('ValueError', "Boundary 'left' not found")
appears in stderr — the Tier-2 runner greps for these
substrings.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
from skfem import MeshQuad, Basis, ElementQuad1


def main() -> int:
    m = MeshQuad.init_tensor(
        np.linspace(0, 1, 5), np.linspace(0, 1, 5)
    )
    # Deliberately NO with_boundaries call.
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
