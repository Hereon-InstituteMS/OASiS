"""Tier-2: OrientedBoundary lives in skfem.generic_utils, not at top level.

Claim: skfem mixed_poisson#5 -- skfem.OrientedBoundary raises AttributeError;
import it from skfem.generic_utils. It subclasses numpy.ndarray and carries an
`ori` attribute of +-1 per facet, and FacetBasis branches on
isinstance(self.find, OrientedBoundary) to decide whether to apply orientation,
so passing a plain ndarray silently disables it.

Wrong variant: the top-level spelling. The silent-disable half of the claim is
checked by reading the FacetBasis source for the isinstance branch, since a
silent behaviour cannot be observed from its output by construction.
"""
from __future__ import annotations

import inspect
import sys

import numpy as np
import skfem


def main() -> int:
    ok = True

    # --- WRONG variant: the top-level spelling --------------------------
    print(f"toplevel_has_orientedboundary={hasattr(skfem, 'OrientedBoundary')}")
    msg = ""
    try:
        skfem.OrientedBoundary            # noqa: B018 - deliberate probe
    except AttributeError as exc:
        msg = str(exc)
    print(f"toplevel_attributeerror={msg!r}")
    if "no attribute 'OrientedBoundary'" not in msg:
        print(f"FAIL: expected the top-level AttributeError, got {msg!r}",
              file=sys.stderr)
        ok = False

    # --- RIGHT variant --------------------------------------------------
    imported = False
    try:
        from skfem.generic_utils import OrientedBoundary
        imported = True
    except ImportError as exc:
        print(f"FAIL: skfem.generic_utils import failed: {exc}",
              file=sys.stderr)
        return 2
    print(f"generic_utils_import_succeeds={imported}")
    print(f"is_ndarray_subclass={issubclass(OrientedBoundary, np.ndarray)}")
    if not issubclass(OrientedBoundary, np.ndarray):
        print("FAIL: OrientedBoundary is no longer an ndarray subclass",
              file=sys.stderr)
        ok = False

    ob = OrientedBoundary(np.array([0, 1, 2]), np.array([1, -1, 1]))
    has_ori = hasattr(ob, "ori")
    print(f"carries_ori_attribute={has_ori}")
    print(f"ori_values={np.asarray(getattr(ob, 'ori', [])).tolist()}")
    if not has_ori:
        print("FAIL: the constructed OrientedBoundary has no `ori`",
              file=sys.stderr)
        ok = False

    # The silent-disable branch, read from the source.
    from skfem.assembly.basis import FacetBasis
    src = inspect.getsource(FacetBasis)
    branch = "OrientedBoundary" in src and "isinstance" in src
    print(f"facetbasis_isinstance_check_present={branch}")
    if not branch:
        print("FAIL: FacetBasis no longer branches on OrientedBoundary, so a "
              "plain ndarray may no longer silently disable orientation",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
