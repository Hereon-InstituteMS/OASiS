"""Tier-2: split the [u; p] vector at basis.N, never at a hand-derived count.

Claim: skfem navier_stokes#6 -- use ib_u.N for the velocity-block split;
"hard-coding `n_u_dof = nx * dim * 2` for the velocity-block size gives
off-by-one errors on hex / triangle meshes where the DOF count depends on the
element family; ib_u.N is the canonical accessor."

Measured on skfem 12.0.1 across four velocity spaces on the same
MeshTri.init_tensor grid, plus a MeshQuad grid:

  * basis.N is exact by construction and the hand-derived formula is not
    merely off by one -- for a P2 velocity space it is off by more than a
    factor of three, because the formula counts vertices while P2 also
    carries edge DOFs.  Calling this an "off-by-one" understates it.
  * the formula happens to be RIGHT for the P1 vector space on a triangle
    grid, which is what makes it dangerous: it is correct in the first case
    a reader tries and silently wrong the moment the element order changes.
  * the consequence of splitting at the wrong index is silent: both slices
    come back with plausible lengths and the pressure slice is filled with
    velocity entries.  Nothing raises.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology site
-- split_index() returns basis.N instead of the hand-derived vertices-times-dim
count, both in the per-space comparison and in the [u; p] split itself.  Every
space then matches, nothing is contaminated, and
'formula_is_wrong_for_at_least_one_space=True',
'discrepancy_exceeds_a_factor_of_three=True' and
'naive_split_contaminates_pressure=True' disappear from the output.  Re-run:
T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from skfem import (
    Basis,
    ElementQuad1,
    ElementQuad2,
    ElementTriP1,
    ElementTriP2,
    ElementVector,
    MeshQuad,
    MeshTri,
)

NX = 8
MUTATE = os.environ.get("T2_MUTATE") == "1"


def split_index(bu, naive):
    """Where the [u; p] vector gets cut, and what the DOF count is compared to.

    THE PATHOLOGY is the hand-derived `naive` count; the documented fix is
    `ib_u.N`, which T2_MUTATE substitutes here and nowhere else.
    """
    return bu.N if MUTATE else naive


def main() -> int:
    ok = True
    tri = MeshTri.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                              np.linspace(0.0, 1.0, NX + 1))
    quad = MeshQuad.init_tensor(np.linspace(0.0, 1.0, NX + 1),
                                np.linspace(0.0, 1.0, NX + 1))
    dim = 2
    guess = (NX + 1) ** 2 * dim          # vertices * dim, the naive count
    print(f"nx={NX} naive_vertex_times_dim_count={guess}")

    rows = []
    for tag, mesh, elem in (("tri_P1", tri, ElementTriP1()),
                            ("tri_P2", tri, ElementTriP2()),
                            ("quad_Q1", quad, ElementQuad1()),
                            ("quad_Q2", quad, ElementQuad2())):
        bu = Basis(mesh, ElementVector(elem))
        used = split_index(bu, guess)
        rows.append((tag, bu.N, used))
        print(f"{tag}_basis_N={bu.N} naive={used} "
              f"ratio={bu.N / used:.4f}")
        print(f"{tag}_naive_matches_basis_N={bu.N == used}")

    matches = [t for t, n, u in rows if n == u]
    misses = [(t, n, u) for t, n, u in rows if n != u]
    print(f"spaces_where_the_formula_is_right={matches}")
    print(f"spaces_where_the_formula_is_wrong={[t for t, _, _ in misses]}")
    print(f"formula_is_right_for_at_least_one_space={bool(matches)}")
    print(f"formula_is_wrong_for_at_least_one_space={bool(misses)}")
    worst = max((abs(n - u) for _, n, u in misses), default=0)
    print(f"largest_absolute_discrepancy={worst}")
    print(f"discrepancy_is_off_by_one={worst == 1}")
    print(f"discrepancy_exceeds_a_factor_of_three="
          f"{any(n > 3 * u or u > 3 * n for _, n, u in misses)}")
    if not matches:
        print("FAIL: the naive formula was never right, so it would not "
              "survive a first test", file=sys.stderr)
        ok = False
    if not misses:
        print("FAIL: the naive formula was right everywhere", file=sys.stderr)
        ok = False
    if worst == 1:
        print("FAIL: the discrepancy really was off-by-one", file=sys.stderr)
        ok = False

    # --- the silent consequence -------------------------------------------
    bu = Basis(tri, ElementVector(ElementTriP2()))
    bp = Basis(tri, ElementTriP1())
    total = bu.N + bp.N
    x = np.arange(total, dtype=float)
    cut = split_index(bu, guess)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        good_u, good_p = x[:bu.N], x[bu.N:]
        bad_u, bad_p = x[:cut], x[cut:]
        msgs = sorted({str(c.message) for c in caught})
    print(f"total_length={total} basis_N={bu.N} pressure_N={bp.N}")
    print(f"correct_split_pressure_length={len(good_p)}")
    print(f"naive_split_pressure_length={len(bad_p)}")
    print(f"naive_split_warnings={msgs!r}")
    print(f"naive_split_is_silent={not msgs}")
    print(f"naive_pressure_slice_is_nonempty={len(bad_p) > 0}")
    contaminated = int(np.sum(bad_p[:max(len(bad_p) - bp.N, 0)] < bu.N))
    print(f"naive_pressure_entries_taken_from_velocity={contaminated}")
    print(f"naive_split_contaminates_pressure={contaminated > 0}")
    if msgs:
        print(f"FAIL: the naive split warned {msgs!r}", file=sys.stderr)
        ok = False
    if contaminated <= 0:
        print("FAIL: the naive split did not pull velocity entries into the "
              "pressure slice", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
