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

    # WHAT THE VECTOR BEING SPLIT HAS TO BE.
    #
    # It used to be `x = np.arange(total)`, and the contamination test was
    # `bad_p[:len(bad_p) - bp.N] < bu.N`.  On an arange that reduces to
    # `index < bu.N` -- an arithmetic identity, worth exactly `max(bu.N - cut,
    # 0)` for every mesh, every element and every physics, and saying nothing
    # about a velocity or a pressure.  It also merely restated
    # `tri_P2_naive_matches_basis_N=False`, which is asserted above.
    #
    # The vector is now a real pair of FIELDS: a velocity projected onto the
    # ElementVector space and a pressure projected onto P1, on separated value
    # ranges (|u| near 10, |p| <= 1) so an entry can be told apart by WHAT IT
    # IS rather than by where it sits.  "Contaminated" then means the pressure
    # slice holds entries carrying velocity magnitudes, which is a statement
    # about skfem's DOF layout and can come out false -- and the field error
    # below says how wrong the resulting pressure is.
    u_h = bu.project(lambda x: np.array([10.0 + np.sin(np.pi * x[0]),
                                         10.0 + np.cos(np.pi * x[1])]))
    p_h = bp.project(lambda x: np.sin(2.0 * np.pi * x[0]) * np.cos(np.pi * x[1]))
    vec = np.concatenate([u_h, p_h])
    print(f"velocity_block_min_abs={np.abs(u_h).min():.4f}")
    print(f"pressure_block_max_abs={np.abs(p_h).max():.4f}")
    ranges_separate = bool(np.abs(u_h).min() > 2.0 and np.abs(p_h).max() < 2.0)
    print(f"velocity_and_pressure_value_ranges_separate={ranges_separate}")
    if not ranges_separate:
        print("FAIL: the two blocks overlap in value, so 'came from the "
              "velocity block' cannot be read off the numbers", file=sys.stderr)
        ok = False

    cut = split_index(bu, guess)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        good_u, good_p = vec[:bu.N], vec[bu.N:]
        bad_u, bad_p = vec[:cut], vec[cut:]
        msgs = sorted({str(c.message) for c in caught})
    print(f"total_length={total} basis_N={bu.N} pressure_N={bp.N}")
    print(f"correct_split_pressure_length={len(good_p)}")
    print(f"naive_split_pressure_length={len(bad_p)}")
    print(f"naive_split_warnings={msgs!r}")
    print(f"naive_split_is_silent={not msgs}")
    print(f"naive_pressure_slice_is_nonempty={len(bad_p) > 0}")
    contaminated = int(np.sum(np.abs(bad_p) > 2.0))
    print(f"naive_pressure_entries_taken_from_velocity={contaminated}")
    print(f"naive_split_contaminates_pressure={contaminated > 0}")
    # And what the mis-sliced pressure is as a FIELD.  A caller reads the slice
    # from its FRONT -- p[0], p[1], ... -- so the comparison is over the leading
    # bp.N entries, which is what downstream code would use as the pressure.
    # (Aligning on the TAIL instead compares vec[bu.N:] with itself and prints
    # a relative error of exactly 0 whatever the split did; that is the same
    # mistake one level up, and it was caught by running this.)
    got_p = bad_p[:len(good_p)]
    n = min(len(got_p), len(good_p))
    rel = float(np.abs(got_p[:n] - good_p[:n]).max()
                / max(1e-30, np.abs(good_p).max()))
    print(f"naive_pressure_field_relative_error={rel:.4e}")
    print(f"naive_pressure_field_is_wrong={rel > 0.5}")
    if msgs:
        print(f"FAIL: the naive split warned {msgs!r}", file=sys.stderr)
        ok = False
    if contaminated <= 0:
        print("FAIL: the naive split did not pull velocity entries into the "
              "pressure slice", file=sys.stderr)
        ok = False
    if rel <= 0.5:
        print(f"FAIL: the mis-sliced pressure field is within {rel!r} of the "
              f"real one, so the split is not silently wrong", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
