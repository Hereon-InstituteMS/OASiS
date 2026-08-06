"""Tier-2: m.f2t[1] is -1 on boundary facets, and numpy will not tell you.

Claim: skfem adaptive_poisson#3 -- m.f2t has shape (2, n_facets) and the
second row is -1 for boundary facets; iterating over all facets without
filtering f2t[1] >= 0 adds spurious boundary-jump contributions.  Signal:
"IndexError 'index -1 is out of bounds' if you dereference f2t[1, fi] as an
element id without the >= 0 check".

Measured on skfem 12.0.1, MeshTri().refined(2) -- 32 elements, 56 facets, 16
of them on the boundary:

  * the shape and the -1 sentinel are exactly as claimed, and the count of
    -1 entries equals len(m.boundary_facets()).
  * THE QUOTED IndexError NEVER FIRES.  numpy treats -1 as "the last
    element", so m.t[:, m.f2t[1]] returns a full (3, 56) array with the LAST
    element's connectivity silently substituted on every boundary facet.
    There is no exception, no warning, and the array has exactly the shape a
    correct one would have -- so neither a try/except nor a shape check
    catches it.
  * the damage is quantifiable: the substituted columns disagree with the
    real interior neighbours on every boundary facet, and a jump computed
    that way picks up a contribution from an unrelated element on the far
    side of the mesh.
  * the guard that works is the mask f2t[1] >= 0, which selects exactly the
    interior facets.

Mutation control: T2_MUTATE=1 applies that documented guard at the
dereference site -- the facet list fed to m.t[:, f2t[1]] becomes the masked
interior set instead of every facet -- so no -1 is ever dereferenced and the
neighbour array no longer spans all 56 facets.  Re-run with
T2_MUTATE=1 python source.py.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from skfem import MeshTri

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    ok = True
    m = MeshTri().refined(2)
    ne, nf = m.t.shape[1], m.facets.shape[1]
    f2t = m.f2t
    print(f"elements={ne} facets={nf}")
    print(f"f2t_shape={f2t.shape}")
    print(f"f2t_is_two_by_nfacets={f2t.shape == (2, nf)}")
    print(f"f2t_second_row_min={int(f2t[1].min())}")
    print(f"f2t_first_row_min={int(f2t[0].min())}")
    n_minus_one = int((f2t[1] < 0).sum())
    n_boundary = len(m.boundary_facets())
    print(f"count_of_minus_one={n_minus_one}")
    print(f"boundary_facets_reported={n_boundary}")
    print(f"minus_one_marks_boundary_facets={n_minus_one == n_boundary}")
    print(f"minus_one_set_matches_boundary_facets="
          f"{bool(np.array_equal(np.sort(np.nonzero(f2t[1] < 0)[0]),
                                 np.sort(m.boundary_facets())))}")
    if f2t.shape != (2, nf) or n_minus_one != n_boundary:
        print("FAIL: f2t does not carry the -1 boundary sentinel",
              file=sys.stderr)
        ok = False

    # --- the quoted IndexError -------------------------------------------
    # The pathology: every facet is dereferenced through f2t[1], including the
    # boundary ones carrying -1.  T2_MUTATE=1 applies the documented guard
    # (the mask f2t[1] >= 0) here, so only interior facets are dereferenced.
    fsel = np.arange(nf) if not MUTATE else np.nonzero(f2t[1] >= 0)[0]
    err = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            neigh = m.t[:, f2t[1][fsel]]
        except IndexError as e:
            err = e
            neigh = None
        msgs = sorted({str(c.message) for c in caught})
    print(f"indexing_by_minus_one_raises={err is not None}")
    print(f"indexing_warnings={msgs!r}")
    print(f"indexing_is_silent={err is None and not msgs}")
    if err is not None:
        print(f"FAIL: indexing by -1 raised {type(err).__name__}: {err}",
              file=sys.stderr)
        ok = False
    else:
        print(f"neighbour_array_shape={neigh.shape}")
        print(f"neighbour_array_has_the_expected_shape="
              f"{neigh.shape == (3, nf)}")
        bnd = np.nonzero(f2t[1][fsel] < 0)[0]
        wrapped = m.t[:, [ne - 1]]
        print(f"boundary_columns_equal_the_last_element="
              f"{bool(len(bnd) > 0
                      and np.array_equal(neigh[:, bnd],
                                         np.repeat(wrapped, len(bnd),
                                                   axis=1)))}")
        real = m.t[:, f2t[0][fsel][bnd]]
        print(f"substituted_columns_differ_from_the_real_neighbour="
              f"{int(np.sum(np.any(neigh[:, bnd] != real, axis=0)))}"
              f"_of_{len(bnd)}")
        print(f"every_boundary_facet_is_corrupted="
              f"{len(bnd) > 0
                 and int(np.sum(np.any(neigh[:, bnd] != real, axis=0)))
                 == len(bnd)}")
        if neigh.shape != (3, nf):
            print("FAIL: the wrapped indexing did not even change the shape, "
                  "so a shape check would catch it", file=sys.stderr)
            ok = False

    # --- the guard that works ---------------------------------------------
    interior = np.nonzero(f2t[1] >= 0)[0]
    print(f"interior_facets={len(interior)}")
    print(f"interior_plus_boundary_equals_all="
          f"{len(interior) + n_boundary == nf}")
    print(f"guarded_indices_are_all_valid="
          f"{bool((f2t[1][interior] >= 0).all()
                  and (f2t[1][interior] < ne).all())}")
    if len(interior) + n_boundary != nf:
        print("FAIL: the interior/boundary split does not cover all facets",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
