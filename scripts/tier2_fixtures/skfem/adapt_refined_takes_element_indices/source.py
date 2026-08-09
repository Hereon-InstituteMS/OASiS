"""Tier-2: MeshTri.refined interprets its argument as ELEMENT indices.

Claim: skfem adaptive_poisson#1 -- skfem >= 8 expects
MeshTri.refined(element_indices); a boolean mask of length n_elements works
(it is converted via np.where), but passing FACET indices raises IndexError
because refined interprets its argument as element indices.  Signal:
IndexError 'index N is out of bounds for axis 0 with size M' from
MeshTri.refined, or a mesh with fewer elements than expected.

Measured on skfem 12.0.1, MeshTri().refined(2): 32 elements, 56 facets.

  * an integer index array and the equivalent boolean mask give IDENTICAL
    meshes -- same element count, same vertex count, same connectivity -- so
    the mask conversion really is exact and not merely similar.
  * passing the full facet index array raises IndexError; the message names
    axis 1 rather than axis 0 on this version, and it fires at the first
    index that exceeds the element count, so the number in the message is
    the element count, not the facet count.  A gate matching the literal
    'axis 0' would miss it.
  * a scalar integer argument does NOT mean "refine that element": it is the
    uniform-refinement overload, so refined(3) uniformly refines three times
    and multiplies the element count by 64.  That is the shape a mask/index
    mix-up is most likely to take, and it is completely silent.

Mutation control: T2_MUTATE=1 applies the documented fix at the pathology
site -- the argument handed to refined() becomes the ELEMENT index array
np.arange(n_elements) instead of the facet index array np.arange(n_facets) --
so the call no longer raises and the IndexError evidence disappears.  Re-run
with T2_MUTATE=1 python source.py.
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
    ne, nf, nv = m.t.shape[1], m.facets.shape[1], m.p.shape[1]
    print(f"elements={ne} facets={nf} vertices={nv}")
    print(f"more_facets_than_elements={nf > ne}")

    mask = np.zeros(ne, dtype=bool)
    mask[[0, 5, 17]] = True
    by_index = m.refined(np.nonzero(mask)[0])
    by_mask = m.refined(mask)
    print(f"refined_by_index_elements={by_index.t.shape[1]}")
    print(f"refined_by_mask_elements={by_mask.t.shape[1]}")
    print(f"index_and_mask_agree_on_count="
          f"{by_index.t.shape[1] == by_mask.t.shape[1]}")
    print(f"index_and_mask_agree_on_connectivity="
          f"{bool(np.array_equal(by_index.t, by_mask.t))}")
    print(f"index_and_mask_agree_on_points="
          f"{bool(np.allclose(by_index.p, by_mask.p))}")
    if not np.array_equal(by_index.t, by_mask.t):
        print("FAIL: the boolean mask did not reproduce the index form",
              file=sys.stderr)
        ok = False

    # --- facet indices ---------------------------------------------------
    # The pathology: FACET indices are handed to refined(), which reads its
    # argument as ELEMENT indices.  T2_MUTATE=1 applies the documented fix --
    # pass element indices -- at this one call.
    marked = np.arange(nf) if not MUTATE else np.arange(ne)
    err = None
    try:
        m.refined(marked)
    except IndexError as e:
        err = e
    print(f"facet_indices_raise_IndexError={isinstance(err, IndexError)}")
    print(f"facet_indices_message={str(err)!r}")
    print(f"message_says_out_of_bounds={'out of bounds' in str(err)}")
    print(f"message_names_axis_0={'axis 0' in str(err)}")
    print(f"message_names_axis_1={'axis 1' in str(err)}")
    print(f"message_quotes_the_element_count={f'size {ne}' in str(err)}")
    print(f"message_quotes_the_facet_count={f'size {nf}' in str(err)}")
    if err is None:
        print("FAIL: passing facet indices did not raise", file=sys.stderr)
        ok = False

    out = None
    try:
        m.refined(np.array([ne + 5]))
    except IndexError as e:
        out = e
    print(f"out_of_range_index_raises={isinstance(out, IndexError)}")
    print(f"out_of_range_message={str(out)!r}")
    if out is None:
        print("FAIL: an out-of-range element index did not raise",
              file=sys.stderr)
        ok = False

    # --- the silent mix-up: a SCALAR argument -----------------------------
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scalar = m.refined(3)
        msgs = sorted({str(c.message) for c in caught})
    print(f"scalar_argument_raises=False")
    print(f"scalar_argument_elements={scalar.t.shape[1]}")
    print(f"scalar_argument_is_uniform_refinement="
          f"{scalar.t.shape[1] == ne * 4 ** 3}")
    print(f"scalar_argument_warnings={msgs!r}")
    print(f"scalar_argument_is_silent={not msgs}")
    single = m.refined(np.array([3]))
    print(f"single_element_index_elements={single.t.shape[1]}")
    print(f"scalar_and_single_index_differ="
          f"{scalar.t.shape[1] != single.t.shape[1]}")
    if scalar.t.shape[1] != ne * 4 ** 3:
        print("FAIL: a scalar argument was not uniform refinement",
              file=sys.stderr)
        ok = False
    if msgs:
        print(f"FAIL: the scalar overload warned {msgs!r}", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
