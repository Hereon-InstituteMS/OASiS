"""Tier-2: p[:n_vertices] silently truncates a non-P1 pressure field.

Claim: skfem hydraulic_resistance#6 -- the pressure block starts at index
ib_u.N; extracting `p_nodal = p[:m.p.shape[1]]` assumes the P1 pressure DOFs
match the mesh vertices, which is the skfem default for ElementTriP1.  With a
different element (e.g. ElementTriP2 for pressure) the DOF count exceeds the
vertex count and this slice drops data.

Measured on skfem 12.0.1, MeshTri().refined(3):

  * for ElementTriP1 the slice is exactly the whole field -- the DOF count
    equals the vertex count and doflocs equals mesh.p -- so the idiom is
    correct and stays correct.
  * for ElementTriP2 the slice keeps only the vertex share and drops the
    edge DOFs.  Crucially the FIRST n_vertices doflocs still coincide with
    mesh.p, so the truncated field is a plausible vertex field: it plots,
    it has the right length for the mesh, and nothing raises or warns.
  * THE ENTRY'S SYMPTOM IS UNRELIABLE.  It says the printed max/min will
    disagree with what ParaView renders.  For a smooth pressure field the
    extremes land on vertices, so the truncated slice has EXACTLY the same
    min and max as the full field and the check sees nothing.  Only when the
    extremes sit on edge midpoints -- the P2-only DOFs the slice discards --
    does the range move.  So the range check is field-dependent and cannot
    be relied on.
  * the guard that always works is a length comparison, basis.N against
    mesh.p.shape[1], done BEFORE slicing.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
from skfem import Basis, ElementTriP1, ElementTriP2, MeshTri


def field(x):
    return np.sin(2.0 * np.pi * x[0]) * np.cos(2.0 * np.pi * x[1])


def main() -> int:
    ok = True
    m = MeshTri().refined(3)
    nverts = m.p.shape[1]
    print(f"mesh_vertices={nverts}")

    results = {}
    for tag, elem in (("P1", ElementTriP1()), ("P2", ElementTriP2())):
        bp = Basis(m, elem)
        p = field(bp.doflocs)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sliced = p[:nverts]
            msgs = sorted({str(c.message) for c in caught})
        dropped = bp.N - nverts
        prefix_ok = bool(np.allclose(bp.doflocs[:, :nverts], m.p))
        results[tag] = (bp, p, sliced, dropped)
        print(f"{tag}_dofs={bp.N} vertices={nverts} dropped={dropped}")
        print(f"{tag}_dof_count_equals_vertex_count={bp.N == nverts}")
        print(f"{tag}_doflocs_prefix_equals_mesh_p={prefix_ok}")
        print(f"{tag}_slice_length={len(sliced)}")
        print(f"{tag}_slice_warnings={msgs!r}")
        print(f"{tag}_slice_is_silent={not msgs}")
        if msgs:
            print(f"FAIL: {tag} slicing warned {msgs!r}", file=sys.stderr)
            ok = False
        if not prefix_ok:
            print(f"FAIL: {tag} doflocs prefix does not equal mesh.p, so the "
                  f"truncated field would not look plausible", file=sys.stderr)
            ok = False

    bp1, p1, s1, d1 = results["P1"]
    bp2, p2, s2, d2 = results["P2"]
    print(f"p1_slice_is_the_whole_field={d1 == 0 and len(s1) == bp1.N}")
    print(f"p2_slice_drops_data={d2 > 0}")
    print(f"p2_dropped_fraction={d2 / bp2.N:.4f}")
    if d1 != 0:
        print("FAIL: the P1 slice dropped data", file=sys.stderr)
        ok = False
    if d2 <= 0:
        print("FAIL: the P2 slice dropped nothing", file=sys.stderr)
        ok = False

    # --- the truncation is invisible by shape, visible by range ----------
    print(f"p2_full_min={p2.min():.6f} p2_full_max={p2.max():.6f}")
    print(f"p2_slice_min={s2.min():.6f} p2_slice_max={s2.max():.6f}")
    same_range = (abs(p2.min() - s2.min()) < 1e-9
                  and abs(p2.max() - s2.max()) < 1e-9)
    print(f"p2_slice_has_the_same_range={same_range}")
    print(f"p2_slice_length_matches_the_mesh={len(s2) == nverts}")
    print(f"truncation_is_invisible_by_length={len(s2) == nverts}")
    print(f"smooth_field_range_check_detects_truncation={not same_range}")
    if not same_range:
        print("FAIL: the range check DID detect the truncation on a smooth "
              "field, so the entry's symptom is reliable after all",
              file=sys.stderr)
        ok = False

    # A field whose extremes sit on EDGE midpoints -- the P2-only DOFs the
    # slice throws away.  Now, and only now, the range does move.
    edge_only = np.zeros(bp2.N)
    edge_only[nverts:] = 1.0
    sliced_edge = edge_only[:nverts]
    print(f"edge_peaked_full_max={edge_only.max():.6f}")
    print(f"edge_peaked_slice_max={sliced_edge.max():.6f}")
    print(f"edge_peaked_range_check_detects_truncation="
          f"{abs(edge_only.max() - sliced_edge.max()) > 1e-9}")
    print(f"range_check_is_field_dependent="
          f"{same_range and abs(edge_only.max() - sliced_edge.max()) > 1e-9}")
    if not abs(edge_only.max() - sliced_edge.max()) > 1e-9:
        print("FAIL: even an edge-peaked field did not move the range",
              file=sys.stderr)
        ok = False

    print(f"length_guard_fires_for_p2={bp2.N != nverts}")
    print(f"length_guard_quiet_for_p1={bp1.N == nverts}")
    if not (bp2.N != nverts and bp1.N == nverts):
        print("FAIL: the length guard does not discriminate", file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
