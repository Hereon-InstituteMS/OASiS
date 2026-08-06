"""Tier-2: DG DOFs are not nodal -- writing them straight to VTK is wrong.

Claim: skfem dg_methods#7 -- use Basis.project(Basis.interpolator(u)) for nodal
post-processing.  Visualizing the ElementTriDG solution directly with the wrong
VTK writer produces an all-zero or step-pattern field because DG DOFs are not
nodal; projecting to ElementTriP1 first restores a smooth visualization.

Wrong variant: treat the DG coefficient vector as if it were nodal -- hand it
to meshio as ``point_data``, and (the version that does not even raise) slice
its first ``n_vertices`` entries and call them vertex values.

Test field: f(x, y) = x + 2y, represented EXACTLY in DG-P1, so the correct
vertex values are known to machine precision and any error is pure
misinterpretation of the DOF layout.

Observed on skfem 12.0.1, MeshTri().refined(2) (32 elements, 25 vertices):
  * len(u_dg) = 96 = 3 * 32, not 25 -- meshio refuses the array with
    ValueError: len(points) = 25, but len(point_data["u"]) = 96
  * truncating to the first 25 entries gives max|err| = 2.25 (the field only
    spans [0, 3], so that is 75% of the range)
  * ib_p1.project(ib_dg.interpolator(u_dg)) gives max|err| = 1.3e-15
  * the DG DOF locations repeat: 25 vertices carry 96 DOFs, so 71 of them are
    duplicate coordinates -- that is why a nodal reading collapses.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import Basis, ElementDG, ElementTriP1, MeshTri
import meshio


def field(x):
    return x[0] + 2.0 * x[1]


def main() -> int:
    ok = True
    m = MeshTri().refined(2)
    nvert = int(m.p.shape[1])
    e_dg = ElementDG(ElementTriP1())
    ib_dg = Basis(m, e_dg)
    ib_p1 = Basis(m, ElementTriP1())

    u_dg = ib_dg.project(field)
    exact = field(m.p)

    print(f"n_elements={m.nelements}")
    print(f"n_vertices={nvert}")
    print(f"n_dg_dofs={len(u_dg)}")
    print(f"dg_dofs_equal_3_times_elements={len(u_dg) == 3 * m.nelements}")
    print(f"dg_dofs_match_vertex_count={len(u_dg) == nvert}")

    # DG DOF coordinates are duplicated -- they are element-local, not nodal.
    n_unique = int(np.unique(np.round(ib_dg.doflocs.T, 12), axis=0).shape[0])
    print(f"n_unique_dg_dof_locations={n_unique}")
    print(f"dg_dof_locations_are_duplicated={n_unique < len(u_dg)}")
    if n_unique >= len(u_dg):
        print("FAIL: DG DOF locations were all distinct -- layout changed",
              file=sys.stderr)
        ok = False

    # --- WRONG variant 1: hand the DG vector to the VTK writer -------------
    points = np.column_stack([m.p.T, np.zeros(nvert)])
    try:
        meshio.Mesh(points, [("triangle", m.t.T)], point_data={"u": u_dg})
    except ValueError as exc:
        vtk_msg = str(exc)
    else:
        vtk_msg = ""
    print(f"vtk_point_data_rejected={bool(vtk_msg)}")
    print(f"vtk_message={vtk_msg!r}")
    if 'len(point_data["u"]) = 96' not in vtk_msg:
        print("FAIL: meshio accepted DG DOFs as point_data", file=sys.stderr)
        ok = False

    # --- WRONG variant 2: silently truncate to the first n_vertices --------
    naive = np.asarray(u_dg)[:nvert]
    err_naive = float(np.abs(naive - exact).max())
    span = float(exact.max() - exact.min())
    print(f"naive_truncation_runs_without_error=True")
    print(f"naive_truncation_err_gt_half_of_range={err_naive > 0.5 * span}")
    print(f"naive_truncation_err={err_naive:.4f} field_span={span:.4f}")
    if err_naive <= 0.5 * span:
        print("FAIL: reading DG DOFs as nodal values was accurate", file=sys.stderr)
        ok = False

    # --- RIGHT variant: Basis.project(Basis.interpolator(u)) --------------
    u_p1 = ib_p1.project(ib_dg.interpolator(u_dg))
    err_proj = float(np.abs(u_p1 - exact).max())
    print(f"projected_length_matches_vertices={len(u_p1) == nvert}")
    print(f"projected_err_lt_1e-12={err_proj < 1e-12}")
    print(f"projected_err={err_proj:.3e}")
    if len(u_p1) != nvert or err_proj >= 1e-12:
        print("FAIL: the documented projection did not recover the vertex values",
              file=sys.stderr)
        ok = False

    # the projected vector IS writable as point_data
    meshio.Mesh(points, [("triangle", m.t.T)], point_data={"u": u_p1})
    print("projected_accepted_by_vtk_writer=True")

    print(f"projection_beats_truncation_by_gt_1e10="
          f"{err_naive / max(err_proj, 1e-300) > 1e10}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
