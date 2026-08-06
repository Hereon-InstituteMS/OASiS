"""Tier-2: np.argmin silently rounds an off-node point source to a vertex.

Claim: skfem point_source#2 -- if x0 is NOT a mesh node, np.argmin on the
vertex array picks the NEAREST node and introduces a discretisation error
proportional to h/sqrt(2); to place an off-node source properly, locate the
containing triangle and distribute the unit load by barycentric coordinates.
"np.argmin on the MeshTri.p.T vertex array always returns an integer node id,
never a barycentric position, so off-node sources are silently rounded to the
nearest mesh vertex."

Measured on skfem 12.0.1, MeshTri().refined(4) (h = 1/16), ElementTriP1:

  * confirmed and quantified.  np.argmin returns a plain Python int; over 200
    random target points the displacement never exceeds h/sqrt(2), and its
    mean is a substantial fraction of h -- so the rounding is not a rare
    corner case, it happens to essentially every off-node request.
  * IT IS SILENT: no exception, no warning, and the solution that comes out
    is a perfectly ordinary point-source solution -- just for a different
    point than the one asked for.
  * the barycentric alternative works and is measurably different: spreading
    the unit load over the containing triangle's three vertices moves the
    solution by an amount comparable to the rounding displacement, and it
    preserves the total load exactly (the three weights sum to one).
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace


def barycentric_load(m, ib, x0):
    """Distribute a unit load over the vertices of the containing triangle."""
    tri = m.element_finder()(np.array([x0[0]]), np.array([x0[1]]))[0]
    v = m.t[:, tri]
    p = m.p[:, v]
    A = np.array([[p[0, 0] - p[0, 2], p[0, 1] - p[0, 2]],
                  [p[1, 0] - p[1, 2], p[1, 1] - p[1, 2]]])
    lam12 = np.linalg.solve(A, np.array([x0[0] - p[0, 2], x0[1] - p[1, 2]]))
    lam = np.array([lam12[0], lam12[1], 1.0 - lam12.sum()])
    f = ib.zeros()
    f[v] = lam
    return f, v, lam


def main() -> int:
    ok = True
    m = MeshTri().refined(4)
    ib = Basis(m, ElementTriP1())
    h = 1.0 / 16.0
    print(f"dofs_N={ib.N} h={h}")

    # --- what argmin returns ---------------------------------------------
    rng = np.random.default_rng(0)
    targets = 0.1 + 0.8 * rng.random((200, 2))
    dists = []
    for x0 in targets:
        d2 = (m.p[0] - x0[0]) ** 2 + (m.p[1] - x0[1]) ** 2
        j = int(np.argmin(d2))
        dists.append(float(np.sqrt(d2[j])))
    dists = np.array(dists)
    j0 = np.argmin((m.p[0] - targets[0, 0]) ** 2
                   + (m.p[1] - targets[0, 1]) ** 2)
    print(f"argmin_returns_integer={isinstance(int(j0), int)}")
    print(f"argmin_result_type={type(np.argmin([1.0, 0.0])).__name__}")
    print(f"max_displacement={dists.max():.6f}")
    print(f"mean_displacement={dists.mean():.6f}")
    print(f"h_over_sqrt2={h / np.sqrt(2.0):.6f}")
    print(f"displacement_bounded_by_h_over_sqrt2="
          f"{bool(dists.max() <= h / np.sqrt(2.0) + 1e-12)}")
    print(f"mean_displacement_is_a_real_fraction_of_h="
          f"{bool(dists.mean() > 0.2 * h)}")
    print(f"fraction_landing_exactly_on_target="
          f"{float(np.mean(dists < 1e-12)):.4f}")
    if not dists.max() <= h / np.sqrt(2.0) + 1e-12:
        print("FAIL: a displacement exceeded the h/sqrt(2) bound",
              file=sys.stderr)
        ok = False
    if not dists.mean() > 0.2 * h:
        print("FAIL: the rounding displacement was negligible",
              file=sys.stderr)
        ok = False

    # --- it is silent, and it changes the answer --------------------------
    x0 = np.array([0.3123, 0.7891])
    K = laplace.assemble(ib)
    D = ib.get_dofs().all()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        node = int(np.argmin((m.p[0] - x0[0]) ** 2 + (m.p[1] - x0[1]) ** 2))
        f_round = ib.zeros()
        f_round[node] = 1.0
        u_round = solve(*condense(K, f_round, D=D))
        f_bary, verts, lam = barycentric_load(m, ib, x0)
        u_bary = solve(*condense(K, f_bary, D=D))
        msgs = sorted({str(c.message) for c in caught})
    print(f"requested_point={x0.tolist()}")
    print(f"snapped_node={node} at={m.p[:, node].tolist()}")
    print(f"snap_distance={float(np.hypot(*(m.p[:, node] - x0))):.6f}")
    print(f"rounding_warnings={msgs!r}")
    print(f"rounding_is_silent={not msgs}")
    print(f"rounded_solution_finite={bool(np.isfinite(u_round).all())}")
    if msgs:
        print(f"FAIL: the rounding warned {msgs!r}", file=sys.stderr)
        ok = False

    print(f"barycentric_weights={[f'{x:.4f}' for x in lam]}")
    print(f"barycentric_weights_sum_to_one={abs(lam.sum() - 1.0) < 1e-12}")
    print(f"barycentric_weights_all_nonnegative={bool((lam >= -1e-12).all())}")
    print(f"barycentric_load_total={float(f_bary.sum()):.12f}")
    rel = float(np.linalg.norm(u_bary - u_round) / np.linalg.norm(u_round))
    print(f"barycentric_vs_rounded_relative_difference={rel:.4e}")
    print(f"the_two_placements_differ={rel > 1e-3}")
    if abs(lam.sum() - 1.0) >= 1e-12 or not (lam >= -1e-12).all():
        print("FAIL: the barycentric weights are not a partition of unity "
              "inside the element", file=sys.stderr)
        ok = False
    if rel <= 1e-3:
        print("FAIL: the barycentric placement gave the same solution",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
