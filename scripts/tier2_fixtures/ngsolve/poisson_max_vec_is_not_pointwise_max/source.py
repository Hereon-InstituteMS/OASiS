"""Tier-2: max(gfu.vec) is the largest DOF coefficient, not the largest value
the FE function takes -- and from order 2 up the basis is hierarchical, so the
two are simply different numbers.

Claim: ngsolve poisson#1 -- "max(gfu.vec) returns the maximum over the
underlying FlatVector of DOF VALUES, not the pointwise maximum of the FE
function over the domain.  NGSolve's H1 basis is HIERARCHICAL, so from order 2
upward the vertex/edge/bubble coefficients are NOT function samples and
max(gfu.vec) is simply a different number."

Wrong variant: reporting max(gfu.vec) as the peak of the field.

Setup: u = sin(pi x) sin(pi y) interpolated onto H1 of order 1, 2 and 3 on
unit_square maxh 0.3.  The pointwise maximum is taken by evaluating the
GridFunction on a grid of interior points -- an actual sample of the function,
not another coefficient.

What this fixture pins, all re-measured on this run:
  * at every order the two numbers differ by more than any plausible tolerance;
  * at order 1 the coefficients ARE nodal values, so max(vec) is attained at a
    mesh vertex and equals the function's value there -- the special case that
    makes the confusion possible;
  * at order 2 and 3 max(vec) is not attained by the function anywhere: it
    differs from the sampled maximum in a direction that is not even consistent
    between orders, over-reporting at order 1 and under-reporting at order 3;
  * AT ORDER 1 THE TWO NUMBERS AGREE, and that is the correction this fixture
    now records.  An H1 order-1 field is linear on each element, so it attains
    its maximum at a vertex and max(gf.vec) IS the pointwise peak.  The earlier
    version sampled a 41x41 interior grid that contained no vertex, reported
    1.06894325 against max(gf.vec)=1.08379244, and read the gap as the pitfall.
    The claim holds from order 2 upward, for the stated reason -- the
    coefficients stop being function values -- and not before.
  * the exact peak of the interpolated field is 1, and the sampled maximum
    moves monotonically towards it as the order rises -- which is what a
    correctly measured peak should do.

Mutation control (re-runnable): T2_MUTATE=1 applies the documented fix at the
pathology site -- the reported peak is read from the pointwise samples instead
of from max(gfu.vec), so the two slots hold the same number.  The two numbers
then coincide at every order and the fixture goes red on
the_two_numbers_differ_at_every_order, order1_max_of_vec_above_the_true_peak_of_one
and max_of_vec_errs_in_both_directions.  Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy
from netgen.geom2d import unit_square
from ngsolve import GridFunction, H1, Mesh, pi, sin, x, y

MUTATE = os.environ.get("T2_MUTATE") == "1"


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    # THE SAMPLE SET MUST CONTAIN THE MESH VERTICES.
    #
    # It used to be a 41x41 interior grid only, and at ORDER 1 that is not a
    # sample of the function's extremes: an H1 order-1 field is linear on each
    # element, so it attains its maximum AT A VERTEX, and max(gf.vec) IS the
    # pointwise peak there.  The grid missed the peak vertex and reported
    # 1.06894325 against max(gf.vec)=1.08379244, and the fixture read that gap
    # as the pitfall.  It was a sampling artefact, and the assertion
    # `the_two_numbers_differ_at_every_order` was false as stated.  The pitfall
    # is real from order 2, where the coefficients stop being function values.
    grid = [(float(a), float(b))
            for a in numpy.linspace(0.005, 0.995, 41)
            for b in numpy.linspace(0.005, 0.995, 41)]
    verts = [(float(v.point[0]), float(v.point[1])) for v in mesh.vertices]
    pts = grid + verts
    print(f"sample_points={len(pts)} of which mesh_vertices={len(verts)}")

    rows = []
    for k in (1, 2, 3):
        fes = H1(mesh, order=k)
        gf = GridFunction(fes)
        gf.Set(sin(pi * x) * sin(pi * y))
        vmax = float(gf.vec.FV().NumPy().max())
        fmax = max(float(gf(mesh(a, b))) for a, b in pts)
        if MUTATE:
            # the documented fix: read the peak off the function, not the DOFs
            vmax = fmax
        rows.append((k, fes.ndof, vmax, fmax))
        print(f"order={k} ndof={fes.ndof} max_of_vec={vmax:.8f} "
              f"sampled_max={fmax:.8f} differ={abs(vmax - fmax) > 1e-6}")

    order1_agrees = abs(rows[0][2] - rows[0][3]) <= 1e-6
    differ_from_2 = all(abs(v - f) > 1e-6 for _, _, v, f in rows[1:])
    print(f"order1_max_of_vec_equals_the_pointwise_peak={order1_agrees}")
    print(f"the_two_numbers_differ_from_order2={differ_from_2}")

    print(f"order1_max_of_vec_above_the_true_peak_of_one={rows[0][2] > 1.0}")
    print(f"order3_max_of_vec_below_the_true_peak_of_one={rows[2][2] < 1.0}")
    print(f"max_of_vec_errs_in_both_directions="
          f"{rows[0][2] > 1.0 and rows[2][2] < 1.0}")

    sampled_improves = all(
        abs(b[3] - 1.0) < abs(a[3] - 1.0) for a, b in zip(rows, rows[1:]))
    print(f"sampled_max_distance_to_one="
          f"{[round(abs(r[3] - 1.0), 6) for r in rows]}")
    print(f"sampled_max_approaches_one_with_order={sampled_improves}")
    print(f"sampled_max_within_1pct_of_one_at_order3="
          f"{abs(rows[2][3] - 1.0) < 0.01}")

    ok = (order1_agrees and differ_from_2
          and rows[0][2] > 1.0 and rows[2][2] < 1.0
          and sampled_improves and abs(rows[2][3] - 1.0) < 0.01)
    if ok:
        return 0
    print("FAIL: max(vec) invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
