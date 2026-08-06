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
  * the exact peak of the interpolated field is 1, and the sampled maximum
    moves monotonically towards it as the order rises -- which is what a
    correctly measured peak should do.
"""
from __future__ import annotations

import sys

import numpy
from netgen.geom2d import unit_square
from ngsolve import GridFunction, H1, Mesh, pi, sin, x, y


def main() -> int:
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    pts = [(float(a), float(b))
           for a in numpy.linspace(0.005, 0.995, 41)
           for b in numpy.linspace(0.005, 0.995, 41)]

    rows = []
    for k in (1, 2, 3):
        fes = H1(mesh, order=k)
        gf = GridFunction(fes)
        gf.Set(sin(pi * x) * sin(pi * y))
        vmax = float(gf.vec.FV().NumPy().max())
        fmax = max(float(gf(mesh(a, b))) for a, b in pts)
        rows.append((k, fes.ndof, vmax, fmax))
        print(f"order={k} ndof={fes.ndof} max_of_vec={vmax:.8f} "
              f"sampled_max={fmax:.8f} differ={abs(vmax - fmax) > 1e-6}")

    all_differ = all(abs(v - f) > 1e-6 for _, _, v, f in rows)
    print(f"the_two_numbers_differ_at_every_order={all_differ}")

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

    ok = (all_differ and rows[0][2] > 1.0 and rows[2][2] < 1.0
          and sampled_improves and abs(rows[2][3] - 1.0) < 0.01)
    if ok:
        return 0
    print("FAIL: max(vec) invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
