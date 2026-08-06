"""Tier-2: ib.zeros() one-hot versus unit_load -- and what unit_load holds.

Claim: skfem point_source#1 -- ib.zeros() returns a fresh float64 ndarray of
length ib.N and setting f[source_node] = 1.0 is the correct one-hot assembly
for a node-coincident source.  "Trying to use unit_load.assemble(ib) (the f=1
constant-source assembly) gives a 1.0 at every DOF -- different physics
entirely.  Signal: solution u has a uniformly-distributed shape with peaks at
all interior nodes, not a concentrated peak near the source location."

Measured on skfem 12.0.1, MeshTri().refined(4), ElementTriP1, 289 DOFs:

  * the ib.zeros() half is exactly right: float64, length ib.N, all zero, and
    a fresh array each call (mutating one does not affect the next).
  * "unit_load GIVES A 1.0 AT EVERY DOF" IS FALSE.  unit_load assembles
    int 1 * v dx, so each entry is the INTEGRAL of that basis function -- a
    spread of small numbers whose TOTAL is the domain area 1.0, not 1.0 each.
    Interior entries are larger than boundary ones because interior hat
    functions have more support.
  * "PEAKS AT ALL INTERIOR NODES" IS ALSO FALSE.  The unit_load solution is a
    single smooth bump with ONE interior maximum, at the centre -- the same
    place the point-source solution peaks.  What actually separates them is
    the peak VALUE and the spread, not the number of maxima.
  * exact reciprocity ties the two together: the unit-load solution evaluated
    at the source point equals the integral of the point-source solution, to
    several digits, because the Green's function is symmetric.  That is a far
    sharper check than counting peaks.
"""
from __future__ import annotations

import sys

import numpy as np
from skfem import Basis, ElementTriP1, MeshTri, condense, solve
from skfem.models.poisson import laplace, mass, unit_load


def main() -> int:
    ok = True
    m = MeshTri().refined(4)
    ib = Basis(m, ElementTriP1())
    print(f"dofs_N={ib.N} elements={m.t.shape[1]}")

    # --- ib.zeros() ------------------------------------------------------
    z = ib.zeros()
    print(f"zeros_dtype={z.dtype}")
    print(f"zeros_is_float64={z.dtype == np.float64}")
    print(f"zeros_length_equals_N={len(z) == ib.N}")
    print(f"zeros_all_zero={bool((z == 0.0).all())}")
    z[0] = 99.0
    print(f"zeros_is_a_fresh_array={bool((ib.zeros() == 0.0).all())}")
    if z.dtype != np.float64 or len(z) != ib.N:
        print("FAIL: ib.zeros() is not a float64 array of length ib.N",
              file=sys.stderr)
        ok = False

    # --- what unit_load actually contains --------------------------------
    ul = unit_load.assemble(ib)
    print(f"unit_load_min={ul.min():.6e} unit_load_max={ul.max():.6e}")
    print(f"unit_load_sum={ul.sum():.6f}")
    print(f"unit_load_is_one_at_every_dof={bool(np.allclose(ul, 1.0))}")
    print(f"unit_load_sums_to_domain_area={abs(ul.sum() - 1.0) < 1e-12}")
    print(f"unit_load_entries_are_basis_integrals="
          f"{bool(ul.max() < 0.01 and ul.min() > 0.0)}")
    if np.allclose(ul, 1.0):
        print("FAIL: unit_load really is 1.0 at every DOF", file=sys.stderr)
        ok = False
    if abs(ul.sum() - 1.0) >= 1e-12:
        print("FAIL: unit_load did not sum to the domain area",
              file=sys.stderr)
        ok = False

    # --- the two solutions ------------------------------------------------
    K = laplace.assemble(ib)
    M = mass.assemble(ib)
    D = ib.get_dofs().all()
    src = int(np.argmin((ib.doflocs[0] - 0.5) ** 2
                        + (ib.doflocs[1] - 0.5) ** 2))
    f_point = ib.zeros()
    f_point[src] = 1.0
    u_point = solve(*condense(K, f_point, D=D))
    u_load = solve(*condense(K, ul, D=D))
    print(f"point_peak={u_point.max():.6f}")
    print(f"unit_load_peak={u_load.max():.6f}")
    print(f"point_peak_exceeds_unit_load_peak="
          f"{u_point.max() > 5.0 * u_load.max()}")

    interior = np.setdiff1d(np.arange(ib.N), D)
    print(f"point_argmax_is_source_node="
          f"{int(np.argmax(u_point)) == src}")
    print(f"unit_load_argmax_is_also_the_centre="
          f"{int(np.argmax(u_load)) == src}")
    print(f"unit_load_has_peaks_at_all_interior_nodes="
          f"{int(np.sum(u_load[interior] >= u_load.max() - 1e-12))
             == len(interior)}")
    if int(np.argmax(u_load)) != src:
        print("FAIL: the unit-load solution did not peak at the centre",
              file=sys.stderr)
        ok = False
    if not u_point.max() > 5.0 * u_load.max():
        print("FAIL: the point-source peak was not sharply larger",
              file=sys.stderr)
        ok = False

    # --- reciprocity -------------------------------------------------------
    integral = float(np.ones(ib.N) @ (M @ u_point))
    at_source = float(u_load[src])
    print(f"integral_of_point_solution={integral:.8f}")
    print(f"unit_load_solution_at_source={at_source:.8f}")
    print(f"reciprocity_relative_gap="
          f"{abs(integral - at_source) / abs(at_source):.3e}")
    print(f"greens_function_reciprocity_holds="
          f"{abs(integral - at_source) / abs(at_source) < 1e-5}")
    if abs(integral - at_source) / abs(at_source) >= 1e-5:
        print("FAIL: reciprocity between the two loads did not hold",
              file=sys.stderr)
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
