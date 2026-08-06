"""Tier-2: MatrixValued(..., symmetric=True) stores only the upper triangle, and
the DOF count proves it.

Claim: ngsolve linear_elasticity#3 -- "Stress visualization uses
MatrixValued(H1(mesh, order=k), symmetric=True).  The 'symmetric=True' kwarg
packs only the upper triangle into the dof vector (ndof reduced from
d^2*scalar_ndof to d*(d+1)/2*scalar_ndof).  Without symmetric=True the
MatrixValued space has the full d^2 storage."

Wrong variant: omitting symmetric=True and paying for d^2 storage, or assuming
the symmetric space can hold a non-symmetric tensor.

What this fixture pins, all re-measured on this run:
  * the DOF counts are exactly d^2 and d(d+1)/2 times the scalar space's, in
    both 2D and 3D, so the packing claim is arithmetic and not an estimate;
  * a GridFunction on the symmetric space cannot represent a non-symmetric
    tensor: interpolating one gives back a symmetric field, and the part that
    was lost is exactly the skew part -- measured, not asserted;
  * the full space CAN represent it, reproducing the same tensor to roundoff;
  * a genuinely symmetric stress field is represented identically by both
    spaces, so the saving costs nothing when the field really is symmetric.

Mutation control (INVERTED direction): this fixture executes no pathology -- it
only demonstrates the correct spelling -- so there is nothing to remove.
``T2_MUTATE=1 python source.py`` instead COMMITS the documented mistake at the
site the pitfall is about: SYMMETRIC_PACKING becomes False, i.e. the "symmetric"
space is built as MatrixValued(H1(...), symmetric=False) with the kwarg omitted
in effect.  Its ndof then equals the full d^2 count and it can hold the skew
part, so ``2d_sym_is_half_d_dplus1_times_scalar=True``,
``3d_sym_is_half_d_dplus1_times_scalar=True`` and
``symmetric_space_cannot_hold_the_skew_part=True`` disappear and the fixture
goes red.
"""
from __future__ import annotations

import os
import sys

from netgen.csg import unit_cube
from netgen.geom2d import unit_square
from ngsolve import (
    CoefficientFunction,
    GridFunction,
    H1,
    Integrate,
    InnerProduct,
    MatrixValued,
    Mesh,
    sqrt,
    x,
    y,
)

MUTATE = os.environ.get("T2_MUTATE") == "1"

# The kwarg the pitfall is about.  Under T2_MUTATE it is dropped to False.
SYMMETRIC_PACKING = not MUTATE


def main() -> int:
    for label, mesh, d in (("2d", Mesh(unit_square.GenerateMesh(maxh=0.3)), 2),
                           ("3d", Mesh(unit_cube.GenerateMesh(maxh=0.5)), 3)):
        scalar = H1(mesh, order=1)
        full = MatrixValued(H1(mesh, order=1))
        sym = MatrixValued(H1(mesh, order=1), symmetric=SYMMETRIC_PACKING)
        exp_full = d * d * scalar.ndof
        exp_sym = d * (d + 1) // 2 * scalar.ndof
        print(f"{label}_scalar_ndof={scalar.ndof} full_ndof={full.ndof} "
              f"sym_ndof={sym.ndof}")
        print(f"{label}_full_is_d2_times_scalar={full.ndof == exp_full}")
        print(f"{label}_sym_is_half_d_dplus1_times_scalar={sym.ndof == exp_sym}")
        print(f"{label}_space_type={type(sym).__name__}")

    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    full = MatrixValued(H1(mesh, order=1))
    sym = MatrixValued(H1(mesh, order=1), symmetric=SYMMETRIC_PACKING)

    # A deliberately NON-symmetric tensor.
    T = CoefficientFunction((x, 2 * y, -y, x + y), dims=(2, 2))
    skew = 0.5 * (T - T.trans)
    skew_norm = float(sqrt(Integrate(InnerProduct(skew, skew), mesh)))
    print(f"input_tensor_skew_part_norm={skew_norm:.6f}")
    print(f"input_tensor_is_not_symmetric={skew_norm > 1e-6}")

    gs = GridFunction(sym)
    gs.Set(T)
    gs_skew = float(sqrt(Integrate(
        InnerProduct(0.5 * (gs - gs.trans), 0.5 * (gs - gs.trans)), mesh)))
    print(f"symmetric_space_result_skew_norm={gs_skew:.3e}")
    print(f"symmetric_space_cannot_hold_the_skew_part={gs_skew < 1e-12}")

    gf = GridFunction(full)
    gf.Set(T)
    gf_err = float(sqrt(Integrate(InnerProduct(gf - T, gf - T), mesh)))
    print(f"full_space_reproduction_error={gf_err:.3e}")
    print(f"full_space_can_hold_it={gf_err < 1e-10}")

    # A symmetric field: both spaces agree.
    S = CoefficientFunction((x, y, y, x + y), dims=(2, 2))
    g1 = GridFunction(sym)
    g1.Set(S)
    g2 = GridFunction(full)
    g2.Set(S)
    agree = float(sqrt(Integrate(InnerProduct(g1 - g2, g1 - g2), mesh)))
    print(f"symmetric_field_difference_between_spaces={agree:.3e}")
    print(f"no_loss_when_the_field_really_is_symmetric={agree < 1e-10}")

    mesh3 = Mesh(unit_cube.GenerateMesh(maxh=0.5))
    s3 = H1(mesh3, order=1)
    ok = (
        MatrixValued(H1(mesh, order=1)).ndof == 4 * H1(mesh, order=1).ndof
        and MatrixValued(H1(mesh, order=1), symmetric=SYMMETRIC_PACKING).ndof
        == 3 * H1(mesh, order=1).ndof
        and MatrixValued(s3).ndof == 9 * s3.ndof
        and MatrixValued(s3, symmetric=SYMMETRIC_PACKING).ndof == 6 * s3.ndof
        and skew_norm > 1e-6 and gs_skew < 1e-12
        and gf_err < 1e-10 and agree < 1e-10
    )
    if ok:
        return 0
    print("FAIL: MatrixValued symmetric invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
