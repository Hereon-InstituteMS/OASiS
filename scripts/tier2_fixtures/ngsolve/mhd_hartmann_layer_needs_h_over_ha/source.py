"""Tier-2: the Hartmann wall integral needs a cell size that scales like 1/Ha,
and the claim's h_wall = L/(10*Ha) rule does land near a percent.

Claim: ngsolve mhd#3 -- "Hartmann layers need mesh refinement near walls
proportional to 1/Ha.  Signal: an unrefined mesh gives core-velocity accuracy
O(1) but boundary-layer wall shear off by 5-10x at Ha=100; geometric grading
with first-cell height h_wall = L / (10*Ha) restores ~1% accuracy on the wall
integrals."

Wrong variant: a cell size fixed in absolute terms while Ha rises, so the layer
gets thinner and the mesh does not.

Setup: the same closed-form Hartmann channel as mhd#2 -- -u'' + Ha^2 u = G on
-1 <= y <= 1 with u(+-1) = 0, exact solution (G/Ha^2)(1 - cosh(Ha y)/cosh(Ha)),
wall shear (G/Ha) tanh(Ha).  What is measured here is the SCALING, not one
mesh: at three Hartmann numbers, one cell size held fixed in absolute terms and
one held fixed relative to the layer.

CORRECTION this fixture records.  The claim's stated magnitude ("off by 5-10x")
does not appear.  At Ha = 100 with cells twenty times the layer thickness the
wall shear is a factor of about 1.8 too small, and the deficit is bounded: the
computed shear cannot fall below zero, so "5-10x" is not the shape of this
error at all.  The claim's REFINEMENT RULE, on the other hand, holds: at
L/(10*Ha) the wall integral is inside a percent at every Ha tried.

What this fixture pins, all re-measured on this run:
  * with a cell size fixed in absolute terms, the wall-shear error GROWS with
    Ha -- the layer outruns the mesh, which is what "proportional to 1/Ha"
    means;
  * with the cell size held at the claim's L/(10*Ha), the error is under a
    percent at every Ha tried and does NOT grow with Ha;
  * the core velocity is accurate in every one of those runs, so the core again
    tells you nothing;
  * on the absolute-cell-size runs the shear is too SMALL, never too large, so
    the error is one-sided and bounded by the exact value.
"""
from __future__ import annotations

import math
import sys

from netgen.geom2d import SplineGeometry
from ngsolve import (
    BilinearForm,
    GridFunction,
    H1,
    LinearForm,
    Mesh,
    dx,
    grad,
)

G = 1.0
WIDTH = 0.05
ORDER = 2
LENGTH = 2.0


def exact(Ha, yy):
    return (G / (Ha * Ha)) * (1 - math.cosh(Ha * yy) / math.cosh(Ha))


def solve(Ha, h):
    g = SplineGeometry()
    g.AddRectangle((0, -1), (WIDTH, 1), bcs=["bot", "right", "top", "left"])
    mesh = Mesh(g.GenerateMesh(maxh=h))
    fes = H1(mesh, order=ORDER, dirichlet="bot|top")
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += (grad(u) * grad(v) + Ha * Ha * u * v) * dx
    f = LinearForm(fes)
    f += G * v * dx
    a.Assemble()
    f.Assemble()
    gf = GridFunction(fes)
    gf.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="umfpack") * f.vec
    xm = WIDTH / 2
    hw = 1e-6
    core = float(gf(mesh(xm, 0.0)))
    shear = float((gf(mesh(xm, -1.0 + hw)) - gf(mesh(xm, -1.0))) / hw)
    shear_ex = (exact(Ha, -1.0 + hw) - exact(Ha, -1.0)) / hw
    return (mesh.ne,
            abs(core - exact(Ha, 0.0)) / abs(exact(Ha, 0.0)),
            (shear - shear_ex) / abs(shear_ex),
            shear, shear_ex)


def main() -> int:
    has = (10.0, 30.0, 100.0)
    fixed_h = 0.1

    fixed_rows, scaled_rows = [], []
    for Ha in has:
        ne, ec, es, sh, shex = solve(Ha, fixed_h)
        fixed_rows.append((Ha, ec, es))
        print(f"fixed_h Ha={Ha:g} elements={ne} h={fixed_h} "
              f"h_over_layer={fixed_h * Ha:.1f} core_relerr={ec:.3e} "
              f"shear_signed_relerr={es:+.4e}")
    for Ha in has:
        h = LENGTH / (10.0 * Ha)
        ne, ec, es, sh, shex = solve(Ha, h)
        scaled_rows.append((Ha, ec, es))
        print(f"scaled_h Ha={Ha:g} elements={ne} h={h:.5f} "
              f"h_over_layer={h * Ha:.2f} core_relerr={ec:.3e} "
              f"shear_signed_relerr={es:+.4e}")

    fixed_grows = all(abs(b[2]) > abs(a[2])
                      for a, b in zip(fixed_rows, fixed_rows[1:]))
    print(f"fixed_h_shear_error_grows_with_Ha={fixed_grows}")
    print(f"fixed_h_worst_shear_error={max(abs(r[2]) for r in fixed_rows):.4f}")
    print(f"fixed_h_shear_always_too_small="
          f"{all(r[2] < 0 for r in fixed_rows)}")

    scaled_ok = all(abs(r[2]) < 0.01 for r in scaled_rows)
    print(f"scaled_h_worst_shear_error="
          f"{max(abs(r[2]) for r in scaled_rows):.6f}")
    print(f"L_over_10Ha_rule_gives_under_one_percent={scaled_ok}")
    scaled_flat = abs(scaled_rows[-1][2]) < 3 * abs(scaled_rows[0][2]) + 1e-6
    print(f"scaled_h_error_does_not_grow_with_Ha={scaled_flat}")

    core_ok = all(r[1] < 1e-5 for r in fixed_rows + scaled_rows)
    print(f"core_accurate_in_every_run={core_ok}")
    print(f"claimed_5_to_10x_deficit_not_reproduced="
          f"{max(abs(r[2]) for r in fixed_rows) < 0.80}")

    ok = fixed_grows and scaled_ok and scaled_flat and core_ok and \
        all(r[2] < 0 for r in fixed_rows)
    if ok:
        return 0
    print("FAIL: Hartmann refinement-scaling invariant not held",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
