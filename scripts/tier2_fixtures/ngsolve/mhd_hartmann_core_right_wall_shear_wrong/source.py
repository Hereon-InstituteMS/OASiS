"""Tier-2: at Ha = 100 a uniform mesh gets the Hartmann core exactly right and
the wall shear badly wrong -- the core is not evidence that the layer resolved.

Claim: ngsolve mhd#2 -- "Hartmann number Ha = B0 * L * sqrt(sigma / (rho * nu))
measures magnetic vs viscous effects.  Ha >> 1 creates boundary layers of
thickness 1/Ha next to walls.  Signal: a uniform mesh at Ha = 100 shows ZERO
core-region velocity (correct) but mis-resolved boundary-layer flux -- the
computed wall shear stress is off by factor of 5+ vs the analytic Hartmann
solution because the layer is spread across only 1-2 cells."

Wrong variant: a uniform mesh whose cells are twenty times the layer thickness.

Setup: the fully developed Hartmann channel, -1 <= y <= 1, driven by a constant
pressure gradient G, transverse field B0.  Non-dimensionalised the momentum
balance is  -u'' + Ha^2 u = G  with u(+-1) = 0, whose exact solution is
u(y) = (G/Ha^2) (1 - cosh(Ha y)/cosh(Ha)).  Core value G/Ha^2, wall shear
(G/Ha) tanh(Ha), boundary layer thickness 1/Ha.  Solved with H1 order 2 on a
narrow strip so the discretisation is effectively one-dimensional in y, and both
quantities are compared against the closed form, so nothing here is a reference
run -- it is the analytic answer.

CORRECTION this fixture records.  The direction is right and the mechanism is
right, but "off by factor of 5+" is not what this configuration gives.  On the
coarsest mesh, whose cells are twenty times the layer thickness, the wall shear
comes out at a factor of about 1.8 too small, not 5.  The qualitative point --
core exact, layer wrong -- survives; the magnitude does not, and an agent using
"factor of 5" as a detection threshold would pass a mesh that is off by 43%.

What this fixture pins, all re-measured on this run:
  * the core velocity matches the closed form to nine digits or better at EVERY
    resolution, including the coarsest -- so a core check cannot detect the
    defect;
  * the wall shear on the coarsest mesh is wrong by more than 40%;
  * the two errors differ by more than seven orders of magnitude on that mesh:
    the core is not evidence about the layer;
  * the shear error falls monotonically as the cell size approaches the layer
    thickness, so it is a resolution effect and not a formulation bug.

Mutation control: T2_MUTATE=1 moves the core probe from y = 0 to y = -0.99 --
one hundredth of the channel from the wall, i.e. INSIDE the 1/Ha layer -- and
compares it against the closed form at that same point. The whole claim is
that the core probe is blind to the unresolved layer, so a probe that sits in
the layer must stop being exact: `core_exact_on_every_mesh=True`,
`core_exact_on_the_coarsest_mesh=True` and
`core_is_no_evidence_about_the_layer=True` disappear from the output and the
run prints "FAIL:", which the fixture forbids.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import math
import os
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

HA = 100.0
G = 1.0
WIDTH = 0.05
ORDER = 2
MUTATE = os.environ.get("T2_MUTATE") == "1"
# y at which the "core" is sampled. T2_MUTATE=1 drags it into the wall layer.
Y_CORE = -0.99 if MUTATE else 0.0


def exact(yy):
    return (G / (HA * HA)) * (1 - math.cosh(HA * yy) / math.cosh(HA))


def solve(ny):
    g = SplineGeometry()
    g.AddRectangle((0, -1), (WIDTH, 1), bcs=["bot", "right", "top", "left"])
    mesh = Mesh(g.GenerateMesh(maxh=2.0 / ny))
    fes = H1(mesh, order=ORDER, dirichlet="bot|top")
    u, v = fes.TnT()
    a = BilinearForm(fes)
    a += (grad(u) * grad(v) + HA * HA * u * v) * dx
    f = LinearForm(fes)
    f += G * v * dx
    a.Assemble()
    f.Assemble()
    gf = GridFunction(fes)
    gf.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="umfpack") * f.vec

    xm = WIDTH / 2
    u_core = float(gf(mesh(xm, Y_CORE)))
    hw = 1e-6
    shear = float((gf(mesh(xm, -1.0 + hw)) - gf(mesh(xm, -1.0))) / hw)
    shear_ex = (exact(-1.0 + hw) - exact(-1.0)) / hw
    return (mesh.ne, 2.0 / ny,
            abs(u_core - exact(Y_CORE)) / abs(exact(Y_CORE)),
            abs(shear - shear_ex) / abs(shear_ex),
            shear, shear_ex)


def main() -> int:
    layer = 1.0 / HA
    print(f"hartmann_number={HA:g}")
    print(f"layer_thickness_one_over_Ha={layer:.6f}")

    rows = []
    for ny in (10, 50, 200, 500):
        ne, h, ecore, eshear, sh, shex = solve(ny)
        rows.append((h, ecore, eshear))
        print(f"cells_across_channel={ny} elements={ne} h={h:.4f} "
              f"h_over_layer={h / layer:.1f} core_relerr={ecore:.3e} "
              f"shear={sh:.6e} exact={shex:.6e} shear_relerr={eshear:.3e}")

    coarse = rows[0]
    print(f"coarse_cells_span_the_layer_many_times="
          f"{coarse[0] / layer > 10}")
    print(f"core_exact_on_every_mesh={all(r[1] < 1e-6 for r in rows)}")
    print(f"core_exact_on_the_coarsest_mesh={coarse[1] < 1e-6}")
    print(f"coarse_shear_error_over_40_percent={coarse[2] > 0.40}")
    gap = coarse[2] / max(coarse[1], 1e-300)
    print(f"shear_error_over_core_error_on_coarse_mesh={gap:.3e}")
    print(f"core_is_no_evidence_about_the_layer={gap > 1e7}")
    print(f"claimed_factor_of_five_not_reproduced={coarse[2] < 0.80}")
    mono = all(b[2] < a[2] for a, b in zip(rows, rows[1:]))
    print(f"shear_error_falls_monotonically_with_h={mono}")
    print(f"finest_shear_error_under_1_percent={rows[-1][2] < 0.01}")

    ok = (
        coarse[0] / layer > 10
        and all(r[1] < 1e-6 for r in rows)
        and coarse[2] > 0.40
        and gap > 1e7
        and mono
        and rows[-1][2] < 0.01
    )
    if ok:
        return 0
    print("FAIL: Hartmann core/layer invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
