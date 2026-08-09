"""Tier-2: the +p*div(v) convention returns MINUS the physical pressure -- and
the check needs no second solver, only Poiseuille's analytic gradient.

Claim: ngsolve stokes#5 -- "NGSolve Stokes uses +p*div(v) convention (opposite
sign from FEniCS / skfem).  Both are valid weak forms.  Signal: a Poiseuille-flow
benchmark solved in NGSolve and FEniCS gives pressure GridFunction values that
differ by a sign at every node (max(p_ngsolve) approximately -max(p_fenics)).
Be aware when comparing cross-solver results."

Wrong variant: reading the pressure out of the +p*div(v) form as if it were the
physical pressure.

TWO CORRECTIONS this fixture records.

  (a) The Signal is written as a cross-solver comparison, so it can only be
      checked by anyone who has FEniCS installed as well.  It does not need to
      be: plane Poiseuille has an analytic pressure gradient,
      dp/dx = -8*nu*Umax/H^2, and the sign can be settled against that.  This
      fixture does so and needs one solver.
  (b) "Both are valid weak forms" is true of the VELOCITY and misleading about
      the pressure.  The two conventions give velocities that agree, but only
      the -p*div(v) form returns the physical pressure; the +p*div(v) form
      returns its negative.  An agent that reports that field as "the pressure"
      reports the wrong sign of a physical quantity, silently.

Setup: plane Poiseuille in a 4 x 1 channel, nu = 1, parabolic inlet with peak 1,
no-slip walls, traction-free outlet.  Taylor-Hood VectorH1 order 2 * H1 order 1.
The two runs differ ONLY in the sign in front of the two coupling terms.

What this fixture pins, all re-measured on this run:
  * both conventions reproduce the Poiseuille velocity profile -- the peak
    matches the prescribed 1 -- so the velocity really is convention-independent;
  * the analytic pressure gradient is -8, and the -p*div(v) form reproduces it;
  * the +p*div(v) form gives +8: the same magnitude with the wrong sign;
  * the two pressures are exact negatives of each other pointwise, not merely
    different;
  * the catalog's own NGSolve Stokes generator emits the +p*div(v) spelling, so
    the convention the claim describes is the one shipped -- read out of the
    generated source rather than assumed.

Mutation control: the pathology is the +p*div(v) run.  T2_MUTATE=1 applies the
documented fix to it -- the sign fed to solve() for the "plus_p_divv" case flips
from +1 to -1, so both runs use the -p*div(v) convention, the contrast vanishes
and the fixture loses plus_convention_has_the_wrong_sign=True,
pressures_are_exact_negatives=True and both_valid_is_true_of_velocity_only=True.
Re-run: T2_MUTATE=1 python source.py
"""
from __future__ import annotations

import os
import sys

import numpy
from netgen.geom2d import SplineGeometry
from ngsolve import (
    BilinearForm,
    CoefficientFunction,
    GridFunction,
    Grad,
    H1,
    InnerProduct,
    LinearForm,
    Mesh,
    VectorH1,
    div,
    dx,
    y,
)

NU = 1.0
LEN = 4.0
HGT = 1.0
UMAX = 1.0
DPDX_EXACT = -8.0 * NU * UMAX / HGT ** 2
MUTATE = os.environ.get("T2_MUTATE") == "1"


def solve(mesh, sign):
    V = VectorH1(mesh, order=2, dirichlet="wall|inlet")
    Q = H1(mesh, order=1)
    X = V * Q
    (u, p), (v, q) = X.TnT()
    a = BilinearForm(X)
    a += (NU * InnerProduct(Grad(u), Grad(v))
          + sign * div(u) * q + sign * div(v) * p) * dx
    a.Assemble()
    f = LinearForm(X)
    f.Assemble()
    gfu = GridFunction(X)
    gfu.components[0].Set(
        CoefficientFunction((4 * UMAX * y * (HGT - y) / HGT ** 2, 0)),
        definedon=mesh.Boundaries("inlet"))
    r = f.vec.CreateVector()
    r.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(X.FreeDofs(), inverse="umfpack") * r
    return gfu


def main() -> int:
    g = SplineGeometry()
    g.AddRectangle((0, 0), (LEN, HGT),
                   bcs=["wall", "outlet", "wall", "inlet"])
    mesh = Mesh(g.GenerateMesh(maxh=0.2))
    print(f"analytic_dpdx={DPDX_EXACT:+.4f}")

    results = {}
    # MUTATION SITE: the sign in front of the two coupling terms is the whole
    # pitfall.  T2_MUTATE=1 applies the documented fix to the "+" run.
    for sign, label in (((-1 if MUTATE else +1), "plus_p_divv"),
                        (-1, "minus_p_divv")):
        gfu = solve(mesh, sign)
        ph = gfu.components[1]
        p0 = float(ph(mesh(0.05, HGT / 2)))
        p1 = float(ph(mesh(LEN - 0.05, HGT / 2)))
        slope = (p1 - p0) / (LEN - 0.1)
        umax = max(float(gfu.components[0](mesh(LEN / 2, float(yy)))[0])
                   for yy in numpy.linspace(0.02, HGT - 0.02, 60))
        results[label] = (slope, umax, gfu)
        print(f"{label}_dpdx={slope:+.5f} umax={umax:.5f} "
              f"dpdx_matches_analytic={abs(slope - DPDX_EXACT) < 0.05}")

    sp, up, gp = results["plus_p_divv"]
    sm, um, gm = results["minus_p_divv"]
    print(f"both_reproduce_the_poiseuille_peak="
          f"{abs(up - UMAX) < 0.01 and abs(um - UMAX) < 0.01}")
    duv = float(numpy.abs(
        gp.components[0].vec.FV().NumPy()
        - gm.components[0].vec.FV().NumPy()).max())
    print(f"velocity_difference_between_conventions={duv:.3e}")
    print(f"velocity_is_convention_independent={duv < 1e-10}")

    print(f"minus_convention_matches_analytic_sign="
          f"{abs(sm - DPDX_EXACT) < 0.05}")
    print(f"plus_convention_has_the_wrong_sign="
          f"{abs(sp + DPDX_EXACT) < 0.05 and sp * DPDX_EXACT < 0}")
    print(f"magnitudes_agree={abs(abs(sp) - abs(sm)) < 0.05}")

    pv = gp.components[1].vec.FV().NumPy()
    mv = gm.components[1].vec.FV().NumPy()
    negatives = float(numpy.abs(pv + mv).max())
    print(f"max_abs_of_p_plus_and_p_minus_summed={negatives:.3e}")
    print(f"pressures_are_exact_negatives={negatives < 1e-9}")
    print(f"both_valid_is_true_of_velocity_only="
          f"{duv < 1e-10 and negatives < 1e-9}")

    # What does the shipped generator actually write?
    sys.path.insert(0, "/home/alexander/Schreibtisch/ofa-verify-ngs/src")
    import backends.ngsolve.backend as ngb          # noqa: E402
    try:
        ngb.register()
    except Exception:                                          # noqa: BLE001
        pass
    from core.registry import get_backend           # noqa: E402
    src = get_backend("ngsolve").generate_input("stokes", "2d", {})
    lines = [ln.strip() for ln in src.splitlines() if "div(" in ln]
    print(f"generator_coupling_lines={lines}")
    plus_shipped = any("+ div(v)*p" in ln or "div(v)*p*dx" in ln
                       for ln in lines) and not any("-div(v)*p" in ln
                                                    for ln in lines)
    print(f"catalog_generator_ships_the_plus_convention={plus_shipped}")

    ok = (
        abs(up - UMAX) < 0.01 and abs(um - UMAX) < 0.01
        and duv < 1e-10
        and abs(sm - DPDX_EXACT) < 0.05
        and sp * DPDX_EXACT < 0 and abs(sp + DPDX_EXACT) < 0.05
        and negatives < 1e-9
        and plus_shipped
    )
    if ok:
        return 0
    print("FAIL: Stokes pressure-sign invariant not held", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
