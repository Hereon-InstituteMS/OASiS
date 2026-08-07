"""Tier-2: the three reaction-diffusion statements.

  reaction_diffusion#0   a nonlinear reaction written directly in the
                         trial function is accepted and linearised by
                         the scheme's internal Newton; no manual
                         linearisation is needed.
  reaction_diffusion#1   explicit stepping above the reaction's
                         stability limit gives NaN within a handful of
                         steps, while the implicit scheme at the same dt
                         is fine.
  reaction_diffusion#2   a two-species system needs dimRange=2; a scalar
                         space cannot carry the second field at all, and
                         the coupling lives in the UFL form.

dt and the reaction rate are dune.ufl.Constants, so the stability sweep
costs no rebuild. Three modules are compiled: the implicit scalar
scheme, the explicit (mass-matrix) scalar scheme, and the two-species
coupled scheme.

Verified by execution against dune-fem 2.12.0.2.

MUTATION CONTROL. T2_MUTATE=1 runs the "above the limit" explicit sweep
at the STABLE step instead — the pathology removed. The march then
stays finite, so 'explicit_blows_up_above_the_limit=True' is no longer
printed and a FAIL: line appears. dt is a dune.ufl.Constant, so the
mutation costs no rebuild.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

MUTATE = os.environ.get("T2_MUTATE") == "1"

from dune.grid import structuredGrid                            # noqa: E402
from dune.fem.space import lagrange                             # noqa: E402
from dune.fem.scheme import galerkin                            # noqa: E402
from dune.ufl import Constant, DirichletBC                      # noqa: E402
from ufl import (TrialFunction, TestFunction,                    # noqa: E402
                 SpatialCoordinate, as_vector, dot, grad, dx,
                 sin, pi)


def main() -> int:
    fail: list[str] = []
    gridView = structuredGrid([0, 0], [1, 1], [8, 8])
    space = lagrange(gridView, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    x = SpatialCoordinate(space)

    dt = Constant(0.001, name="dt")
    lam = Constant(50.0, name="lam")
    D = Constant(0.01, name="D")
    u0_expr = 0.5 + 0.25 * sin(pi * x[0]) * sin(pi * x[1])
    u_old = space.interpolate(u0_expr, name="u_old")
    dbc = DirichletBC(space, 0.5)
    zero = Constant(0.0, name="zero") * v * dx

    # ── #0: the reaction is written in the TRIAL function ──────────
    implicit = ((u - u_old) / dt * v
                + D * dot(grad(u), grad(v))
                - lam * u * (1 - u) * v) * dx
    scheme_imp = galerkin([implicit == zero, dbc], solver="cg")
    uh = space.interpolate(u0_expr, name="uh")
    info = scheme_imp.solve(target=uh)
    print(f"nonlinear_reaction_accepted=True")
    print(f"implicit_converged={bool(info['converged'])}")
    print(f"implicit_newton_iterations={int(info['iterations'])}")
    print(f"reaction_was_linearised_internally="
          f"{int(info['iterations']) >= 1}")
    if not info["converged"] or int(info["iterations"]) < 1:
        fail.append(f"the implicit nonlinear reaction step reported "
                    f"converged={info['converged']} with "
                    f"{info['iterations']} Newton iterations; the claim "
                    f"is that UFL differentiation builds the Jacobian "
                    f"and Newton iterates per step")

    # ── #1: explicit stepping blows up above the stability limit ───
    explicit = u * v * dx
    explicit_rhs = (u_old * v
                    + dt * (-D * dot(grad(u_old), grad(v))
                            + lam * u_old * (1 - u_old) * v)) * dx
    scheme_exp = galerkin([explicit == explicit_rhs, dbc], solver="cg")

    def march(scheme, steps, step_dt):
        dt.value = step_dt
        u_old.interpolate(u0_expr)
        uh.interpolate(u0_expr)
        for _ in range(steps):
            scheme.solve(target=uh)
            u_old.assign(uh)
            vals = np.array(uh.as_numpy)
            if not np.all(np.isfinite(vals)):
                return False, float("nan")
        return True, float(np.abs(np.array(uh.as_numpy)).max())

    stable_dt = 0.5 * 2.0 / 50.0
    unstable_dt = 20.0 * 2.0 / 50.0
    ok_small, max_small = march(scheme_exp, 10, stable_dt)
    if MUTATE:
        print("mutation=the_above_the_limit_sweep_uses_the_"
              "stable_step")
    ok_big, max_big = march(scheme_exp, 10,
                            stable_dt if MUTATE else unstable_dt)
    print(f"explicit_stability_limit_2_over_lambda={2.0 / 50.0:.4f}")
    print(f"explicit_dt_below_limit={stable_dt:.4f} finite={ok_small}")
    print(f"explicit_dt_above_limit={unstable_dt:.4f} finite={ok_big}")
    print(f"explicit_blows_up_above_the_limit={not ok_big}")
    if not ok_small:
        fail.append(f"explicit stepping at dt={stable_dt} below the "
                    f"2/lambda limit already produced non-finite "
                    f"values; the control is broken")
    if ok_big:
        fail.append(f"explicit stepping at dt={unstable_dt}, twenty "
                    f"times the 2/lambda limit, stayed finite (max "
                    f"{max_big}); the claim is NaN within about ten "
                    f"steps")

    # …and the implicit scheme survives the same step
    ok_imp, max_imp = march(scheme_imp, 10, unstable_dt)
    print(f"implicit_at_the_same_dt_finite={ok_imp}")
    print(f"implicit_at_the_same_dt_max={max_imp}")
    if not ok_imp:
        fail.append("the implicit scheme also blew up at the large "
                    "step, so the fixture cannot show that switching "
                    "to backward Euler is the fix")
    dt.value = 0.001

    # ── #2: two species need dimRange=2 ────────────────────────────
    try:
        space.interpolate([u0_expr, u0_expr], name="two_on_scalar")
        print("scalar_space_holds_two_species=True")
        fail.append("a scalar space accepted a two-component field; the "
                    "claim is that a multi-species system needs "
                    "dimRange=2")
    except Exception as exc:                                 # noqa: BLE001
        print(f"scalar_space_rejects_two_species={type(exc).__name__}")

    space2 = lagrange(gridView, order=1, dimRange=2)
    w, z = TrialFunction(space2), TestFunction(space2)
    w_old = space2.interpolate(as_vector([u0_expr, 1 - u0_expr]),
                               name="w_old")
    # Schnakenberg-like coupling: each species' reaction reads BOTH.
    react = as_vector([w[0] ** 2 * w[1] - w[0],
                       -w[0] ** 2 * w[1] + 0.9])
    coupled = (dot((w - w_old) / dt, z)
               + D * dot(grad(w[0]), grad(z[0]))
               + 10 * D * dot(grad(w[1]), grad(z[1]))
               - dot(react, z)) * dx
    scheme2 = galerkin([coupled == Constant(0.0, name="zero2") * z[0] * dx,
                        DirichletBC(space2, [None, None])], solver="gmres")
    wh = space2.interpolate(as_vector([u0_expr, 1 - u0_expr]),
                            name="wh")
    info2 = scheme2.solve(target=wh)
    pair = np.array(wh.as_numpy).reshape(-1, 2)
    print(f"two_species_space_dimRange={space2.dimRange}")
    print(f"coupled_converged={bool(info2['converged'])}")
    print(f"species_means={pair[:, 0].mean():.6f},{pair[:, 1].mean():.6f}")
    print(f"species_evolve_differently="
          f"{abs(pair[:, 0].mean() - pair[:, 1].mean()) > 1e-6}")
    if not info2["converged"]:
        fail.append("the coupled two-species step did not converge")
    if abs(pair[:, 0].mean() - pair[:, 1].mean()) <= 1e-6:
        fail.append("the two species came out identical, so the "
                    "fixture cannot show that the form couples them")

    if not fail:
        print("dune_reaction_diffusion_traps_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
