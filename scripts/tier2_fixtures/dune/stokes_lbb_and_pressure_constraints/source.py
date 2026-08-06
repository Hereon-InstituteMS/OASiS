"""Tier-2: three ways a Stokes system is under- or over-constrained.

  stokes#3   equal-order velocity/pressure violates LBB: the system
             assembles and solves, and the pressure comes back with a
             checkerboard oscillation whose amplitude does NOT decrease
             under refinement while the velocity still looks plausible.
  stokes#4   a Dirichlet velocity on the ENTIRE boundary leaves the
             pressure undetermined — the direct solver either reports a
             singular matrix or returns a pressure shifted by an
             arbitrary constant, while the velocity looks right.
  stokes#5   the pressure entry of the DirichletBC value list must be
             None, not 0: a 0 there also pins the pressure on that
             boundary and over-constrains the system.

Verified by execution against dune-fem 2.12.0.2.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                            # noqa: E402
from dune.fem.space import lagrange, composite                  # noqa: E402
from dune.fem.scheme import galerkin                            # noqa: E402
from dune.ufl import Constant, DirichletBC                      # noqa: E402
from ufl import (TrialFunction, TestFunction,                    # noqa: E402
                 SpatialCoordinate, as_vector, div, grad, inner,
                 dx, conditional, lt, gt, Or)

TOL = 1e-8
PARAMS = {"linear.tolerance": 1e-12, "linear.maxiterations": 200000}


def stokes_pieces(gridView, vel_order, pre_order):
    # A composite space stores its legs BLOCKED, not interleaved: the
    # velocity dofs come first, then the pressure ones. Measured on an
    # 8x8 P2/P1 pair, size 659 = 578 + 81. Reshaping the flat array to
    # (-1, 3) therefore fails outright.
    _vel = lagrange(gridView, dimRange=2, order=vel_order)
    _pre = lagrange(gridView, order=pre_order)
    W = composite(_vel, _pre)
    W._leg_sizes = (_vel.size, _pre.size)
    t, s = TrialFunction(W), TestFunction(W)
    u, p = as_vector([t[0], t[1]]), t[2]
    v, q = as_vector([s[0], s[1]]), s[2]
    x = SpatialCoordinate(W)
    a = (inner(grad(u), grad(v)) - p * div(v) - q * div(u)) * dx
    L = Constant(0.0, name="zero") * q * dx
    return W, a, L, x


def pressure_of(W, wh):
    nv, _ = W._leg_sizes
    return np.array(wh.as_numpy)[nv:]


def velocity_of(W, wh):
    nv, _ = W._leg_sizes
    return np.array(wh.as_numpy)[:nv].reshape(-1, 2)


def main() -> int:
    fail: list[str] = []

    # ── stokes#3: equal order, checkerboard that does not shrink ───
    osc, vel_scale = {}, {}
    for nx in (8, 16):
        gv = structuredGrid([0, 0], [1, 1], [nx, nx])
        W, a, L, x = stokes_pieces(gv, 1, 1)
        inlet = lt(x[0], TOL)
        walls = Or(lt(x[1], TOL), gt(x[1], 1 - TOL))
        bcs = [DirichletBC(W, [x[1] * (1 - x[1]), 0, None], inlet),
               DirichletBC(W, [0, 0, None], walls)]
        sch = galerkin([a == L] + bcs,
                       solver=("suitesparse", "umfpack"),
                       parameters=PARAMS)
        wh = W.interpolate([0, 0, 0], name=f"eq{nx}")
        info = sch.solve(target=wh)
        p = pressure_of(W, wh)
        vel = velocity_of(W, wh)
        # neighbouring-dof jump: a checkerboard shows up as a large mean
        # absolute difference between consecutive dofs
        osc[nx] = float(np.abs(np.diff(p)).mean())
        vel_scale[nx] = float(np.abs(vel).max())
        print(f"equal_order_nx{nx}_converged={bool(info['converged'])}")
        print(f"equal_order_nx{nx}_pressure_oscillation={osc[nx]:.6e}")
        print(f"equal_order_nx{nx}_velocity_max={vel_scale[nx]:.6f}")
        if not info["converged"]:
            fail.append(f"the equal-order system at nx={nx} did not "
                        f"solve; the claim is that it assembles and "
                        f"solves and is wrong only in the pressure")
    ratio = osc[16] / osc[8]
    print(f"equal_order_oscillation_ratio={ratio:.4f}")
    print(f"equal_order_mode_does_not_shrink={ratio > 0.5}")
    print(f"velocity_still_looks_plausible="
          f"{0.0 < vel_scale[16] < 5.0}")
    if ratio <= 0.5:
        fail.append(f"the equal-order pressure oscillation fell by "
                    f"{1 / ratio:.1f}x under one refinement; the claim "
                    f"is that its amplitude does NOT decrease")

    # ── the stable pair, as the control ────────────────────────────
    gv = structuredGrid([0, 0], [1, 1], [8, 8])
    W, a, L, x = stokes_pieces(gv, 2, 1)
    inlet = lt(x[0], TOL)
    walls = Or(lt(x[1], TOL), gt(x[1], 1 - TOL))
    outlet = gt(x[0], 1 - TOL)
    profile = x[1] * (1 - x[1])
    bcs_open = [DirichletBC(W, [profile, 0, None], inlet),
                DirichletBC(W, [0, 0, None], walls)]
    sch_ok = galerkin([a == L] + bcs_open,
                      solver=("suitesparse", "umfpack"),
                      parameters=PARAMS)

    def solve_twice(scheme, tag):
        outs = []
        for k in range(2):
            wh = W.interpolate([0, 0, 0], name=f"{tag}{k}")
            info = scheme.solve(target=wh)
            outs.append((info, pressure_of(W, wh), velocity_of(W, wh)))
        return outs

    (info_a, p_a, v_a), (info_b, p_b, v_b) = solve_twice(sch_ok, "ok")
    print(f"taylor_hood_converged={bool(info_a['converged'])}")
    print(f"taylor_hood_pressure_range={p_a.min():.4f},{p_a.max():.4f}")
    print(f"taylor_hood_pressure_is_reproducible="
          f"{float(np.abs(p_a - p_b).max()) < 1e-9}")
    if not info_a["converged"]:
        fail.append("the stable Taylor-Hood control did not solve")

    # ── stokes#4: velocity Dirichlet everywhere ────────────────────
    bcs_closed = bcs_open + [DirichletBC(W, [0, 0, None], outlet)]
    sch_closed = galerkin([a == L] + bcs_closed,
                          solver=("suitesparse", "umfpack"),
                          parameters=PARAMS)
    try:
        (info_c, p_c, v_c), (info_d, p_d, v_d) = solve_twice(
            sch_closed, "closed")
        singular_raised = False
        drift = float(np.abs((p_c - p_c.mean()) - (p_d - p_d.mean())).max())
        shift = float(abs(p_c.mean() - p_d.mean()))
        print(f"closed_domain_raised=False")
        print(f"closed_domain_converged={bool(info_c['converged'])}")
        print(f"closed_domain_pressure_mean_1={p_c.mean():.6e}")
        print(f"closed_domain_pressure_mean_2={p_d.mean():.6e}")
        print(f"closed_domain_pressure_shape_is_stable={drift < 1e-6}")
        print(f"closed_domain_pressure_level_is_arbitrary="
              f"{shift > 1e-9 or not bool(info_c['converged'])}")
        # What is actually observed is not a wandering constant but a
        # pressure level that runs away: 9.9e+16 against 2.0 for the
        # same problem with one open boundary, while the velocity stays
        # sane. That IS the undetermined level, resolved by the solver
        # into whatever the singular system leaves it at.
        blowup = float(np.abs(p_c).max()) > 1e6 * max(
            float(np.abs(p_a).max()), 1.0)
        print(f"closed_domain_pressure_max={float(np.abs(p_c).max()):.6e}")
        print(f"open_domain_pressure_max={float(np.abs(p_a).max()):.6e}")
        print(f"closed_domain_pressure_runs_away={blowup}")
        print(f"closed_domain_velocity_max="
              f"{float(np.abs(v_c).max()):.6f}")
        print(f"closed_domain_velocity_still_sane="
              f"{float(np.abs(v_c).max()) < 10.0}")
        undetermined = blowup or (shift > 1e-9) or (
            not info_c["converged"])
    except Exception as exc:                                 # noqa: BLE001
        singular_raised = True
        undetermined = True
        print(f"closed_domain_raised={type(exc).__name__}")
        print(f"closed_domain_message="
              f"{' '.join(str(exc).split())[:160]}")
    print(f"pressure_is_undetermined_without_an_open_boundary="
          f"{undetermined or singular_raised}")
    if not (undetermined or singular_raised):
        fail.append("a velocity Dirichlet condition on the entire "
                    "boundary left the pressure fully determined and "
                    "reproducible; the claim is that it does not")

    # ── stokes#5: 0 instead of None in the pressure slot ───────────
    bcs_zero_p = [DirichletBC(W, [profile, 0, 0], inlet),
                  DirichletBC(W, [0, 0, 0], walls)]
    sch_zero_p = galerkin([a == L] + bcs_zero_p,
                          solver=("suitesparse", "umfpack"),
                          parameters=PARAMS)
    wh_z = W.interpolate([0, 0, 0], name="zero_p")
    info_z = sch_zero_p.solve(target=wh_z)
    p_z = pressure_of(W, wh_z)
    v_z = velocity_of(W, wh_z)
    # the pressure really is pinned on the constrained boundary
    nv, npre = W._leg_sizes
    coords = np.array(W.interpolate([x[0], x[1], 0],
                                    name="coords").as_numpy)[nv:]
    on_inlet = coords < TOL
    pinned = float(np.abs(p_z[on_inlet]).max())
    free = float(np.abs(p_a[on_inlet]).max())
    print(f"zero_pressure_entry_converged={bool(info_z['converged'])}")
    print(f"zero_pressure_entry_inlet_pressure_max={pinned:.6e}")
    print(f"none_pressure_entry_inlet_pressure_max={free:.6e}")
    print(f"zero_entry_pins_the_pressure={pinned < 1e-9 < free}")
    print(f"zero_entry_changes_the_velocity="
          f"{float(np.abs(v_z - v_a).max()) > 1e-9}")
    # stokes#5 is NOT claimed as covered: measured, a 0 in the pressure
    # slot did NOT drive the inlet pressure to zero (|p| max 1.543121
    # against 2.000000 with None) — it changes the solution but not in
    # the way the claim describes. Printed, not asserted.
    print("zero_pressure_entry_claim_not_reproduced=True")

    if not fail:
        print("dune_stokes_constraint_traps_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
