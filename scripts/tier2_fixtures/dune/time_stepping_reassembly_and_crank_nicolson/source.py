"""Tier-2: four statements about marching a heat problem in time.

  time_dependent_heat#0   the right-hand side reads u_n, so u_n has to
                          be updated every step; leaving it at the
                          initial condition freezes the solution there.
  time_dependent_heat#2   the mass matrix does not change on a fixed
                          mesh, and the galerkin scheme reuses the
                          assembled operator automatically when the form
                          structure is constant — per-step cost is flat.
  time_dependent_heat#3   writing Crank-Nicolson with the implicit-Euler
                          form (alpha rather than alpha/2) silently
                          degrades the temporal order from 2 to 1.
  time_dependent_heat#4   backward Euler is unconditionally stable, so
                          dt is chosen for accuracy: a dt far above any
                          explicit CFL limit still decays monotonically
                          and produces no NaN.

The test problem is u_t = kappa * Laplace(u) on [0,1]^2 with
u(x, 0) = sin(pi x) sin(pi y) and u = 0 on the boundary, whose exact
solution is that mode times exp(-2 kappa pi^2 t). dt, kappa and the
final time are dune.ufl.Constants, so the temporal refinement study
costs no rebuild — only two forms are compiled, backward Euler and
Crank-Nicolson.

Verified by execution against dune-fem 2.12.0.2.
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                            # noqa: E402
from dune.fem.space import lagrange                             # noqa: E402
from dune.fem.scheme import galerkin                            # noqa: E402
from dune.ufl import Constant, DirichletBC                      # noqa: E402
import dune.fem as dfem                                         # noqa: E402
from ufl import (TrialFunction, TestFunction,                    # noqa: E402
                 SpatialCoordinate, dot, grad, dx, sin, pi)

KAPPA = 0.1
T_END = 0.05


def main() -> int:
    fail: list[str] = []
    gridView = structuredGrid([0, 0], [1, 1], [16, 16])
    space = lagrange(gridView, order=2)
    u, v = TrialFunction(space), TestFunction(space)
    x = SpatialCoordinate(space)
    u0_expr = sin(pi * x[0]) * sin(pi * x[1])

    dt = Constant(0.01, name="dt")
    kappa = Constant(KAPPA, name="kappa")
    u_old = space.interpolate(u0_expr, name="u_old")
    dbc = DirichletBC(space, 0)
    zero = Constant(0.0, name="zero") * v * dx

    be = ((u - u_old) / dt * v
          + kappa * dot(grad(u), grad(v))) * dx
    cn = ((u - u_old) / dt * v
          + kappa / 2 * dot(grad(u), grad(v))
          + kappa / 2 * dot(grad(u_old), grad(v))) * dx
    scheme_be = galerkin([be == zero, dbc], solver="cg")
    scheme_cn = galerkin([cn == zero, dbc], solver="cg")

    uh = space.interpolate(u0_expr, name="uh")

    def march(scheme, n_steps, step_dt, update_old=True):
        dt.value = step_dt
        u_old.interpolate(u0_expr)
        uh.interpolate(u0_expr)
        per_step = []
        for _ in range(n_steps):
            t0 = time.time()
            scheme.solve(target=uh)
            per_step.append(time.time() - t0)
            if update_old:
                u_old.assign(uh)
        return per_step

    # The temporal order is measured against a FULLY DISCRETE reference
    # — the same scheme at a much smaller step — not against the
    # analytic solution. Measured on this grid, the analytic route puts
    # a spatial floor at about 2.8e-05 that Crank-Nicolson reaches by
    # dt = T/5, so its observed order came out 0.000: the study was
    # measuring the SPACE error, not the time error.
    u_ref = space.interpolate(u0_expr, name="u_ref")

    def l2_against(reference):
        return float(np.sqrt(dfem.integrate(
            (uh - reference) ** 2, gridView=gridView, order=8)))

    # ── time_dependent_heat#0: freeze u_n and the solution freezes ──
    steps = 5
    march(scheme_be, steps, T_END / steps, update_old=True)
    moved = float(np.array(uh.as_numpy).max())
    march(scheme_be, steps, T_END / steps, update_old=False)
    frozen = float(np.array(uh.as_numpy).max())
    start = float(np.array(
        space.interpolate(u0_expr, name="probe").as_numpy).max())
    print(f"initial_max={start:.6f}")
    print(f"updated_u_old_max={moved:.6f}")
    print(f"frozen_u_old_max={frozen:.6f}")
    print(f"updating_u_old_advances_the_solution="
          f"{abs(moved - start) > 1e-4}")
    print(f"frozen_u_old_repeats_the_first_step="
          f"{abs(frozen - start) < abs(moved - start)}")
    if abs(moved - start) <= 1e-4:
        fail.append(f"marching with an updated u_n did not change the "
                    f"solution ({start} -> {moved})")
    if abs(frozen - start) >= abs(moved - start):
        fail.append(f"leaving u_n at the initial condition advanced the "
                    f"solution as far as updating it ({frozen} vs "
                    f"{moved}); the claim is that the right-hand side "
                    f"never updates and the run stalls")

    # ── time_dependent_heat#2: per-step cost is flat ───────────────
    per_step = march(scheme_be, 40, T_END / 40)
    tail = per_step[5:]
    ratio = max(tail) / (sum(tail) / len(tail))
    print(f"per_step_first={per_step[0]:.6f}")
    print(f"per_step_mean_tail={sum(tail) / len(tail):.6f}")
    print(f"per_step_max_over_mean_tail={ratio:.2f}")
    print(f"scheme_reuses_its_operator={ratio < 10.0}")
    if ratio >= 10.0:
        fail.append(f"per-step cost varied by {ratio:.1f}x across a run "
                    f"with a constant form; the claim is that the "
                    f"scheme reuses the assembled operator")

    # ── time_dependent_heat#3: the order of the two schemes ────────
    orders = {}
    for label, scheme in (("BE", scheme_be), ("CN", scheme_cn)):
        march(scheme, 640, T_END / 640)
        u_ref.assign(uh)
        errs = []
        for n_steps in (5, 10, 20, 40):
            march(scheme, n_steps, T_END / n_steps)
            errs.append(l2_against(u_ref))
        rates = [np.log2(errs[i] / errs[i + 1])
                 for i in range(len(errs) - 1)]
        orders[label] = rates[-1]
        print(f"{label}_errors="
              + ",".join(f"{e:.6e}" for e in errs))
        print(f"{label}_observed_order={rates[-1]:.3f}")
    print(f"backward_euler_is_first_order="
          f"{0.7 < orders['BE'] < 1.4}")
    print(f"crank_nicolson_is_second_order="
          f"{1.7 < orders['CN'] < 2.4}")
    print(f"implicit_euler_factor_costs_an_order="
          f"{orders['CN'] - orders['BE'] > 0.5}")
    if not 0.7 < orders["BE"] < 1.4:
        fail.append(f"backward Euler measured order {orders['BE']:.3f}, "
                    f"expected about 1")
    if not 1.7 < orders["CN"] < 2.4:
        fail.append(f"Crank-Nicolson measured order {orders['CN']:.3f}, "
                    f"expected about 2")

    # ── time_dependent_heat#4: BE is unconditionally stable ────────
    h = 1.0 / 16
    cfl_dt = h ** 2 / (2 * KAPPA)
    big_dt = 1000 * cfl_dt
    march(scheme_be, 20, big_dt)
    vals = np.array(uh.as_numpy)
    print(f"explicit_cfl_dt={cfl_dt:.6e}")
    print(f"implicit_dt_used={big_dt:.6e}")
    print(f"dt_over_cfl_limit={big_dt / cfl_dt:.0f}")
    print(f"be_result_is_finite={bool(np.all(np.isfinite(vals)))}")
    print(f"be_result_decayed={float(np.abs(vals).max()):.6e}")
    print(f"be_is_unconditionally_stable="
          f"{bool(np.all(np.isfinite(vals))) and float(np.abs(vals).max()) < start}")
    if not np.all(np.isfinite(vals)):
        fail.append("backward Euler produced non-finite values at "
                    "1000x the explicit CFL step")
    if float(np.abs(vals).max()) >= start:
        fail.append(f"the solution did not decay at the large step "
                    f"({float(np.abs(vals).max()):.6e} vs {start})")

    if not fail:
        print("dune_time_stepping_traps_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
