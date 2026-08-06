"""Tier-2: the two heat-equation traps that still converge.

  heat#0   DirichletBC(space, T_fixed) with a CONSTANT cannot give
           different values on different boundary segments; the DUNE
           pattern is DirichletBC(space, conditional(...)), where the
           UFL conditional selects the value from the coordinate.
  heat#1   assembling only the stiffness term — forgetting the mass
           contribution — gives the STEADY-STATE solution at every time
           step, whatever dt is. The transient is simply missing.

Two compiled forms serve both: a steady one and a transient one. dt is
a dune.ufl.Constant, so changing it is free.

Verified by execution against dune-fem 2.12.0.2.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from dune.grid import structuredGrid                            # noqa: E402
from dune.fem.space import lagrange                             # noqa: E402
from dune.fem.scheme import galerkin                            # noqa: E402
from dune.ufl import Constant, DirichletBC                      # noqa: E402
from ufl import (TrialFunction, TestFunction,                    # noqa: E402
                 SpatialCoordinate, dot, grad, dx, conditional, lt)

TOL = 1e-8
T_LEFT, T_RIGHT = 100.0, 0.0


def main() -> int:
    fail: list[str] = []
    gridView = structuredGrid([0, 0], [1, 1], [8, 8])
    space = lagrange(gridView, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    x = SpatialCoordinate(space)

    coords = np.array(space.interpolate(x[0], name="xc").as_numpy)
    left_dofs = coords < TOL
    right_dofs = coords > 1 - TOL

    a_steady = dot(grad(u), grad(v)) * dx
    rhs_zero = Constant(0.0, name="zero") * v * dx

    # ── heat#0: one constant cannot serve two segments ─────────────
    const_bc = DirichletBC(space, T_LEFT)
    scheme_const = galerkin([a_steady == rhs_zero, const_bc],
                            solver="cg")
    uh_const = space.interpolate(0, name="uh_const")
    info_const = scheme_const.solve(target=uh_const)
    vals_const = np.array(uh_const.as_numpy)
    l_const = float(vals_const[left_dofs].mean())
    r_const = float(vals_const[right_dofs].mean())
    print(f"constant_bc_converged={bool(info_const['converged'])}")
    print(f"constant_bc_left_mean={l_const:.4f}")
    print(f"constant_bc_right_mean={r_const:.4f}")
    print(f"constant_bc_cannot_separate_segments="
          f"{abs(l_const - r_const) < 1e-9}")
    if abs(l_const - r_const) >= 1e-9:
        fail.append(f"a single-value DirichletBC produced different "
                    f"boundary values ({l_const} vs {r_const}); the "
                    f"claim is that it cannot")

    # the conditional pattern does separate them
    seg = conditional(lt(x[0], 0.5), T_LEFT, T_RIGHT)
    scheme_seg = galerkin([a_steady == rhs_zero,
                           DirichletBC(space, seg)], solver="cg")
    uh_seg = space.interpolate(0, name="uh_seg")
    info_seg = scheme_seg.solve(target=uh_seg)
    vals_seg = np.array(uh_seg.as_numpy)
    l_seg = float(vals_seg[left_dofs].mean())
    r_seg = float(vals_seg[right_dofs].mean())
    print(f"conditional_bc_converged={bool(info_seg['converged'])}")
    print(f"conditional_bc_left_mean={l_seg:.4f}")
    print(f"conditional_bc_right_mean={r_seg:.4f}")
    print(f"conditional_bc_separates_segments="
          f"{abs(l_seg - T_LEFT) < 1e-9 and abs(r_seg - T_RIGHT) < 1e-9}")
    if not (abs(l_seg - T_LEFT) < 1e-9 and abs(r_seg - T_RIGHT) < 1e-9):
        fail.append(f"the conditional BC gave left {l_seg} and right "
                    f"{r_seg}; expected {T_LEFT} and {T_RIGHT}")

    # ── heat#1: no mass term means no transient ────────────────────
    dt = Constant(0.01, name="dt")
    kappa = Constant(1.0, name="kappa")
    u_old = space.interpolate(0, name="u_old")
    bc_seg = DirichletBC(space, seg)

    a_trans = ((u - u_old) / dt * v
               + kappa * dot(grad(u), grad(v))) * dx
    scheme_trans = galerkin([a_trans == rhs_zero, bc_seg], solver="cg")

    def march(scheme, target, old, steps):
        old.interpolate(0)
        target.interpolate(0)
        history = []
        for _ in range(steps):
            scheme.solve(target=target)
            old.assign(target)
            history.append(float(np.array(target.as_numpy).mean()))
        return history

    uh_t = space.interpolate(0, name="uh_t")
    hist_trans = march(scheme_trans, uh_t, u_old, 5)
    print("transient_mean_history="
          + ",".join(f"{h:.4f}" for h in hist_trans))
    moves = all(hist_trans[i + 1] > hist_trans[i] + 1e-6
                for i in range(len(hist_trans) - 1))
    print(f"transient_field_evolves={moves}")
    if not moves:
        fail.append(f"with the mass term the field did not evolve: "
                    f"{hist_trans}")

    # the same loop with the mass term dropped
    uh_s = space.interpolate(0, name="uh_s")
    hist_steady = []
    for _ in range(5):
        scheme_seg.solve(target=uh_s)
        hist_steady.append(float(np.array(uh_s.as_numpy).mean()))
    print("stiffness_only_mean_history="
          + ",".join(f"{h:.4f}" for h in hist_steady))
    frozen = max(hist_steady) - min(hist_steady) < 1e-12
    print(f"stiffness_only_is_steady_at_every_step={frozen}")
    print(f"stiffness_only_equals_the_steady_solution="
          f"{abs(hist_steady[0] - float(vals_seg.mean())) < 1e-12}")
    if not frozen:
        fail.append(f"the stiffness-only loop changed between steps: "
                    f"{hist_steady}")

    # and it is INDEPENDENT of dt, which is the tell
    dt.value = 100.0
    hist_big = march(scheme_trans, uh_t, u_old, 5)
    dt.value = 0.01
    print("transient_mean_history_dt100="
          + ",".join(f"{h:.4f}" for h in hist_big))
    print(f"transient_depends_on_dt="
          f"{abs(hist_big[0] - hist_trans[0]) > 1e-6}")
    print(f"steady_form_would_not={frozen}")
    if abs(hist_big[0] - hist_trans[0]) <= 1e-6:
        fail.append("changing dt by four orders of magnitude did not "
                    "change the first transient step, so the fixture "
                    "cannot show that the stiffness-only form is the "
                    "dt-independent one")

    if not fail:
        print("dune_heat_transient_traps_verified=True")
        return 0
    for r in fail:
        print(f"FAIL: {r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
