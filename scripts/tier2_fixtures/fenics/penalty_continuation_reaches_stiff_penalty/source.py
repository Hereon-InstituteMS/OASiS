"""Tier-2 for fenics contact#3: if a very stiff penalty is genuinely needed, do
not just raise it and hope — use penalty continuation, re-solving decade by
decade and reusing the previous solution as the starting point.

Wrong variant: the same obstacle problem started cold (u = 0) at gamma = 1e12.
It reports DIVERGED_DTOL after a single Newton iteration and returns a body
that has passed straight through the obstacle.

Right variant, measured in the same process: gamma is a fem.Constant, so the
ladder 1e2 -> 1e4 -> 1e6 -> 1e8 -> 1e10 -> 1e12 is just a new value assigned to
it followed by another problem.solve(), with u left where the previous decade
put it. Every decade converges (CONVERGED_FNORM_ABS for the first decades,
CONVERGED_FNORM_RELATIVE / CONVERGED_SNORM_RELATIVE for the last ones), the
last three decades need a single Newton iteration each, and the penetration
falls monotonically from 2.19 element edges at gamma = 1e2 to exactly zero.

FINDING, minor: the claim names CONVERGED_FNORM_RELATIVE or
CONVERGED_SNORM_RELATIVE as the reasons continuation produces. On this
installation the early decades stop on CONVERGED_FNORM_ABS (reason 2) instead;
all of them are positive, which is what the fixture asserts.

Mutation control: T2_MUTATE=1 reaches gamma = 1e12 through the continuation
ladder instead of cold, and the divergence disappears.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

from dolfinx import fem, mesh  # noqa: E402
import dolfinx.fem.petsc as dfp  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N = 24
H = 1.0 / N
PHI = -0.2
LADDER = (1.0e2, 1.0e4, 1.0e6, 1.0e8, 1.0e10, 1.0e12)
STIFF = LADDER[-1]


def main() -> int:
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, mesh.CellType.triangle)
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    V = fem.functionspace(msh, ("Lagrange", 1))
    facets = mesh.exterior_facet_indices(msh.topology)
    dofs = fem.locate_dofs_topological(V, tdim - 1, facets)
    bc = fem.dirichletbc(0.0, dofs, V)

    u = fem.Function(V, name="u")
    v = ufl.TestFunction(V)
    phi = fem.Constant(msh, PHI)
    gamma = fem.Constant(msh, LADDER[0])
    f = fem.Constant(msh, -10.0)
    F = (ufl.dot(ufl.grad(u), ufl.grad(v)) - f * v
         - gamma * ufl.max_value(phi - u, 0.0) * v) * ufl.dx
    problem = dfp.NonlinearProblem(
        F, u, bcs=[bc], petsc_options_prefix="t2_cont_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_rtol": 1e-9, "snes_atol": 1e-10,
                       "snes_max_it": 30, "snes_linesearch_type": "basic",
                       "snes_converged_reason": None})

    def step(g, cold):
        if cold:
            u.x.array[:] = 0.0
        gamma.value = float(g)
        problem.solve()
        pen = max(0.0, PHI - float(u.x.array.min()))
        return (problem.solver.getConvergedReason(),
                problem.solver.getIterationNumber(), pen / H)

    # ---- the slot: how gamma = 1e12 is reached --------------------------
    if MUTATE:
        u.x.array[:] = 0.0
        for g in LADDER[:-1]:
            step(g, cold=False)
        r_slot, it_slot, pen_slot = step(STIFF, cold=False)
    else:
        r_slot, it_slot, pen_slot = step(STIFF, cold=True)
    print(f"slot_reason={r_slot} iterations={it_slot} "
          f"penetration_over_h={pen_slot:.6f}")
    print(f"slot_reason_is_diverged_dtol={r_slot == -9}")
    print(f"slot_diverged_after_one_iteration={it_slot == 1}")

    # ---- the continuation ladder, always run from cold ------------------
    u.x.array[:] = 0.0
    reasons, iters, pens = [], [], []
    for g in LADDER:
        r, it, pen = step(g, cold=False)
        reasons.append(r)
        iters.append(it)
        pens.append(pen)
        print(f"continuation gamma={g:.0e} reason={r} iterations={it} "
              f"penetration_over_h={pen:.6f}")
    all_conv = all(r > 0 for r in reasons)
    shrinks = all(pens[i + 1] <= pens[i] for i in range(len(pens) - 1))
    cheap = all(it <= 2 for it in iters[-3:])
    tiny = pens[-1] < 1.0e-3
    print(f"continuation_every_decade_converged={all_conv}")
    print(f"continuation_penetration_shrinks_monotonically={shrinks}")
    print(f"continuation_last_three_decades_take_at_most_two_iterations="
          f"{cheap}")
    print(f"continuation_final_penetration_is_negligible={tiny}")

    if (r_slot == -9 and it_slot == 1 and all_conv and shrinks and cheap
            and tiny):
        print("VERDICT=continuation_reaches_what_a_cold_start_cannot")
        return 0
    print("VERDICT=cold_start_at_stiff_penalty_worked")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
