"""Tier-2 for fenics reaction_diffusion#4: a nonlinear reaction term needs a
true Newton solve. Lagging one factor of the nonlinearity (Picard:
-0.01*lap(u_new) + 5*u_old*u_new = 1, re-solved with LinearProblem) is far worse
than it looks.

Steady problem: -0.01*lap(u) + 5*u^2 = 1 on a 24x24 unit square, u = 1 on the
whole boundary, initial guess u = 1. The residual reported for both methods is
the SAME quantity: the l2 norm of the true nonlinear residual with the Dirichlet
rows zeroed.

Observed on dolfinx 0.10.0:
  Newton through NonlinearProblem converges in 5 iterations with the history
  1.597e-01, 2.879e-02, 2.766e-03, 4.657e-05, 1.657e-08, 2.460e-15 -- every step
  satisfies r_{k+1} < r_k**1.5 once r < 0.1.
  200 lagged (Picard) iterations end at 2.968e-09, still six orders above the
  Newton answer after forty times the work, and both are converging on the same
  solution (max difference below 1e-3).
  NOTE on the ratio: measured on the true nonlinear residual with the Dirichlet
  rows zeroed, the Picard ratio is a STEADY 0.920 per iteration. Neither the
  "steady ~0.5" of the original claim nor the "alternating 0.24 / 2.78" of its
  correction reproduced here; only the 0.92 envelope did.

Mutation control: T2_MUTATE=1 runs Newton in the lagged slot as well.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402

import dolfinx  # noqa: E402
import dolfinx.fem.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from petsc4py import PETSc  # noqa: E402

N, KAPPA, RATE, NPIC = 24, 0.01, 5.0, 200


def setup():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 1.0), dofs, V)
    return msh, V, bc, dofs


def residual_norm(u, V, dofs) -> float:
    """l2 norm of the true nonlinear residual, Dirichlet rows removed."""
    v = ufl.TestFunction(V)
    F = (KAPPA * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + RATE * u * u * v * ufl.dx - 1.0 * v * ufl.dx)
    r = dolfinx.fem.assemble_vector(dolfinx.fem.form(F))
    r.scatter_reverse(dolfinx.la.InsertMode.add)
    arr = r.array.copy()
    arr[dofs] = 0.0
    return float(np.linalg.norm(arr))


def newton():
    msh, V, bc, dofs = setup()
    u = dolfinx.fem.Function(V)
    u.x.array[:] = 1.0
    v = ufl.TestFunction(V)
    F = (KAPPA * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + RATE * u * u * v * ufl.dx - 1.0 * v * ufl.dx)
    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, u, bcs=[bc], petsc_options_prefix="t2_rd4_newton_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_rtol": 1.0e-12, "snes_atol": 1.0e-14})
    prob.solver.setConvergenceHistory()
    prob.solve()
    u.x.scatter_forward()
    hist = np.array(prob.solver.getConvergenceHistory()[0], dtype=float)
    return (prob.solver.getIterationNumber(), hist,
            residual_norm(u, V, dofs), u.x.array.copy())


def lagged():
    """Picard: lag one factor of u^2. T2_MUTATE=1 runs Newton here instead."""
    if MUTATE:
        its, _, res, arr = newton()
        return its, [res], res, arr
    msh, V, bc, dofs = setup()
    u_old = dolfinx.fem.Function(V)
    u_old.x.array[:] = 1.0
    u_new = dolfinx.fem.Function(V)
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = (KAPPA * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + RATE * u_old * u * v * ufl.dx)
    L = 1.0 * v * ufl.dx
    prob = dolfinx.fem.petsc.LinearProblem(
        a, L, bcs=[bc], u=u_new, petsc_options_prefix="t2_rd4_picard_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    hist = []
    for _ in range(NPIC):
        prob.solve()
        u_new.x.scatter_forward()
        u_old.x.array[:] = u_new.x.array
        hist.append(residual_norm(u_old, V, dofs))
    return NPIC, hist, hist[-1], u_old.x.array.copy()


def main() -> int:
    n_its, n_hist, n_res, n_u = newton()
    p_its, p_hist, p_res, p_u = lagged()

    print(f"newton_iterations={n_its} newton_final_residual={n_res:.3e}")
    print("newton_history=" + " ".join(f"{r:.3e}" for r in n_hist))
    print(f"lagged_iterations={p_its} lagged_final_residual={p_res:.3e}")
    print("lagged_last_5=" + " ".join(f"{r:.3e}" for r in p_hist[-5:]))

    # super-linear: once r < 0.1 every step satisfies r_{k+1} < r_k**1.5
    quad = bool(len(n_hist) >= 4 and all(
        n_hist[k + 1] < n_hist[k] ** 1.5
        for k in range(1, len(n_hist) - 1) if n_hist[k] < 0.1))
    print(f"newton_converged_in_under_10_iterations={n_its < 10}")
    print(f"newton_final_residual_below_1e-12={n_res < 1e-12}")
    print(f"newton_residual_exponent_at_least_1p5_each_step={quad}")

    stalled = p_res > 1e-10
    same_answer = bool(np.max(np.abs(p_u - n_u)) < 1e-3)
    print(f"lagged_and_newton_head_for_the_same_solution={same_answer}")
    print(f"lagged_residual_after_200_iterations_still_above_1e-10={stalled}")
    if len(p_hist) > 20:
        rat = np.array(p_hist[-11:-1]) / np.array(p_hist[-12:-2])
        env = (p_hist[-1] / p_hist[-21]) ** (1.0 / 20.0)
        steady = bool(rat.max() - rat.min() < 0.05)
        slow_env = bool(0.85 < env < 0.98)
        print(f"lagged_ratio_min={rat.min():.3f} max={rat.max():.3f} "
              f"envelope_per_iteration={env:.3f}")
        print(f"lagged_ratio_is_steady_not_alternating={steady}")
        print(f"lagged_envelope_shrinks_slower_than_0p98_per_iteration="
              f"{slow_env}")
    else:
        steady = slow_env = False
        print("lagged_ratio_is_steady_not_alternating=False")
        print("lagged_envelope_shrinks_slower_than_0p98_per_iteration=False")

    if (n_its < 10 and n_res < 1e-12 and quad and stalled and steady
            and slow_env):
        print("VERDICT=lagging_the_reaction_stalls_where_newton_is_quadratic")
        return 0
    print("VERDICT=lagging_the_reaction_was_as_good_as_newton")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
