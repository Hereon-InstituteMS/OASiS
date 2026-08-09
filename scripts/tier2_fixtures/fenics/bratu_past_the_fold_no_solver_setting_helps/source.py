"""Tier-2 for fenics nonlinear_pde#4: a super-linear source (R(u) = lambda*exp(u))
has a load beyond which NO steady solution exists, and past it Newton cannot be
made to converge by any solver setting.

Wrong variant: solve Bratu, -div(grad u) = lambda*exp(u) with u = 0 on the whole
boundary, straight at lambda = 20 on the unit square and try to rescue it with a
bigger iteration budget or a different line search.

Observed on dolfinx 0.10.0 (16x16, P1):
  * lambda = 1 converges in a couple of iterations to max(u) ~ 7.8e-02.
  * lambda = 20 fails for every combination tried: budget 30 and budget 200,
    line search 'basic' and 'bt'. 'basic' burns the whole budget (reason -5,
    DIVERGED_MAX_IT, at 30 and again at 200 iterations); 'bt' stops itself with
    reason -6 (DIVERGED_LINE_SEARCH) after the same 23 iterations whatever the
    budget.
  * load continuation, reusing the previous solution as the initial iterate,
    walks up to lambda = 6 and then fails at lambda = 7 -- the turning point of
    the continuous problem, which no solver option can move.

Mutation control: T2_MUTATE=1 puts the target load below the fold (lambda = 1),
where every arm converges and the continuation reaches the target.
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

N = 16
BIG = 1.0 if MUTATE else 20.0
SMALL = 1.0
STEPS = 20


def build():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), dofs, V)
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    lam = dolfinx.fem.Constant(msh, SMALL)
    F = (ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - lam * ufl.exp(u) * v * ufl.dx)
    return msh, u, v, lam, F, bc


def one_shot(tag: str, lmbda: float, maxit: int, ls: str):
    _, u, _, lam, F, bc = build()
    lam.value = lmbda
    problem = dolfinx.fem.petsc.NonlinearProblem(
        F, u, bcs=[bc], petsc_options_prefix=f"t2_np4_{tag}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_max_it": maxit, "snes_linesearch_type": ls})
    problem.solve()
    r = problem.solver.getConvergedReason()
    it = problem.solver.getIterationNumber()
    print(f"one_shot lambda={lmbda:g} snes_max_it={maxit} linesearch={ls} "
          f"reason={r} iterations={it} max_u={float(u.x.array.max()):.4e} "
          f"finite={bool(np.all(np.isfinite(u.x.array)))}")
    return r, it


def continuation(target: float):
    _, u, _, lam, F, bc = build()
    problem = dolfinx.fem.petsc.NonlinearProblem(
        F, u, bcs=[bc], petsc_options_prefix="t2_np4_cont_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_max_it": 50})
    reached = 0.0
    for k in range(1, STEPS + 1):
        lam.value = target * k / STEPS
        keep = u.x.array.copy()
        problem.solve()
        r = problem.solver.getConvergedReason()
        if r > 0:
            reached = float(lam.value)
            u.x.scatter_forward()
            continue
        u.x.array[:] = keep
        print(f"continuation stopped: lambda={float(lam.value):.3f} reason={r}")
        break
    print(f"continuation_target={target:g} continuation_reached={reached:g}")
    return reached


def main() -> int:
    r_small, _ = one_shot("small", SMALL, 30, "bt")
    arms = [one_shot("big_bt_30", BIG, 30, "bt"),
            one_shot("big_basic_30", BIG, 30, "basic"),
            one_shot("big_bt_200", BIG, 200, "bt"),
            one_shot("big_basic_200", BIG, 200, "basic")]
    reasons = [r for r, _ in arms]
    its = [it for _, it in arms]
    print(f"big_load_reasons={reasons} big_load_iterations={its}")
    print(f"small_load_converges={r_small > 0}")
    all_fail = all(r < 0 for r in reasons)
    print(f"big_load_fails_in_every_arm={all_fail}")
    bigger_budget = reasons[2] < 0 and reasons[3] < 0
    other_linesearch = reasons[0] < 0 and reasons[1] < 0
    print(f"raising_snes_max_it_does_not_help={bigger_budget}")
    print(f"changing_the_line_search_does_not_help={other_linesearch}")

    reached = continuation(BIG)
    stalled = reached < BIG - 1.0e-12
    print(f"continuation_stalls_below_the_target={stalled}")

    if r_small > 0 and all_fail and stalled:
        print("VERDICT=past_the_fold_no_solver_setting_converges")
        return 0
    print("VERDICT=the_load_was_reachable")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
