"""Tier-2 for fenics nonlinear_pde#9: the four reference-free checks that catch a
wrong nonlinear solve -- (1) the Galerkin residual must vanish at every
UNCONSTRAINED dof, (2) the energy identity int D(u)|grad u|^2 dx = int f*u dx
(v = u is admissible because u vanishes on the Dirichlet boundary), (3) the
maximum principle u >= 0 for f > 0 with homogeneous data, (4) a non-zero spread
of D(u), without which the run was effectively linear.

Wrong variant: take the field back from a solve that ran out of iterations
(snes_max_it = 1) and check only that it is finite, non-negative and that the
diffusivity varies -- which it is, and which it does.

Observed on dolfinx 0.10.0 with D(u) = 1 + u^2, f = 10, 16x16 P1, u = 0 on the
whole boundary:
  * converged run: residual over the free dofs of order 1e-11, energy identity
    closing to a relative gap of order 1e-12, u in [0, 6.46e-01], D(u) in
    [1.000000, 1.410773].
  * truncated run (reason -5): u is still non-negative and D(u) still varies, so
    checks (3) and (4) pass, but the free-dof residual is of order 1e-2 and the
    energy identity is off by ~9e-02 relative -- the two checks that carry the
    information catch it.

Mutation control: T2_MUTATE=1 gives the as-written solve a real iteration budget,
so the residual and energy checks stop catching anything.
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

N, SOURCE = 16, 10.0
BUDGET_GOOD = 30
BUDGET_AS_WRITTEN = BUDGET_GOOD if MUTATE else 1
RES_TOL, GAP_TOL = 1.0e-8, 1.0e-8


def solve_and_check(tag: str, maxit: int):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    bdofs = dolfinx.fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), bdofs, V)

    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    f = dolfinx.fem.Constant(msh, SOURCE)
    D = 1.0 + u ** 2
    F = D * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
    problem = dolfinx.fem.petsc.NonlinearProblem(
        F, u, bcs=[bc], petsc_options_prefix=f"t2_np9_{tag}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_max_it": maxit})
    problem.solve()
    u.x.scatter_forward()
    reason = problem.solver.getConvergedReason()

    # (1) Galerkin residual at the free dofs
    res = dolfinx.fem.assemble_vector(dolfinx.fem.form(F))
    res.scatter_reverse(dolfinx.la.InsertMode.add)
    r = res.array.copy()
    free = np.ones(r.size, dtype=bool)
    free[bdofs] = False
    res_free = float(np.abs(r[free]).max())

    # (2) energy identity with v = u
    lhs = float(dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(D * ufl.inner(ufl.grad(u), ufl.grad(u)) * ufl.dx)))
    rhs = float(dolfinx.fem.assemble_scalar(dolfinx.fem.form(f * u * ufl.dx)))
    gap = abs(lhs - rhs) / abs(rhs)

    # (3) maximum principle
    umin, umax = float(u.x.array.min()), float(u.x.array.max())

    # (4) spread of the diffusivity
    W = dolfinx.fem.functionspace(msh, ("Discontinuous Lagrange", 0))
    Dh = dolfinx.fem.Function(W)
    Dh.interpolate(dolfinx.fem.Expression(D, W.element.interpolation_points))
    dmin, dmax = float(Dh.x.array.min()), float(Dh.x.array.max())

    print(f"{tag}: snes_max_it={maxit} reason={reason} "
          f"res_free={res_free:.3e} energy_gap={gap:.3e} "
          f"u=[{umin:.4e}, {umax:.4e}] D=[{dmin:.6f}, {dmax:.6f}]")
    return dict(reason=reason, res=res_free, gap=gap, umin=umin,
                dspread=dmax - dmin)


def main() -> int:
    good = solve_and_check("converged", BUDGET_GOOD)
    aswr = solve_and_check("as_written", BUDGET_AS_WRITTEN)

    g_all = (good["res"] < RES_TOL and good["gap"] < GAP_TOL
             and good["umin"] >= 0.0 and good["dspread"] > 1.0e-3
             and good["reason"] > 0)
    print(f"converged_run_passes_all_four_checks={g_all}")

    caught_res = aswr["res"] > 1.0e-6
    caught_gap = aswr["gap"] > 1.0e-6
    print(f"truncated_solve_fails_the_residual_check={caught_res}")
    print(f"truncated_solve_fails_the_energy_identity={caught_gap}")
    blind = aswr["umin"] >= 0.0 and aswr["dspread"] > 1.0e-3
    print(f"maximum_principle_and_d_spread_do_not_catch_it={blind and caught_res}")
    print(f"as_written_reason_is_negative={aswr['reason'] < 0}")

    if g_all and caught_res and caught_gap and blind:
        print("VERDICT=residual_and_energy_identity_catch_the_truncated_solve")
        return 0
    print("VERDICT=the_checks_saw_nothing_wrong")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
