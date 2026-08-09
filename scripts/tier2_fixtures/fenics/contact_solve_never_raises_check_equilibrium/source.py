"""Tier-2 for fenics contact#6: problem.solve() on a NonlinearProblem never
raises on non-convergence, so an unchecked contact solve silently returns a
body that has passed through the obstacle.

Wrong variant: the identical script run twice, once with a sane penalty and
once with a penalty stiff enough to break SNES, with nothing between the solve
and the output but a print. Neither call raises. Both fields are finite, both
have the shape a contact solution is expected to have (negative under the load,
zero on the clamped boundary), and their magnitudes are within a factor of four
of each other — nothing in the field itself says which one is the answer.

What does distinguish them is measured here two ways, both of which the caller
has to do on purpose:
  * problem.solver.getConvergedReason(): 2 (CONVERGED_FNORM_ABS) against -9
    (DIVERGED_DTOL);
  * equilibrium. The discrete residual on the unconstrained dofs, normalised by
    the load vector, is ~1e-13 for the converged solve and ~1e12 for the
    diverged one; and the total penalty force integral(gamma*max(phi - u, 0)dx)
    against the applied load |integral(f dx)| is of order one when the solve is
    good and ten orders of magnitude too large when it is not.

Mutation control: T2_MUTATE=1 gives the second run the sane penalty too, so
both solves converge and the "diverged but silent" expectations are lost.
"""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("XDG_CACHE_HOME", tempfile.mkdtemp(prefix="ffcx_t2_"))

import numpy as np  # noqa: E402
import ufl  # noqa: E402
from mpi4py import MPI  # noqa: E402
from petsc4py import PETSc  # noqa: E402

from dolfinx import fem, mesh  # noqa: E402
import dolfinx.fem.petsc as dfp  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

N = 24
PHI = -0.2
GOOD, BAD = 1.0e5, 1.0e12


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
    gamma = fem.Constant(msh, GOOD)
    f = fem.Constant(msh, -10.0)
    gap = ufl.max_value(phi - u, 0.0)
    F = (ufl.dot(ufl.grad(u), ufl.grad(v)) - f * v - gamma * gap * v) * ufl.dx
    F_form = fem.form(F)
    load_form = fem.form(f * v * ufl.dx)
    contact_force_form = fem.form(gamma * gap * ufl.dx)
    applied_load_form = fem.form(f * ufl.dx)

    problem = dfp.NonlinearProblem(
        F, u, bcs=[bc], petsc_options_prefix="t2_eq_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_rtol": 1e-9, "snes_atol": 1e-10,
                       "snes_max_it": 30, "snes_linesearch_type": "basic"})

    n_local = V.dofmap.index_map.size_local
    free = np.setdiff1d(np.arange(n_local), dofs)

    def residual_norm(form):
        r = dfp.assemble_vector(form)
        r.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)
        return float(np.linalg.norm(r.array[free]))

    def run(g):
        u.x.array[:] = 0.0
        gamma.value = float(g)
        raised = ""
        try:
            problem.solve()
        except Exception as exc:                        # pragma: no cover
            raised = f"{type(exc).__name__}: {exc}"
        arr = u.x.array
        eq = residual_norm(F_form) / residual_norm(load_form)
        cf = abs(fem.assemble_scalar(contact_force_form))
        ap = abs(fem.assemble_scalar(applied_load_form))
        return {
            "reason": problem.solver.getConvergedReason(),
            "raised": raised,
            "min": float(arr.min()), "max": float(arr.max()),
            "finite": bool(np.all(np.isfinite(arr))),
            "eq": eq, "force_ratio": cf / ap,
        }

    good = run(GOOD)
    bad = run(GOOD if MUTATE else BAD)
    for tag, d in (("converged_run", good), ("second_run", bad)):
        print(f"{tag}: reason={d['reason']} raised={d['raised']!r} "
              f"min_u={d['min']:.6e} max_u={d['max']:.6e} "
              f"finite={d['finite']} equilibrium_residual={d['eq']:.6e} "
              f"contact_force_over_applied_load={d['force_ratio']:.6e}")

    both_quiet = good["raised"] == "" and bad["raised"] == ""
    print(f"both_solves_raised_nothing={both_quiet}")
    print(f"second_run_reason_is_negative={bad['reason'] < 0}")
    plausible = (bad["finite"] and bad["min"] < 0.0
                 and abs(bad["min"]) < 4.0 * abs(good["min"]))
    print(f"second_run_field_looks_plausible={plausible}")
    print(f"equilibrium_holds_for_converged_run={good['eq'] < 1.0e-8}")
    print(f"equilibrium_fails_for_second_run={bad['eq'] > 1.0e3}")
    print(f"contact_force_balances_load_for_converged_run="
          f"{good['force_ratio'] < 10.0}")
    print(f"contact_force_absurd_for_second_run="
          f"{bad['force_ratio'] > 1.0e3}")

    if (both_quiet and bad["reason"] < 0 and good["reason"] > 0 and plausible
            and good["eq"] < 1.0e-8 and bad["eq"] > 1.0e3
            and good["force_ratio"] < 10.0 and bad["force_ratio"] > 1.0e3):
        print("VERDICT=solve_is_silent_only_reason_and_equilibrium_tell_you")
        return 0
    print("VERDICT=divergence_was_reported_by_solve")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
