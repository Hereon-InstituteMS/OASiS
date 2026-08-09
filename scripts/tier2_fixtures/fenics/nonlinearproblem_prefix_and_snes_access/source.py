"""Tier-2 for fenics reaction_diffusion#10: dolfinx.fem.petsc.NonlinearProblem
takes petsc_options_prefix as a keyword-ONLY argument, is driven by calling
problem.solve() directly, and exposes a petsc4py SNES as problem.solver.

Wrong variants: constructing it without the prefix, and wrapping it in the
legacy dolfinx.nls.petsc.NewtonSolver.

Steady reaction-diffusion -0.01*lap(u) + 5*u^2 = 1 on an 8x8 unit square with
u = 1 on the whole boundary.

Observed on dolfinx 0.10.0:
  no prefix      -> TypeError: NonlinearProblem.__init__() missing 1 required
                    keyword-only argument: 'petsc_options_prefix'
  NewtonSolver   -> AttributeError: 'NonlinearProblem' object has no attribute
                    'a'
  correct        -> problem.solver is a petsc4py PETSc.SNES,
                    getConvergedReason() = 3 (CONVERGED_FNORM_RELATIVE) and
                    getIterationNumber() = 5.

Mutation control: T2_MUTATE=1 passes the prefix and never touches NewtonSolver.
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
import dolfinx.nls.petsc  # noqa: E402

MUTATE = os.environ.get("T2_MUTATE") == "1"

from petsc4py import PETSc  # noqa: E402

# An object whose __init__ dies before it assigns its PETSc handles prints an
# "Exception ignored in __del__" traceback when it is collected; class-level
# defaults keep each failure to the one exception this fixture is about.
dolfinx.nls.petsc.NewtonSolver._A = None
dolfinx.nls.petsc.NewtonSolver._b = None
for _attr in ("_snes", "_A", "_b", "_x", "_P_mat"):
    setattr(dolfinx.fem.petsc.NonlinearProblem, _attr, None)


def main() -> int:
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    u.x.array[:] = 1.0
    v = ufl.TestFunction(V)
    F = (0.01 * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
         + 5.0 * u * u * v * ufl.dx - 1.0 * v * ufl.dx)
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 1.0), dofs, V)

    prefix_msg = ""
    try:
        if MUTATE:
            dolfinx.fem.petsc.NonlinearProblem(
                F, u, bcs=[bc], petsc_options_prefix="t2_rd10_probe_")
        else:
            dolfinx.fem.petsc.NonlinearProblem(F, u, bcs=[bc])
        prefix_ok = True
    except TypeError as exc:
        prefix_ok = False
        prefix_msg = f"{type(exc).__name__}: {exc}"

    prob = dolfinx.fem.petsc.NonlinearProblem(
        F, u, bcs=[bc], petsc_options_prefix="t2_rd10_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

    legacy_msg = ""
    legacy_ok = True
    if not MUTATE:
        try:
            dolfinx.nls.petsc.NewtonSolver(msh.comm, prob)
        except AttributeError as exc:
            legacy_ok = False
            legacy_msg = f"{type(exc).__name__}: {exc}"

    if prefix_msg:
        print(f"no_prefix_error: {prefix_msg}")
    print(f"petsc_options_prefix_is_required={not prefix_ok}")
    if legacy_msg:
        print(f"newtonsolver_error: {legacy_msg}")
    print(f"legacy_newtonsolver_cannot_wrap_it={not legacy_ok}")

    prob.solve()
    u.x.scatter_forward()
    reason = prob.solver.getConvergedReason()
    its = prob.solver.getIterationNumber()
    print(f"solver_class={type(prob.solver).__name__} "
          f"converged_reason={reason} iterations={its} "
          f"u_range=[{u.x.array.min():.6f}, {u.x.array.max():.6f}]")
    is_snes = isinstance(prob.solver, PETSc.SNES)
    print(f"problem_solver_is_a_petsc4py_snes={is_snes}")
    print(f"converged_reason_is_positive={reason > 0}")
    print(f"iteration_count_is_reported={its > 0}")
    print(f"solution_is_finite={bool(np.all(np.isfinite(u.x.array)))}")

    if (not prefix_ok and not legacy_ok and is_snes and reason > 0
            and its > 0):
        print("VERDICT=nonlinearproblem_needs_the_prefix_and_owns_a_snes")
        return 0
    print("VERDICT=prefix_or_newtonsolver_was_accepted")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
