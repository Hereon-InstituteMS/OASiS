"""Tier-2 for fenics contact#1: a penalty stiffness that is too small gives a
WRONG answer with no error message at all — the most dangerous failure mode of
penalty contact.

Wrong variant: the scalar obstacle problem -laplacian(u) = f with f = -10 on
the unit square, u = 0 on the boundary, a flat obstacle at phi = -0.2 enforced
by the penalty residual -gamma*max(phi - u, 0)*v*dx, solved with gamma = 1 on a
24x24 triangle mesh (h = 1/24). gamma is a fem.Constant, so the stiff reference
run that follows differs in nothing else.

Observed: PETSc prints "Nonlinear t2_contact_ solve converged due to
CONVERGED_FNORM_ABS iterations 3", getConvergedReason() returns 2,
problem.solve() raises nothing — and the body has gone straight through the
obstacle: min(u) sits more than twelve element edges below phi. There is no
warning of any kind. The discrete equations are solved to machine precision;
it is the model that is wrong, which is why no solver diagnostic can catch it.
The same problem with gamma = 1e5 converges just as quietly with a penetration
of less than a hundredth of an element edge.

Mutation control: T2_MUTATE=1 puts the adequate penalty in the slot, so the
penetration collapses and the "converged and wrong" expectation is lost.
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
WEAK, STIFF = 1.0, 1.0e5


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
    gamma = fem.Constant(msh, WEAK)
    f = fem.Constant(msh, -10.0)
    gap = ufl.max_value(phi - u, 0.0)
    F = (ufl.dot(ufl.grad(u), ufl.grad(v)) - f * v - gamma * gap * v) * ufl.dx

    problem = dfp.NonlinearProblem(
        F, u, bcs=[bc], petsc_options_prefix="t2_contact_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_rtol": 1e-9, "snes_atol": 1e-10,
                       "snes_max_it": 30, "snes_linesearch_type": "basic",
                       "snes_converged_reason": None})

    def solve(g):
        u.x.array[:] = 0.0
        gamma.value = float(g)
        raised = ""
        try:
            problem.solve()
        except Exception as exc:                        # pragma: no cover
            raised = f"{type(exc).__name__}: {exc}"
        arr = u.x.array
        pen = max(0.0, PHI - float(arr.min()))
        return (problem.solver.getConvergedReason(),
                problem.solver.getIterationNumber(), pen / H, raised,
                float(arr.min()))

    r_slot, it_slot, pen_slot, raised, umin = solve(STIFF if MUTATE else WEAK)
    print(f"slot_reason={r_slot} iterations={it_slot} min_u={umin:.6e} "
          f"penetration_over_h={pen_slot:.4f}")
    print(f"slot_snes_reported_success={r_slot > 0}")
    print(f"slot_solve_raised_nothing={raised == ''}")
    print(f"slot_penetration_exceeds_ten_element_edges={pen_slot > 10.0}")

    r_ref, it_ref, pen_ref, _, umin_ref = solve(STIFF)
    print(f"stiff_reference_reason={r_ref} iterations={it_ref} "
          f"min_u={umin_ref:.6e} penetration_over_h={pen_ref:.6f}")
    print(f"stiff_reference_snes_reported_success={r_ref > 0}")
    print(f"stiff_reference_penetration_is_negligible={pen_ref < 0.1}")

    if (r_slot > 0 and raised == "" and pen_slot > 10.0
            and r_ref > 0 and pen_ref < 0.1):
        print("VERDICT=converged_reason_is_not_evidence_the_constraint_holds")
        return 0
    print("VERDICT=undersized_penalty_was_reported")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
