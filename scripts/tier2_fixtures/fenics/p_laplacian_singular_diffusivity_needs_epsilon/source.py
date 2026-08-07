"""Tier-2 for fenics nonlinear_pde#2: a diffusivity that is singular at the
starting iterate must be regularised. The p-Laplacian D = |grad u|^(p-2) with
p = 1.5, evaluated at the zero initial iterate, is 0^(-1/4) = inf.

Wrong variant: D = (grad u . grad u)**((p-2)/2) with u starting at 0.
Right variant: D = (grad u . grad u + eps**2)**((p-2)/2), eps = 1e-6.

Observed on dolfinx 0.10.0:
  * neither assembling the residual nor assembling the Jacobian raises anything
    -- in particular no ZeroDivisionError -- but the assembled residual vector
    is entirely NaN and the Jacobian assembled at u = 0 DOES contain NaN
    entries (its PETSc norm is nan). The claim's remark that 'nan entries in J'
    do not reproduce is wrong; what does not reproduce is an exception.
  * SNES returns converged reason -4 (DIVERGED_FNORM_NAN) at iteration 0 and
    leaves u exactly at zero, so np.isnan(u.x.array).any() is False: a NaN check
    on the solution does not catch the failure.
  * the eps-regularised form converges in a handful of iterations with a
    positive reason.

Mutation control: T2_MUTATE=1 writes the eps-regularised diffusivity in place of
the singular one, so the -4 reason, the NaN Jacobian and the frozen zero field
all disappear.
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

N, P_EXP, EPS = 16, 1.5, 1.0e-6
EPS_AS_WRITTEN = EPS if MUTATE else None


def build(eps):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), dofs, V)
    u = dolfinx.fem.Function(V)  # starts at exactly zero
    v = ufl.TestFunction(V)
    g2 = ufl.inner(ufl.grad(u), ufl.grad(u))
    D = g2 ** ((P_EXP - 2) / 2) if eps is None else \
        (g2 + eps ** 2) ** ((P_EXP - 2) / 2)
    F = (D * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
         - dolfinx.fem.Constant(msh, 1.0) * v * ufl.dx)
    return msh, u, F, bc


def run(tag: str, eps):
    msh, u, F, bc = build(eps)
    problem = dolfinx.fem.petsc.NonlinearProblem(
        F, u, bcs=[bc], petsc_options_prefix=f"t2_np2_{tag}_",
        petsc_options={"ksp_type": "preonly", "pc_type": "lu",
                       "snes_max_it": 30})
    problem.solve()
    arr = u.x.array.copy()
    return (problem.solver.getConvergedReason(),
            problem.solver.getIterationNumber(), arr)


def assemble_probe(eps):
    """Assemble residual and Jacobian at the zero iterate."""
    _, u, F, _ = build(eps)
    raised = ""
    nan_b = nan_J = -1
    try:
        b = dolfinx.fem.assemble_vector(dolfinx.fem.form(F))
        nan_b = int(np.isnan(b.array).sum())
        A = dolfinx.fem.petsc.assemble_matrix(
            dolfinx.fem.form(ufl.derivative(F, u)))
        A.assemble()
        dense = A.copy().convert("dense").getDenseArray()
        nan_J = int(np.isnan(dense).sum())
    except Exception as exc:
        raised = f"{type(exc).__name__}: {exc}"
    return raised, nan_b, nan_J


def main() -> int:
    print(f"p={P_EXP} eps_as_written={EPS_AS_WRITTEN}")
    raised, nan_b, nan_J = assemble_probe(EPS_AS_WRITTEN)
    print(f"assembly_raised={raised or 'nothing'}")
    print(f"assembly_raised_an_exception={bool(raised)}")
    print(f"nan_entries_in_residual={nan_b} nan_entries_in_jacobian={nan_J}")
    print(f"jacobian_at_the_start_has_nan_entries={nan_J > 0}")

    reason, its, arr = run("aswritten", EPS_AS_WRITTEN)
    print(f"as_written_reason={reason} iterations={its} "
          f"abs_max_u={float(np.abs(arr).max()):.4e}")
    fnorm_nan = reason == -4
    frozen = bool(np.all(arr == 0.0))
    no_nan_in_u = not bool(np.isnan(arr).any())
    print(f"unregularised_reason_is_fnorm_nan={fnorm_nan}")
    print(f"solution_left_exactly_at_zero={frozen}")
    print(f"nan_check_on_the_solution_misses_it="
          f"{fnorm_nan and frozen and no_nan_in_u}")

    reason_r, its_r, arr_r = run("regularised", EPS)
    print(f"regularised_reason={reason_r} iterations={its_r} "
          f"abs_max_u={float(np.abs(arr_r).max()):.4e}")
    fixed = reason_r > 0 and float(np.abs(arr_r).max()) > 0.0
    print(f"eps_regularisation_converges={fixed}")

    if fnorm_nan and frozen and no_nan_in_u and not raised and fixed:
        print("VERDICT=singular_diffusivity_dies_at_iteration_zero_"
              "with_no_nan_in_u")
        return 0
    print("VERDICT=singular_diffusivity_was_harmless")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
