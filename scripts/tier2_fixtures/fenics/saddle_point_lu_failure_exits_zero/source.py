"""Tier-2 for fenics mixed_poisson#1: a direct solve of this saddle point can
fail while the script still exits 0. Nothing is printed, the return code is
zero, the Function is populated — and ksp.getConvergedReason() is negative and
the vector is full of inf. Always test getConvergedReason() > 0, and note that
a converged reason is still not proof of a correct answer here.

Wrong variant: solve, then read the solution without looking at the reason.
Right variant: check the reason, and set ksp.setErrorIfNotConverged(True) so
the underlying cause is raised instead of hidden.

RT1 x DG0 on an 8x8 unit square with the flux prescribed on the whole boundary,
factorised by PETSc's own LU (pc_factor_mat_solver_type=petsc). Observed on
dolfinx 0.10.0 / PETSc 3.24.5: getConvergedReason() returns -11
(KSP_DIVERGED_PC_FAILED), all 128 pressure dofs are inf, and the script would
have exited 0 with that in hand. With setErrorIfNotConverged(True) the same
solve raises and the chained petsc4py error carries the real cause,
"Zero pivot in LU factorization" and "Zero pivot row 0 value 0. tolerance
2.22045e-14", through PCSetUp_LU and MatLUFactorNumeric_SeqAIJ. Note this is
factoriser-dependent: MUMPS on this identical matrix does not fail, it returns
reason 4 and a pressure of order 1e15, which is the previous pitfall.

Mutation control: T2_MUTATE=1 switches to the pivoting MUMPS factoriser AND
leaves part of the boundary to the natural pressure condition — both are needed,
because PETSc's own LU does no pivoting and stops on the zero pressure-block
diagonal even when the problem is well posed. With the mutation the solve
converges, the vector is finite and the fixture loses its own expectations.
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

import basix.ufl  # noqa: E402
from petsc4py import PETSc  # noqa: E402

N = 8
DEGREE = 1
# The wrong variant is PETSc's own non-pivoting LU on the singular saddle
# point; the right one is a pivoting factoriser on a well-posed problem.
FACTORISER = "mumps" if MUTATE else "petsc"


def build(flux_everywhere: bool):
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    RT = basix.ufl.element("RT", msh.basix_cell(), DEGREE)
    DG = basix.ufl.element("DG", msh.basix_cell(), DEGREE - 1)
    W = dolfinx.fem.functionspace(msh, basix.ufl.mixed_element([RT, DG]))
    (sig, u) = ufl.TrialFunctions(W)
    (tau, v) = ufl.TestFunctions(W)
    x = ufl.SpatialCoordinate(msh)
    f = 10.0 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    n = ufl.FacetNormal(msh)
    a = (ufl.inner(sig, tau) + ufl.div(tau) * u + ufl.div(sig) * v) * ufl.dx
    L = -f * v * ufl.dx
    V0, _ = W.sub(0).collapse()
    g = dolfinx.fem.Function(V0)
    g.x.array[:] = 0.0
    if flux_everywhere:
        facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    else:
        facets = dolfinx.mesh.locate_entities_boundary(
            msh, fdim, lambda X: np.isclose(X[0], 0.0) | np.isclose(X[0], 1.0))
        natural = dolfinx.mesh.locate_entities_boundary(
            msh, fdim, lambda X: np.isclose(X[1], 0.0) | np.isclose(X[1], 1.0))
        tags = dolfinx.mesh.meshtags(
            msh, fdim, np.sort(natural),
            np.full(len(natural), 1, dtype=np.int32))
        ds = ufl.Measure("ds", domain=msh, subdomain_data=tags)
        L = L - ufl.sin(5.0 * x[0]) * ufl.dot(tau, n) * ds(1)
    bcs = [dolfinx.fem.dirichletbc(
        g, dolfinx.fem.locate_dofs_topological((W.sub(0), V0), fdim, facets),
        W.sub(0))]
    af, Lf = dolfinx.fem.form(a), dolfinx.fem.form(L)
    A = dolfinx.fem.petsc.assemble_matrix(af, bcs=bcs)
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(Lf)
    dolfinx.fem.petsc.apply_lifting(b, [af], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.petsc.set_bc(b, bcs)
    return msh, W, A, b


def lu_solve(msh, W, A, b, raise_on_failure: bool, factoriser: str):
    ksp = PETSc.KSP().create(msh.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType(factoriser)
    if raise_on_failure:
        ksp.setErrorIfNotConverged(True)
    w = dolfinx.fem.Function(W)
    ksp.solve(b, w.x.petsc_vec)
    w.x.scatter_forward()
    return int(ksp.getConvergedReason()), np.array(w.sub(1).collapse().x.array)


def main() -> int:
    msh, W, A, b = build(flux_everywhere=not MUTATE)
    quiet_raised = ""
    try:
        reason, p = lu_solve(msh, W, A, b, raise_on_failure=False,
                             factoriser=FACTORISER)
    except Exception as exc:
        quiet_raised = f"{type(exc).__name__}: {exc}"
        reason, p = 0, np.array([np.nan])
    print(f"primary_reason={reason} "
          f"quiet_solve_raised={quiet_raised[:60]!r} "
          f"inf_entries={int(np.sum(~np.isfinite(p)))} of {p.size}")

    loud = ""
    msh2, W2, A2, b2 = build(flux_everywhere=not MUTATE)
    try:
        r2, _ = lu_solve(msh2, W2, A2, b2, raise_on_failure=True,
                         factoriser=FACTORISER)
        print(f"loud_solve_reason={r2} loud_solve_raised=False")
    except Exception as exc:
        loud = str(exc)
        print(f"loud_solve_raised=True {type(exc).__name__}")
        print("petsc_error_chain>>>")
        print(loud)
        print("<<<petsc_error_chain")

    is_pc_failed = reason == int(
        PETSc.KSP.ConvergedReason.DIVERGED_PCSETUP_FAILED)
    all_inf = p.size > 0 and bool(np.all(np.isinf(p)))
    silent = quiet_raised == ""
    loud_raised = bool(loud)
    print(f"quiet_solve_raised_nothing={silent}")
    print(f"converged_reason_is_diverged_pcsetup_failed_minus_eleven="
          f"{is_pc_failed and reason == -11}")
    print(f"solution_vector_is_entirely_inf={all_inf}")
    print(f"set_error_if_not_converged_surfaces_the_cause={loud_raised}")
    if silent and is_pc_failed and reason == -11 and all_inf and loud_raised:
        print("VERDICT=direct_solve_failed_silently_and_the_script_would_exit_zero")
        return 0
    print("VERDICT=direct_solve_did_not_fail_silently")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
