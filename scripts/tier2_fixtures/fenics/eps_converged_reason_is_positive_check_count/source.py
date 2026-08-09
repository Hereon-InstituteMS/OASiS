"""Tier-2 for fenics eigenvalue#2: eps.setDimensions(nev, ncv) asks for nev
eigenvalues out of an ncv-dimensional search space, and the way to find out
whether you got them is the COUNT, not the reason code.

Wrong variant: treating a non-zero eps.getConvergedReason() as an error, the
way a return code from a C routine is usually read. Right variant:
eps.getConverged() >= nev.

Observed on dolfinx 0.10.0 / slepc4py 3.24.3, 16x16 P1 Dirichlet Laplacian
(A assembled with bcs, B the mass matrix with bcs and diag=0.0, shift-and-
invert): both (nev=4, ncv=8) and (nev=4, ncv=5) return nconv=4 with
getConvergedReason() == 1, and they return the same four eigenvalues; the tight
ncv only costs iterations (52 instead of 7). So "reason != 0 means failure"
flags a perfectly good solve as a failure. The enum confirms the sign
convention: CONVERGED_TOL is +1 while DIVERGED_ITS, DIVERGED_BREAKDOWN and
DIVERGED_SYMMETRY_LOST are -1, -2, -3.

Mutation control: T2_MUTATE=1 makes the primary health check the correct
nconv >= nev test, which does not call the solve a failure, and the fixture
loses its own expectation.
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
from slepc4py import SLEPc  # noqa: E402

N = 16
NEV = 4


def matrices():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    bfacets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(msh, 0.0),
        dolfinx.fem.locate_dofs_topological(V, fdim, bfacets), V)
    a = dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    m = dolfinx.fem.form(u * v * ufl.dx)
    A = dolfinx.fem.petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    B = dolfinx.fem.petsc.assemble_matrix(m, bcs=[bc], diag=0.0)
    B.assemble()
    return msh, A, B


def solve(msh, A, B, ncv):
    eps = SLEPc.EPS().create(msh.comm)
    eps.setOperators(A, B)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setDimensions(NEV, ncv if ncv else PETSc.DECIDE)
    eps.setTolerances(1.0e-10, 1000)
    eps.setTarget(0.0)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.getST().setType(SLEPc.ST.Type.SINVERT)
    eps.solve()
    n = eps.getConverged()
    vals = np.array(sorted(eps.getEigenvalue(i).real for i in range(n)))
    return int(eps.getConvergedReason()), int(n), vals, eps.getIterationNumber()


def main() -> int:
    msh, A, B = matrices()
    r8, n8, v8, i8 = solve(msh, A, B, 8)
    r5, n5, v5, i5 = solve(msh, A, B, 5)
    print(f"nev={NEV} ncv=8: reason={r8} nconv={n8} its={i8} "
          f"eigenvalues={np.round(v8[:NEV], 5).tolist()}")
    print(f"nev={NEV} ncv=5: reason={r5} nconv={n5} its={i5} "
          f"eigenvalues={np.round(v5[:NEV], 5).tolist()}")
    print(f"CONVERGED_TOL={int(SLEPc.EPS.ConvergedReason.CONVERGED_TOL)} "
          f"DIVERGED_ITS={int(SLEPc.EPS.ConvergedReason.DIVERGED_ITS)} "
          f"DIVERGED_BREAKDOWN={int(SLEPc.EPS.ConvergedReason.DIVERGED_BREAKDOWN)} "
          f"DIVERGED_SYMMETRY_LOST="
          f"{int(SLEPc.EPS.ConvergedReason.DIVERGED_SYMMETRY_LOST)}")

    # The health check under test.
    if MUTATE:
        called_failure = not (n8 >= NEV)          # correct test
    else:
        called_failure = (r8 != 0)                # "error code != 0" test

    positive_on_success = (
        r8 == int(SLEPc.EPS.ConvergedReason.CONVERGED_TOL) == 1)
    negatives = all(int(c) < 0 for c in (
        SLEPc.EPS.ConvergedReason.DIVERGED_ITS,
        SLEPc.EPS.ConvergedReason.DIVERGED_BREAKDOWN,
        SLEPc.EPS.ConvergedReason.DIVERGED_SYMMETRY_LOST))
    both_ok = n8 >= NEV and n5 >= NEV
    same = bool(np.allclose(v8[:NEV], v5[:NEV], rtol=1.0e-8))

    print(f"successful_solve_reports_reason_code_positive_one={positive_on_success}")
    print(f"every_documented_failure_code_is_negative={negatives}")
    print(f"both_ncv_settings_returned_nconv_at_least_nev={both_ok}")
    print(f"tight_ncv_returned_the_same_eigenvalues={same}")
    print(f"primary_health_check_calls_this_solve_a_failure={called_failure}")
    if positive_on_success and negatives and both_ok and same and called_failure:
        print("VERDICT=reason_code_is_positive_on_success_so_check_the_count")
        return 0
    print("VERDICT=reason_code_test_behaved_as_a_c_error_code")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
