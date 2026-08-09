"""Tier-2 for fenics eigenvalue#3: in a generalised problem A x = lambda B x
with Dirichlet BCs, A and the mass matrix B must not both get the SAME
Dirichlet diagonal, or every constrained dof contributes a spurious eigenvalue
at A_ii / B_ii.

Wrong variant: assemble_matrix(a, bcs=[bc]) and assemble_matrix(m, bcs=[bc]),
both leaving diag at its default of 1. Right variant: keep A's diagonal at 1
and assemble B with diag=0.0, with the spectral transform set to
SLEPc.ST.Type.SINVERT.

8x8 P1 unit square: 81 dofs, 32 of them constrained, 49 free. The reference is
a dense scipy.linalg.eigvalsh of the interior blocks of the unconstrained A and
B, i.e. the exactly reduced problem.

Observed on dolfinx 0.10.0 / slepc4py 3.24.3: with both diagonals at 1 the
lowest reported eigenvalues come back as 1.0, 1.0, 1.0, 1.0, 1.0 and only then
the physical 20.5055449, 52.6297923, ... — the spurious modes sit at
lambda = 1, NOT at lambda = 0. With diag=0.0 on B the returned values match the
reduced interior problem to 1.3e-14. The keyword really is `diag`: passing
`diagonal=0.0` raises "TypeError: assemble_matrix() got an unexpected keyword
argument 'diagonal'".

Mutation control: T2_MUTATE=1 assembles the primary B with diag=0.0, the
lambda = 1 cluster disappears and the fixture loses its own expectation.
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

import scipy.linalg as sla  # noqa: E402
from petsc4py import PETSc  # noqa: E402
from slepc4py import SLEPc  # noqa: E402

N = 8
NEV = 8


def setup():
    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    bfacets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, fdim, bfacets)
    bc = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(msh, 0.0), dofs, V)
    a = dolfinx.fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
    m = dolfinx.fem.form(u * v * ufl.dx)
    ndof = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
    return msh, a, m, bc, np.asarray(dofs), ndof


def reduced_spectrum(a, m, dofs, ndof):
    """Dense eigenvalues of the exactly reduced interior-dof problem."""
    A0 = dolfinx.fem.petsc.assemble_matrix(a)
    A0.assemble()
    B0 = dolfinx.fem.petsc.assemble_matrix(m)
    B0.assemble()
    Ad = A0.copy().convert("dense").getDenseArray().copy()
    Bd = B0.copy().convert("dense").getDenseArray().copy()
    free = np.setdiff1d(np.arange(ndof), dofs)
    return np.sort(sla.eigvalsh(Ad[np.ix_(free, free)],
                                Bd[np.ix_(free, free)])), free.size


def solve(msh, A, B, nev=NEV):
    eps = SLEPc.EPS().create(msh.comm)
    eps.setOperators(A, B)
    eps.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eps.setDimensions(nev, PETSc.DECIDE)
    eps.setTolerances(1.0e-10, 2000)
    eps.setTarget(0.0)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
    eps.getST().setType(SLEPc.ST.Type.SINVERT)
    eps.solve()
    n = eps.getConverged()
    return np.array(sorted(eps.getEigenvalue(i).real for i in range(n)))


def main() -> int:
    msh, a, m, bc, dofs, ndof = setup()
    reduced, nfree = reduced_spectrum(a, m, dofs, ndof)
    print(f"ndof={ndof} constrained={dofs.size} free={nfree}")
    print(f"reduced_interior_eigenvalues="
          f"{np.round(reduced[:5], 7).tolist()}")

    A = dolfinx.fem.petsc.assemble_matrix(a, bcs=[bc])
    A.assemble()
    B_one = dolfinx.fem.petsc.assemble_matrix(m, bcs=[bc])          # diag=1
    B_one.assemble()
    B_zero = dolfinx.fem.petsc.assemble_matrix(m, bcs=[bc], diag=0.0)
    B_zero.assemble()

    primary = solve(msh, A, B_zero if MUTATE else B_one)
    clean = solve(msh, A, B_zero)
    print(f"primary_eigenvalues={np.round(primary[:8], 10).tolist()}")
    print(f"clean_recipe_eigenvalues={np.round(clean[:5], 7).tolist()}")

    at_one = np.abs(primary - 1.0) < 1.0e-9
    n_at_one = int(np.sum(at_one))
    lowest_are_one = n_at_one > 0 and bool(np.all(at_one[:n_at_one]))
    none_at_zero = bool(np.all(np.abs(primary) > 1.0e-6))
    clean_ok = bool(np.max(np.abs(clean[:5] - reduced[:5])
                           / reduced[:5]) < 1.0e-10)
    print(f"primary_eigenvalues_equal_to_one={n_at_one}")

    kwargs_err = ""
    try:
        dolfinx.fem.petsc.assemble_matrix(m, bcs=[bc], diagonal=0.0)
    except TypeError as exc:
        kwargs_err = str(exc)
    print(f"diagonal_kwarg_error={kwargs_err}")

    print(f"primary_lowest_eigenvalues_are_exactly_one={lowest_are_one}")
    print(f"no_spurious_mode_sits_at_zero={none_at_zero}")
    print(f"clean_recipe_matches_the_reduced_interior_problem={clean_ok}")
    print(f"diagonal_kwarg_raises_typeerror={bool(kwargs_err)}")
    if lowest_are_one and none_at_zero and clean_ok and kwargs_err:
        print("VERDICT=shared_dirichlet_diagonal_puts_spurious_modes_at_lambda_one")
        return 0
    print("VERDICT=shared_dirichlet_diagonal_was_harmless")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
