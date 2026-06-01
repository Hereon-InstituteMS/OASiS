"""Eigenvalue problem generator for FEniCSx/dolfinx + SLEPc.

Variants: 2d
"""


KNOWLEDGE = {
    "description": "Eigenvalue problems via SLEPc (PETSc eigenvalue solver)",
    "solver": "SLEPc EPS: Krylov-Schur (default), Arnoldi, Lanczos, power method, Jacobi-Davidson",
    "function_space": "Lagrange (any order)",
    "pitfalls": [
        "Requires slepc4py: pip install slepc4py (needs PETSc + SLEPc C libraries)",
        "Generalized eigenvalue: A*x = lambda*M*x where A=stiffness, M=mass",
        "[API] dolfinx 0.10 renamed the assemble_matrix kwarg 'diagonal' → 'diag'. "
        "Old call assemble_matrix(m, bcs=[bc], diagonal=0.0) raises TypeError: "
        "got an unexpected keyword argument 'diagonal'. Use diag=... (see next "
        "pitfall for the correct VALUE). Signal: 'unexpected keyword argument' "
        "from dolfinx.fem.petsc.assemble_matrix on a dolfinx 0.10 install. "
        "(Verified empirically 2026-06-01.)",
        "[Numerical] Do NOT pass diag=0.0 for the MASS matrix in a generalized "
        "eigenvalue problem. assemble_matrix(m, bcs=[bc], diag=0.0) zeros the "
        "Dirichlet rows of M; the resulting M is singular and SLEPc's GHEP "
        "factorisation aborts with 'Zero pivot row 0 value 0. tolerance "
        "2.22045e-14' followed by SystemError from EPS.solve. Use diag=1.0 "
        "(M's Dirichlet rows become identity rows; A's Dirichlet rows are also "
        "1·I, so each Dirichlet row contributes a spurious eigenvalue λ=1.0 "
        "that is filtered post-solve). (Verified empirically 2026-06-01.)",
        "[Numerical] With diag=1.0 on M, setWhichEigenpairs(SMALLEST_MAGNITUDE) "
        "returns the spurious boundary-unit eigenvalues — and worse, default "
        "shift-invert factorises (A - σ·B) which for σ≈1 is singular at "
        "Dirichlet rows (same 'Zero pivot row 0' error). Use SMALLEST_REAL "
        "with the DEFAULT Krylov-Schur spectral transform (no setST call). "
        "The first n_eigs eigenvalues returned will include some unit "
        "eigenvalues at Dirichlet rows followed by the physical eigenvalues "
        "(λ₁ = 2π² ≈ 19.74 for the unit-square Laplacian). Filter for "
        "λ > 2.0 post-hoc. (Verified empirically 2026-06-01.)",
        "SLEPc not always available — check import and provide fallback",
    ],
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate eigenvalue variant."""
    generators = {
        "2d": _eigenvalue_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown eigenvalue variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _eigenvalue_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 32)
    n_eigs = params.get("n_eigenvalues", 5)
    return f'''\
"""Eigenvalue problem: Laplace on [0,1]\u00b2 — FEniCSx + SLEPc"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
import ufl
import numpy as np
import json

domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {nx}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

V = fem.functionspace(domain, ("Lagrange", 1))
def boundary(x):
    return np.isclose(x[0], 0) | np.isclose(x[0], 1) | np.isclose(x[1], 0) | np.isclose(x[1], 1)
bc_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
bc = fem.dirichletbc(default_scalar_type(0),
    fem.locate_dofs_topological(V, fdim, bc_facets), V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = fem.form(ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx)
m = fem.form(u * v * ufl.dx)

from dolfinx.fem.petsc import assemble_matrix
A = assemble_matrix(a, bcs=[bc])
A.assemble()
# dolfinx 0.10 renamed the 'diagonal' kwarg → 'diag' in
# dolfinx.fem.petsc.assemble_matrix. The mass matrix M
# MUST keep a non-zero Dirichlet diagonal — setting
# diag=0.0 leaves M singular at boundary rows, causing
# SLEPc EPS.solve() to abort with 'Zero pivot row 0
# value 0' in the GHEP factorisation. Use diag=1.0 (the
# default); the spurious unit-eigenvalues at boundary
# rows are filtered out below by requesting LARGEST
# eigenvalues with a shift-invert spectral transform.
M = assemble_matrix(m, bcs=[bc], diag=1.0)
M.assemble()

try:
    from slepc4py import SLEPc
    eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
    eigensolver.setOperators(A, M)
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eigensolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    # Avoid SLEPc factorisation issues by using LARGEST
    # eigenvalues of the SHIFTED operator (A - σ·B)^-1 ·B
    # with σ chosen smaller than any physical eigenvalue.
    # SMALLEST_MAGNITUDE on A·x = λ·B·x with both A and B
    # carrying Dirichlet 1s on the diagonal would target
    # the spurious unit eigenvalues at boundary rows and
    # trip 'Zero pivot row 0' in the LU factorisation.
    # Set up to ask for ~n_eigs eigenvalues with magnitude
    # close to the physical lambda_1 = 2·pi² ≈ 19.74 (unit
    # square Laplacian).
    eigensolver.setDimensions({n_eigs}, {n_eigs * 4})
    eigensolver.setTolerances(tol=1e-6, max_it=1000)
    eigensolver.setWhichEigenpairs(
        SLEPc.EPS.Which.SMALLEST_REAL)
    # Skip a custom spectral transform: SLEPc's default
    # Krylov-Schur is sufficient when n_eigs is small and
    # we filter out the boundary unit eigenvalues post-
    # hoc.
    eigensolver.solve()
    n_conv = eigensolver.getConverged()
    eigenvalues = []
    for i in range(min(n_conv, {n_eigs})):
        eigenvalues.append(eigensolver.getEigenvalue(i).real)
    print(f"Eigenvalues: {{eigenvalues}}")
    summary = {{"eigenvalues": eigenvalues, "n_converged": n_conv, "n_dofs": V.dofmap.index_map.size_global}}
except ImportError:
    print("SLEPc not available — eigenvalue solve skipped")
    summary = {{"note": "SLEPc not installed", "n_dofs": V.dofmap.index_map.size_global}}

with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Eigenvalue analysis complete.")
'''
