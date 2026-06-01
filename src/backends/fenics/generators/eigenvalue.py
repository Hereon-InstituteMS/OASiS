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
        "assemble_matrix with diagonal=0.0 for the mass matrix (Dirichlet rows zeroed)",
        "Shift-and-invert: setWhichEigenpairs(SMALLEST_MAGNITUDE) with spectral transform",
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
# dolfinx 0.10 renamed the kwarg from 'diagonal' to 'diag'
# in dolfinx.fem.petsc.assemble_matrix. Passing 'diagonal'
# raises TypeError: unexpected keyword argument.
M = assemble_matrix(m, bcs=[bc], diag=0.0)
M.assemble()

try:
    from slepc4py import SLEPc
    eigensolver = SLEPc.EPS().create(MPI.COMM_WORLD)
    eigensolver.setOperators(A, M)
    eigensolver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eigensolver.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eigensolver.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_MAGNITUDE)
    eigensolver.setDimensions({n_eigs})
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
