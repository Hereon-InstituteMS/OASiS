"""Helmholtz equation generator for FEniCSx/dolfinx.

Variants: 2d
"""


KNOWLEDGE = {
    "description": "Helmholtz: -Δu - k²u = f. Indefinite system. Use GMRES or direct solver — NOT CG.",
    "weak_form": "inner(grad(u), grad(v))*dx - k**2 * inner(u, v)*dx = inner(f, v)*dx",
    "function_space": (
        "Lagrange P2+ (need ~10 points per wavelength). For "
        "complex-valued problems use scalar_type=np.complex128 "
        "AND a PETSc build with --with-scalar-type=complex."
    ),
    "solver": {
        "real": "Direct (MUMPS) or GMRES + LU preconditioner",
        "complex": "Same — complex PETSc build required",
    },
    "pitfalls": [
        "[Numerical] System is INDEFINITE — CG diverges. Use GMRES or direct. "
        "Signal: SolverCG fails with 'breakdown' / 'NaN residual' after a "
        "few iterations on a Helmholtz problem.",
        "[Numerical] Resolution rule: ~10 DOFs per wavelength minimum. "
        "Pollution effect grows with k — high-k problems need 20+ DOFs/wavelength. "
        "Signal: solution amplitude shrinks vs analytic plane-wave by factor "
        "(1 - C*k*h^2) for under-resolved meshes.",
        "[Syntax] Complex-valued mode requires dolfinx.default_scalar_type "
        "to be np.complex128 — set via PETSC_DIR / petsc env. "
        "Signal: 'ScalarType is not complex' error at form-compilation time.",
    ],
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate Helmholtz variant."""
    generators = {
        "2d": _helmholtz_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(
            f"Unknown Helmholtz variant: {variant!r}. "
            f"Available: {list(generators)}")
    return gen(params)


def _helmholtz_2d(params: dict) -> str:
    """FORMAT TEMPLATE — Helmholtz in a unit square with
    homogeneous Dirichlet BC and a Gaussian source. Real-valued
    (no PML, no absorbing BC) — for a complex-valued absorbing-
    BC variant, switch scalar_type=np.complex128 and add an
    impedance term."""
    nx = params.get("nx", 64)
    k_val = params.get("k", 6.0)
    return f'''\
"""Helmholtz: -Δu - k²u = f — FEniCSx/dolfinx (real-valued, Dirichlet)"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import basix.ufl
import ufl
import numpy as np

domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {nx},
                                 mesh.CellType.triangle)
V = fem.functionspace(domain,
                      basix.ufl.element("Lagrange",
                                         domain.basix_cell(), 2))

k = fem.Constant(domain, default_scalar_type({k_val}))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(domain)
f = ufl.exp(-50.0 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
     - k * k * ufl.inner(u, v) * ufl.dx)
L = ufl.inner(f, v) * ufl.dx

# Homogeneous Dirichlet on all boundaries
def boundary(x):
    return (np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0)
            | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0))
dofs = fem.locate_dofs_geometrical(V, boundary)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

# Helmholtz is INDEFINITE — direct solver (MUMPS) or GMRES
problem = LinearProblem(
    a, L, bcs=[bc],
    petsc_options_prefix="helmholtz_",
    petsc_options={{"ksp_type": "preonly",
                    "pc_type": "lu",
                    "pc_factor_mat_solver_type": "mumps"}})
uh = problem.solve()
print(f"||u||_L2 = {{np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(uh, uh) * ufl.dx))))}}")
'''
