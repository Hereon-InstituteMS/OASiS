"""Stokes flow generator for FEniCSx/dolfinx.

Variants: 2d
"""


KNOWLEDGE = {
    "description": "Stokes flow with Taylor-Hood P2/P1 or MINI element",
    "weak_form": "nu*(grad(u),grad(v))*dx + div(v)*p*dx + div(u)*q*dx = (f,v)*dx",
    "function_space": (
        "Mixed (basix.ufl 0.10+): "
        "P2 = basix.ufl.element('Lagrange', domain.basix_cell(), 2, shape=(domain.geometry.dim,)); "
        "P1 = basix.ufl.element('Lagrange', domain.basix_cell(), 1); "
        "TH = basix.ufl.mixed_element([P2, P1]); W = fem.functionspace(domain, TH)"
    ),
    "solver": {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    "pitfalls": [
        "System is INDEFINITE — use direct (MUMPS) or block preconditioner, NOT CG",
        "[API] Mixed-element construction in dolfinx 0.10+ uses "
        "basix.ufl, NOT ufl: P2 = basix.ufl.element('Lagrange', "
        "domain.basix_cell(), 2, shape=(domain.geometry.dim,)); "
        "P1 = basix.ufl.element('Lagrange', domain.basix_cell(), 1); "
        "TH = basix.ufl.mixed_element([P2, P1]). The pre-2024 "
        "form 'ufl.VectorElement / ufl.FiniteElement / "
        "ufl.MixedElement' was REMOVED — those names no longer "
        "exist as attributes of the ufl module (verified against "
        "ufl 2025.2.1 / dolfinx 0.10.0, 2026-06-01). Signal: "
        "running a stokes script using ufl.VectorElement raises "
        "AttributeError: module 'ufl' has no attribute "
        "'VectorElement'.",
        "TH = basix.ufl.mixed_element([P2, P1]), W = fem.functionspace(domain, TH)",
        "Non-homogeneous Dirichlet: collapse sub-space, create Function, interpolate, then dirichletbc",
        "Pressure determined up to constant — pin one DOF or ensure outflow BC",
        "MINI element: P1+bubble for velocity, P1 for pressure (simpler but less accurate)",
    ],
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate Stokes variant."""
    generators = {
        "2d": _stokes_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown Stokes variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 32)
    ny = params.get("ny", nx)
    return f'''\
"""Stokes flow — Taylor-Hood P2/P1 — FEniCSx/dolfinx"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import basix.ufl
import ufl
import numpy as np

domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
gdim = domain.geometry.dim
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# Taylor-Hood: P2 velocity + P1 pressure (basix.ufl, dolfinx 0.10+)
P2 = basix.ufl.element("Lagrange", domain.basix_cell(), 2, shape=(gdim,))
P1 = basix.ufl.element("Lagrange", domain.basix_cell(), 1)
TH = basix.ufl.mixed_element([P2, P1])
W = fem.functionspace(domain, TH)

# BCs: set velocity on boundaries (adjust for your problem)
def walls(x):
    return np.isclose(x[0], 0) | np.isclose(x[0], 1) | np.isclose(x[1], 0)
def lid(x):
    return np.isclose(x[1], 1)

W0 = W.sub(0)
wall_facets = mesh.locate_entities_boundary(domain, fdim, walls)
lid_facets = mesh.locate_entities_boundary(domain, fdim, lid)
V0, _ = W0.collapse()
noslip = fem.Function(V0)
noslip.x.array[:] = 0
bc_noslip = fem.dirichletbc(noslip, fem.locate_dofs_topological((W0, V0), fdim, wall_facets), W0)
lid_vel = fem.Function(V0)
lid_vel.interpolate(lambda x: (np.ones(x.shape[1]), np.zeros(x.shape[1])))
bc_lid = fem.dirichletbc(lid_vel, fem.locate_dofs_topological((W0, V0), fdim, lid_facets), W0)

(u, p) = ufl.TrialFunctions(W)
(v, q) = ufl.TestFunctions(W)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + ufl.div(u)*q*ufl.dx + ufl.div(v)*p*ufl.dx
L = ufl.inner(fem.Constant(domain, default_scalar_type((0, 0))), v) * ufl.dx

problem = LinearProblem(a, L, bcs=[bc_noslip, bc_lid],
    petsc_options_prefix="stokes",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}})
wh = problem.solve()

u_h = wh.sub(0).collapse()
p_h = wh.sub(1).collapse()
u_h.name = "velocity"
p_h.name = "pressure"

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "velocity.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_h)
with XDMFFile(domain.comm, "pressure.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(p_h)

print(f"Stokes: DOFs={{W.dofmap.index_map.size_global}}")
print("Stokes solve complete.")
'''
