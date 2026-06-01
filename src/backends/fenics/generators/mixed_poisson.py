"""Mixed Poisson / Darcy flow generator for FEniCSx/dolfinx.

Variants: 2d
"""


KNOWLEDGE = {
    "description": "Mixed Poisson / Darcy flow using Raviart-Thomas + DG pressure",
    "weak_form": "(sigma, tau)*dx + (div(tau), p)*dx + (div(sigma), v)*dx = -(f, v)*dx",
    "function_space": "RT(k) for flux + DG(k-1) for pressure (inf-sup stable pair)",
    "solver": {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    "pitfalls": [
        (
            "[Numerical] System is INDEFINITE (saddle point): use "
            "direct solver or block preconditioner. Signal: "
            "PETSc CG/GMRES reports `DIVERGED_INDEFINITE_PC` or "
            "stagnates with residual ~1.0 (no decrease); LU "
            "succeeds where iterative fails — switch to a "
            "fieldsplit / Schur-complement preconditioner for "
            "scale. (Audit 2026-06-02.)"
        ),
        (
            "[Numerical] RT(k) + DG(k-1): inf-sup stable, locally "
            "conservative (exact div). Signal: when mixing "
            "non-inf-sup-stable pairs (e.g. RT(k) + DG(k)), "
            "the discrete LBB constant collapses with mesh "
            "refinement — pressure norm grows like O(h^-1) "
            "instead of converging, and the divergence error "
            "fails to reach machine precision. (Audit "
            "2026-06-02.)"
        ),
        "BDM(k) + DG(k-1): alternative H(div) pair with full polynomial",
        (
            "[API] Essential BC is on sigma.n (normal flux), NOT "
            "on pressure. Signal: applying a dolfinx DirichletBC "
            "on the pressure DOFs at an inflow surface (instead "
            "of on the flux) gives wildly wrong pressure profile "
            "and ZERO normal flux at that surface; mass balance "
            "is violated by an O(1) factor. (Audit 2026-06-02.)"
        ),
        (
            "[Numerical] Pressure determined up to a constant if "
            "only normal flux BCs. Signal: solver reports a "
            "near-zero pivot or singular system "
            "(`KSPSolve: DIVERGED_BREAKDOWN`); add a pressure "
            "pinning DOF or use a nullspace removal to restore "
            "uniqueness. (Audit 2026-06-02.)"
        ),
        "For heterogeneous permeability: weight the (sigma, tau) term by K^{-1}",
        (
            "[API] Use basix.ufl.element('RT', cell, k) for "
            "Raviart-Thomas in dolfinx. Signal: passing a "
            "legacy fenics-style string like 'RT' to "
            "FunctionSpace gives `AttributeError: module "
            "'dolfinx.fem' has no attribute 'FiniteElement'` or "
            "an obscure cpp-side `unknown element family` "
            "error from basix. (Audit 2026-06-02.)"
        ),
    ],
    "materials": {
        "permeability": {"range": [1e-15, 1.0], "unit": "m^2 (Darcy permeability)"},
    },
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate mixed Poisson variant."""
    generators = {
        "2d": _mixed_poisson_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown mixed_poisson variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _mixed_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable script. All parameter defaults are placeholders. The user/agent must set values appropriate to the specific problem being solved."""
    nx = params.get("nx", 32)
    ny = params.get("ny", 32)
    rt_order = params.get("rt_order", 1)
    return f'''\
"""Mixed Poisson: Raviart-Thomas + DG pressure — FEniCSx/dolfinx
sigma + grad(p) = 0 (Darcy flow / flux formulation)
div(sigma) = f
sigma in H(div), p in L2.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
import ufl
import numpy as np
from basix.ufl import element, mixed_element
from petsc4py import PETSc

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# Mixed function space: Raviart-Thomas for flux + DG for pressure
RT = element("RT", domain.topology.cell_name(), {rt_order})
DG = element("DG", domain.topology.cell_name(), {rt_order - 1})
ME = mixed_element([RT, DG])
W = fem.functionspace(domain, ME)

# Trial and test functions
(sigma, u) = ufl.TrialFunctions(W)
(tau, v) = ufl.TestFunctions(W)

# Source term
f = fem.Constant(domain, default_scalar_type(1.0))

# Bilinear form: (sigma, tau) + (div(tau), u) + (div(sigma), v) = -(f, v)
a = (ufl.inner(sigma, tau) + ufl.div(tau) * u + ufl.div(sigma) * v) * ufl.dx
L = -f * v * ufl.dx

# Essential BC on sigma.n = 0 on boundary (natural for pressure)
# For RT elements: normal component DOFs on boundary facets
boundary_facets = mesh.exterior_facet_indices(domain.topology)
W0 = W.sub(0)
V0, _ = W0.collapse()
sigma_bc = fem.Function(V0)
sigma_bc.x.array[:] = 0.0
bc_dofs = fem.locate_dofs_topological((W0, V0), fdim, boundary_facets)
bc = fem.dirichletbc(sigma_bc, bc_dofs, W0)

# Assemble and solve
a_form = fem.form(a)
L_form = fem.form(L)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

A = assemble_matrix(a_form, bcs=[bc])
A.assemble()
b = assemble_vector(L_form)
apply_lifting(b, [a_form], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])

# Solve — indefinite system, use direct solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.getPC().setFactorSolverType("mumps")

wh = fem.Function(W)
solver.solve(b, wh.x.petsc_vec)

# Extract flux and pressure
sigma_h = wh.sub(0).collapse()
u_h = wh.sub(1).collapse()
sigma_h.name = "flux"
u_h.name = "pressure"

# Output pressure (DG field)
P_out = fem.functionspace(domain, ("Lagrange", 1))
p_out = fem.Function(P_out, name="pressure")
p_out.interpolate(u_h)

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "pressure.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(p_out)

p_arr = p_out.x.array
print(f"Mixed Poisson solved: min(p)={{p_arr.min():.6e}}, max(p)={{p_arr.max():.6e}}")
print(f"DOFs: {{W.dofmap.index_map.size_global}}")
'''
