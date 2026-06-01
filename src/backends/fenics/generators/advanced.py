"""Advanced physics generators for FEniCSx/dolfinx.

Covers physics that require DG formulations, penalty methods, phase-field
models, transient time-stepping, phase separation, nonlinear Newton loops,
and curl-curl electromagnetic formulations.

Variants per physics:
  dg_methods            : 2d
  contact               : 2d
  multiphase            : 2d
  time_dependent_heat   : 2d
  cahn_hilliard         : 2d
  nonlinear_pde         : 2d
  magnetostatics        : 2d
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# dg_methods
# ---------------------------------------------------------------------------

_DG_KNOWLEDGE = {
    "description": (
        "Discontinuous Galerkin (DG) for advection-dominated diffusion. "
        "Upwind numerical flux handles convection; interior-penalty terms handle diffusion."
    ),
    "weak_form": (
        "eps*(grad(u),grad(v))*dx "
        "- avg(eps*grad(u))*jump(v)*dS + alpha/h*jump(u)*jump(v)*dS "
        "+ upwind_flux(b, u)*jump(v)*dS "
        "- b*u*grad(v)*dx + b*u_inflow*v*ds"
    ),
    "function_space": "DG order 1 (or higher) — fully discontinuous Lagrange",
    "solver": {"ksp_type": "preonly", "pc_type": "lu"},
    "pitfalls": [
        "DG advection flux: upwind — use ufl.conditional(ufl.dot(b,n)('+') > 0, u('+'), u('-'))",
        "Interior penalty diffusion: alpha >= O(p^2) for coercivity; alpha=4 safe for p=1",
        "FacetNormal n is outward; avg/jump operators need '+'/'-' sides",
        "Inflow BC imposed weakly via boundary integral, NOT Dirichlet",
        "For pure advection (eps=0): drop diffusion terms entirely",
        "DG mass matrix is block-diagonal — efficient for explicit time stepping",
        "[API] Modern UFL has no ufl.Abs symbol — use Python's "
        "builtin abs() (which UFL overloads for Expr operands) "
        "or ufl.algebra.Abs as the explicit fallback. Calling "
        "ufl.Abs(z) raises AttributeError: module 'ufl' has no "
        "attribute 'Abs'. The idiomatic pattern in DG upwind-flux "
        "construction is e.g. (b_n + abs(b_n))/2 for the outflow "
        "side. Signal: AttributeError with the literal text "
        "\"module 'ufl' has no attribute 'Abs'\" emitted at script "
        "import time before any assembly is attempted. (Verified "
        "empirically 2026-06-01 — Layer F catch.)",
    ],
    "materials": {
        "diffusion": {"range": [1e-8, 1.0], "unit": "m^2/s"},
        "convection_speed": {"range": [0.01, 1e4], "unit": "m/s"},
    },
}


def _dg_methods_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx DG script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 40)
    ny = params.get("ny", 40)
    eps = params.get("diffusion", 0.005)
    bx = params.get("bx", 1.0)
    by = params.get("by", 0.5)
    alpha = params.get("penalty", 4.0)
    return f'''\
"""Discontinuous Galerkin (DG) advection-diffusion — FEniCSx/dolfinx
Interior-penalty diffusion + upwind advection.
eps = {eps}, b = ({bx}, {by})
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
import ufl
import numpy as np
from petsc4py import PETSc

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
domain.topology.create_connectivity(fdim, fdim)

# DG function space — fully discontinuous
V = fem.functionspace(domain, ("DG", 1))

# Trial/test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Problem parameters
eps = {eps}
b = ufl.as_vector([{bx}, {by}])
n = ufl.FacetNormal(domain)
h = ufl.CellDiameter(domain)
h_avg = (h("+") + h("-")) / 2.0
alpha = {alpha}
f_rhs = fem.Constant(domain, default_scalar_type(1.0))
u_D = fem.Constant(domain, default_scalar_type(0.0))  # inflow value

# Upwind flux for advection
bn = ufl.dot(b, n)
bn_plus  = (bn("+") + abs(bn("+")) ) / 2.0   # outflow side
bn_minus = (bn("+") - abs(bn("+")) ) / 2.0   # inflow  side
adv_flux = bn_plus * u("+") + bn_minus * u("-")   # upwind

# Interior-penalty diffusion bilinear form
a_diff = (
    eps * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    - eps * ufl.inner(ufl.avg(ufl.grad(u)), ufl.jump(v, n)) * ufl.dS
    - eps * ufl.inner(ufl.jump(u, n), ufl.avg(ufl.grad(v))) * ufl.dS
    + alpha / h_avg * eps * ufl.inner(ufl.jump(u, n), ufl.jump(v, n)) * ufl.dS
)

# Upwind advection bilinear form
a_adv = (
    - ufl.inner(u * b, ufl.grad(v)) * ufl.dx
    + adv_flux * ufl.jump(v) * ufl.dS
    + ufl.dot(b, n) * u * v * ufl.ds  # outflow boundary (natural)
)

# Weakly imposed inflow boundary condition (negative bn face)
bn_ds = (ufl.dot(b, n) - abs(ufl.dot(b, n))) / 2.0  # only inflow faces
L_inflow = - bn_ds * u_D * v * ufl.ds

# Full system
a = a_diff + a_adv
L = f_rhs * v * ufl.dx + L_inflow

# Assemble and solve (no Dirichlet BCs in DG)
a_form = fem.form(a)
L_form = fem.form(L)

from dolfinx.fem.petsc import assemble_matrix, assemble_vector
A = assemble_matrix(a_form)
A.assemble()
b_vec = assemble_vector(L_form)
b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

uh = fem.Function(V, name="concentration")
solver.solve(b_vec, uh.x.petsc_vec)
uh.x.scatter_forward()

# Output
from dolfinx.io import XDMFFile

# Interpolate DG -> P1 for XDMF output
V_p1 = fem.functionspace(domain, ("Lagrange", 1))
u_out = fem.Function(V_p1, name="concentration")
u_out.interpolate(uh)

with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_out)

u_arr = uh.x.array
print(f"DG advection-diffusion solved")
print(f"u: min={{u_arr.min():.6e}}, max={{u_arr.max():.6e}}")
print(f"DOFs (DG): {{V.dofmap.index_map.size_global}}")
'''


# ---------------------------------------------------------------------------
# contact
# ---------------------------------------------------------------------------

_CONTACT_KNOWLEDGE = {
    "description": (
        "Contact / obstacle problem: find u >= phi such that -laplacian(u) = f "
        "where u >= phi (obstacle). Solved via penalty method or variational inequality."
    ),
    "weak_form": (
        "a(u,v) + gamma*(max(phi-u, 0), v)*dx = (f, v)*dx; "
        "penalty gamma -> infinity enforces u >= phi"
    ),
    "function_space": "Lagrange order 1 (scalar displacement or deflection)",
    "solver": "Newton iteration (NonlinearProblem / SNES)",
    "pitfalls": [
        "Penalty parameter gamma: too small -> constraint violation; too large -> ill-conditioning",
        "Typical gamma: 1e3 to 1e6 (problem-dependent); adaptive augmented Lagrangian is better",
        "max(phi-u, 0) is non-smooth -> Newton may converge slowly; use smooth approximation",
        "Smooth regularization: max(x,0) ~ (x + sqrt(x^2 + delta^2))/2 for small delta",
        "Signorini problem (1D contact): normal stress = 0 on contact zone at convergence",
        "For mechanical contact: need Lagrange multiplier or mortar methods for accuracy",
        "Active set strategy (semi-smooth Newton) is more robust than pure penalty",
    ],
    "materials": {
        "penalty": {"range": [1e2, 1e7], "unit": "N/m^2 or dimensionless"},
    },
}


def _contact_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx penalty-contact script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 32)
    ny = params.get("ny", 32)
    gamma = params.get("penalty", 1e4)
    obstacle_height = params.get("obstacle_height", -0.2)
    return f'''\
"""Contact / obstacle problem — penalty method — FEniCSx/dolfinx
-laplacian(u) = 1 on [0,1]^2, u >= phi (obstacle at height {obstacle_height})
u = 0 on boundary.
Penalty: (gamma * max(phi - u, 0)) added to residual.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

V = fem.functionspace(domain, ("Lagrange", 1))

# Homogeneous Dirichlet BC on all boundaries
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

# Obstacle (flat barrier)
phi = fem.Constant(domain, default_scalar_type({obstacle_height}))

# Penalty parameter
gamma = fem.Constant(domain, default_scalar_type({gamma}))

# Source term
f = fem.Constant(domain, default_scalar_type(1.0))

# Current solution (nonlinear iteration)
u = fem.Function(V, name="displacement")
v = ufl.TestFunction(V)

# Smooth penalty: max(phi - u, 0) approximated by smooth ramp
# penalty_term = gamma * max(phi - u, 0) * v
delta = fem.Constant(domain, default_scalar_type(1e-8))
arg = phi - u
# Smooth max: (arg + sqrt(arg^2 + delta)) / 2
smooth_max = (arg + ufl.sqrt(arg**2 + delta)) / 2.0

# Residual: standard Poisson + penalty
F = (ufl.dot(ufl.grad(u), ufl.grad(v)) - f * v + gamma * smooth_max * v) * ufl.dx

# Newton solve
problem = NonlinearProblem(F, u, bcs=[bc], petsc_options_prefix="contact",
    petsc_options={{
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 50,
        "snes_monitor": None,
    }})
problem.solve()
its = problem.solver.getIterationNumber()
reason = problem.solver.getConvergedReason()
print(f"Contact Newton: {{its}} iterations, reason={{reason}}")

# Check constraint satisfaction
u_arr = u.x.array
phi_val = {obstacle_height}
n_violated = np.sum(u_arr < phi_val - 1e-6)
print(f"Constraint violations (u < phi): {{n_violated}} / {{len(u_arr)}} DOFs")

# Output
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u)

print(f"Contact/obstacle: min(u)={{u_arr.min():.6e}}, max(u)={{u_arr.max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''


# ---------------------------------------------------------------------------
# multiphase
# ---------------------------------------------------------------------------

_MULTIPHASE_KNOWLEDGE = {
    "description": (
        "Two-phase flow via phase-field (Allen-Cahn / level-set). "
        "Phase field phi in [-1, 1] tracks the interface between two fluids."
    ),
    "weak_form": (
        "Allen-Cahn: (dphi/dt, v)*dx + eps*(grad(phi), grad(v))*dx + (1/eps)*(phi^3 - phi, v)*dx = 0. "
        "Level-set: (dphi/dt + b.grad(phi), v)*dx = 0 (advective)."
    ),
    "function_space": "Lagrange order 1 (phase field scalar)",
    "solver": "Newton for Allen-Cahn (nonlinear double-well); linear for advective LS",
    "pitfalls": [
        "Interface width epsilon: set to 2-4 mesh elements; too small -> oscillations",
        "Allen-Cahn: double-well potential W(phi) = (phi^2-1)^2 / (4*eps)",
        "Mobility parameter kappa scales interface diffusion; set to eps^2 typically",
        "Mass conservation: Allen-Cahn does NOT conserve volume; Cahn-Hilliard does",
        "For two-fluid NS: couple with Navier-Stokes via density/viscosity interpolation",
        "Level-set reinitialization required to keep |grad(phi)|=1 (signed distance property)",
        "Smeared physical properties: rho = rho1*H(phi) + rho2*(1-H(phi)) where H is Heaviside",
    ],
    "materials": {
        "epsilon": {"range": [0.01, 0.1], "unit": "m (interface thickness)"},
        "mobility": {"range": [1e-5, 1.0], "unit": "m^2/(N*s)"},
    },
}


def _multiphase_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx Allen-Cahn phase-field script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 64)
    ny = params.get("ny", 64)
    eps = params.get("epsilon", 0.04)
    dt = params.get("dt", 1e-4)
    n_steps = params.get("n_steps", 50)
    return f'''\
"""Two-phase Allen-Cahn phase-field — FEniCSx/dolfinx
dphi/dt = eps*laplacian(phi) - (phi^3 - phi)/eps
phi in [-1,+1]: phi=+1 (fluid 1), phi=-1 (fluid 2)
Interface width ~ {eps} (epsilon parameter)
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

V = fem.functionspace(domain, ("Lagrange", 1))

# No boundary conditions needed (no-flux = natural BC)

# Parameters
eps = fem.Constant(domain, default_scalar_type({eps}))
dt_c = fem.Constant(domain, default_scalar_type({dt}))

# Phase field: phi_new (current iterate), phi_old (previous time step)
phi_old = fem.Function(V, name="phi_old")
phi = fem.Function(V, name="phase_field")

# Initial condition: circular droplet in center
def init_phi(x):
    r = np.sqrt((x[0] - 0.5)**2 + (x[1] - 0.5)**2)
    return np.tanh((0.25 - r) / ({eps} * np.sqrt(2.0)))

phi_old.interpolate(init_phi)
phi.x.array[:] = phi_old.x.array[:]

# Test function
v = ufl.TestFunction(V)

# Allen-Cahn residual (backward Euler in time)
# (phi - phi_old)/dt * v + eps * grad(phi).grad(v) + (phi^3 - phi)/eps * v = 0
F = (
    (phi - phi_old) / dt_c * v * ufl.dx
    + eps * ufl.dot(ufl.grad(phi), ufl.grad(v)) * ufl.dx
    + (phi**3 - phi) / eps * v * ufl.dx
)

# Newton solver
problem = NonlinearProblem(F, phi, bcs=[], petsc_options_prefix="ac",
    petsc_options={{
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-8,
        "snes_max_it": 25,
    }})

# Time loop
n_steps = {n_steps}
from dolfinx.io import XDMFFile

with XDMFFile(domain.comm, "phase_field.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(phi, 0.0)

    for step in range(n_steps):
        phi_old.x.array[:] = phi.x.array[:]
        problem.solve()
        phi.x.scatter_forward()

        t = (step + 1) * {dt}
        phi_arr = phi.x.array
        # Volume fraction of phase +1
        volume_plus = np.sum(phi_arr > 0) / len(phi_arr)

        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            xdmf.write_function(phi, t)
            print(f"Step {{step+1}}/{{n_steps}}, t={{t:.5f}}: "
                  f"phi in [{{phi_arr.min():.4f}}, {{phi_arr.max():.4f}}], "
                  f"vol+={{volume_plus:.4f}}")

print(f"Allen-Cahn phase-field: {{n_steps}} steps complete")
print(f"Final phi: [{{phi.x.array.min():.6e}}, {{phi.x.array.max():.6e}}]")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''


# ---------------------------------------------------------------------------
# time_dependent_heat
# ---------------------------------------------------------------------------

_TIME_DEPENDENT_HEAT_KNOWLEDGE = {
    "description": (
        "Transient heat equation: rho*cp*dT/dt - div(k*grad(T)) = Q. "
        "Backward Euler time discretization. Handles non-uniform sources and convective BCs."
    ),
    "weak_form": (
        "rho*cp*(T^(n+1)-T^n)/dt * v * dx + k * grad(T^(n+1)) . grad(v) * dx "
        "+ h_conv*(T^(n+1) - T_amb) * v * ds(convective) = Q * v * dx"
    ),
    "function_space": "Lagrange order 1 (or 2 for better accuracy)",
    "solver": "Linear system per time step; matrix assembled once if coefficients constant",
    "pitfalls": [
        "Backward Euler: unconditionally stable, 1st order accurate in time",
        "Crank-Nicolson: 2nd order but can oscillate for step-function sources",
        "Convective BC: add h*(T - T_inf)*v*ds to bilinear form (Robin BC)",
        "Heat flux BC: add -q_n*v*ds to linear form (Neumann BC)",
        "Material properties rho, cp, k can be spatially varying Functions",
        "For PCM: enthalpy method with effective capacity for phase change",
        "CFL restriction irrelevant (implicit method); choose dt for accuracy",
    ],
    "materials": {
        "conductivity": {"range": [0.01, 500.0], "unit": "W/(m*K)"},
        "density": {"range": [1.0, 20000.0], "unit": "kg/m^3"},
        "specific_heat": {"range": [100.0, 5000.0], "unit": "J/(kg*K)"},
    },
}


def _time_dependent_heat_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx transient heat script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 40)
    ny = params.get("ny", 40)
    k = params.get("conductivity", 1.0)
    rho = params.get("density", 1.0)
    cp = params.get("specific_heat", 1.0)
    dt = params.get("dt", 0.005)
    n_steps = params.get("n_steps", 100)
    T_hot = params.get("T_hot", 1.0)
    T_init = params.get("T_init", 0.0)
    h_conv = params.get("h_conv", 0.0)
    T_amb = params.get("T_amb", 0.0)
    return f'''\
"""Transient heat equation — backward Euler — FEniCSx/dolfinx
rho*cp*dT/dt - div(k*grad(T)) = Q on [0,1]^2
Left boundary: T = T_hot = {T_hot}
Right boundary: T = 0
Top/bottom: insulated (natural BC)
Convective coefficient h_conv = {h_conv} (set >0 to activate Robin BC)
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
import ufl
import numpy as np
from petsc4py import PETSc

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

V = fem.functionspace(domain, ("Lagrange", 1))

# Boundary conditions
def left_wall(x):
    return np.isclose(x[0], 0.0)

def right_wall(x):
    return np.isclose(x[0], 1.0)

left_facets  = mesh.locate_entities_boundary(domain, fdim, left_wall)
right_facets = mesh.locate_entities_boundary(domain, fdim, right_wall)
dofs_left  = fem.locate_dofs_topological(V, fdim, left_facets)
dofs_right = fem.locate_dofs_topological(V, fdim, right_facets)
bc_hot  = fem.dirichletbc(default_scalar_type({T_hot}), dofs_left,  V)
bc_cold = fem.dirichletbc(default_scalar_type(0.0),     dofs_right, V)
bcs = [bc_hot, bc_cold]

# Thermal properties
k     = fem.Constant(domain, default_scalar_type({k}))
rho   = fem.Constant(domain, default_scalar_type({rho}))
cp    = fem.Constant(domain, default_scalar_type({cp}))
dt_c  = fem.Constant(domain, default_scalar_type({dt}))
Q_src = fem.Constant(domain, default_scalar_type(0.0))  # volumetric heat source
h_conv = fem.Constant(domain, default_scalar_type({h_conv}))
T_amb  = fem.Constant(domain, default_scalar_type({T_amb}))

# Solution functions
T_n = fem.Function(V, name="T_old")   # previous time step
T_n.x.array[:] = {T_init}            # uniform initial temperature
T_h = ufl.TrialFunction(V)
v   = ufl.TestFunction(V)

# Backward Euler bilinear and linear forms
# Convective BC on top/bottom: h_conv*(T - T_amb)*v*ds (Robin)
a = (
    rho * cp / dt_c * T_h * v * ufl.dx
    + k * ufl.dot(ufl.grad(T_h), ufl.grad(v)) * ufl.dx
    + h_conv * T_h * v * ufl.ds         # Robin: convective loss
)
L = (
    rho * cp / dt_c * T_n * v * ufl.dx
    + Q_src * v * ufl.dx
    + h_conv * T_amb * v * ufl.ds       # Robin: ambient contribution
)

# Compile forms
a_form = fem.form(a)
L_form = fem.form(L)

from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

A = assemble_matrix(a_form, bcs=bcs)
A.assemble()

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

T_new = fem.Function(V, name="temperature")

# Time loop
n_steps = {n_steps}
from dolfinx.io import XDMFFile

with XDMFFile(domain.comm, "temperature.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(T_n, 0.0)

    for step in range(n_steps):
        b_vec = assemble_vector(L_form)
        apply_lifting(b_vec, [a_form], bcs=[bcs])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b_vec, bcs)

        solver.solve(b_vec, T_new.x.petsc_vec)
        T_new.x.scatter_forward()
        T_n.x.array[:] = T_new.x.array[:]

        t = (step + 1) * {dt}
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            xdmf.write_function(T_new, t)
            T_arr = T_new.x.array
            print(f"Step {{step+1}}/{{n_steps}}, t={{t:.4f}}: "
                  f"T in [{{T_arr.min():.4f}}, {{T_arr.max():.4f}}]")

print(f"Transient heat: {{n_steps}} steps complete")
print(f"Final T: min={{T_new.x.array.min():.6e}}, max={{T_new.x.array.max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''


# ---------------------------------------------------------------------------
# cahn_hilliard
# ---------------------------------------------------------------------------

_CAHN_HILLIARD_KNOWLEDGE = {
    "description": (
        "Cahn-Hilliard equation for spinodal decomposition / phase separation. "
        "Fourth-order PDE split into two coupled second-order equations: "
        "dphi/dt = div(M*grad(mu));  mu = -eps^2*laplacian(phi) + W'(phi)."
    ),
    "weak_form": (
        "Mixed formulation: (dphi/dt, v)*dx + M*(grad(mu), grad(v))*dx = 0; "
        "(mu, q)*dx - eps^2*(grad(phi), grad(q))*dx - W'(phi)*q*dx = 0; "
        "W(phi) = (phi^2-1)^2/4 (double-well)"
    ),
    "function_space": "Mixed P1+P1 (phi and mu); can use P2+P2 for better accuracy",
    "solver": "Newton (NonlinearProblem/SNES) per time step — nonlinear W'(phi)=phi^3-phi",
    "pitfalls": [
        "eps controls interface width: set eps = 2-4 * h_mesh for resolved interface",
        "Mobility M: constant M or degenerate M(phi) = (1-phi^2)^+",
        "Double-well W(phi) = (phi^2-1)^2/4; W'(phi) = phi^3 - phi",
        "Mass conservation: integral of phi is conserved (no-flux BCs)",
        "Backward Euler: stable but first-order; convex splitting schemes are energy-stable",
        "Convex splitting: treat W_convex implicitly, W_concave explicitly for unconditional stability",
        "Initial condition: small random perturbation around phi=0 triggers spinodal decomposition",
        "Output: phi=+1 (phase A), phi=-1 (phase B), interface at phi=0",
    ],
    "materials": {
        "epsilon": {"range": [0.01, 0.1], "unit": "dimensionless (interface parameter)"},
        "mobility": {"range": [1e-5, 1.0], "unit": "m^2/(N*s) or dimensionless"},
    },
}


def _cahn_hilliard_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx Cahn-Hilliard script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 64)
    ny = params.get("ny", 64)
    eps = params.get("epsilon", 0.02)
    M = params.get("mobility", 1.0)
    dt = params.get("dt", 5e-5)
    n_steps = params.get("n_steps", 50)
    return f'''\
"""Cahn-Hilliard equation — spinodal decomposition — FEniCSx/dolfinx
Mixed formulation: (phi, mu) coupled system.
Double-well potential W(phi) = (phi^2-1)^2 / 4
Backward Euler time discretization.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np
from basix.ufl import element, mixed_element

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# Mixed function space: (phi, mu) both P1
P1 = element("Lagrange", domain.topology.cell_name(), 1)
ME = mixed_element([P1, P1])
W = fem.functionspace(domain, ME)

# Previous and current solution
w_old = fem.Function(W)
w     = fem.Function(W)

# Split: phi (phase field), mu (chemical potential)
phi_old, mu_old = ufl.split(w_old)
phi,     mu     = ufl.split(w)

# Test functions
v_phi, v_mu = ufl.TestFunctions(W)

# Parameters
eps_c = fem.Constant(domain, default_scalar_type({eps}))
M_c   = fem.Constant(domain, default_scalar_type({M}))
dt_c  = fem.Constant(domain, default_scalar_type({dt}))

# Nonlinear double-well: W'(phi) = phi^3 - phi
# Use theta-method weighting for stability (theta=1: fully implicit backward Euler)
theta = 0.5  # Crank-Nicolson in chemical potential (semi-implicit)
phi_mid = (1 - theta) * phi_old + theta * phi

# Cahn-Hilliard weak form (backward Euler in phi, semi-implicit in mu)
# Eq 1: (dphi/dt, v_phi) + M * (grad(mu), grad(v_phi)) = 0
# Eq 2: (mu, v_mu) - eps^2 * (grad(phi), grad(v_mu)) - W'(phi)*v_mu = 0
F = (
    (phi - phi_old) / dt_c * v_phi * ufl.dx
    + M_c * ufl.dot(ufl.grad(mu), ufl.grad(v_phi)) * ufl.dx
    + mu * v_mu * ufl.dx
    - eps_c**2 * ufl.dot(ufl.grad(phi), ufl.grad(v_mu)) * ufl.dx
    - (phi**3 - phi) * v_mu * ufl.dx
)

# No-flux BCs (natural) — no Dirichlet needed

# Initial condition: uniform mixture with small random noise
# phi ~ 0 + noise triggers spinodal decomposition
rng = np.random.default_rng(42)
W0, _ = W.sub(0).collapse()
W1, _ = W.sub(1).collapse()
phi_init = fem.Function(W0)
phi_init.x.array[:] = 0.0 + 0.05 * rng.standard_normal(len(phi_init.x.array))
mu_init = fem.Function(W1)
mu_init.x.array[:] = phi_init.x.array**3 - phi_init.x.array  # mu = W'(phi_0)
w.sub(0).interpolate(phi_init)
w.sub(1).interpolate(mu_init)
w.x.scatter_forward()
w_old.x.array[:] = w.x.array[:]

# Newton solver
problem = NonlinearProblem(F, w, bcs=[], petsc_options_prefix="ch",
    petsc_options={{
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_max_it": 25,
    }})

# Time loop with mass conservation check
n_steps = {n_steps}
from dolfinx.io import XDMFFile
from dolfinx.fem import assemble_scalar, form

phi_mass_form = form(phi * ufl.dx)
initial_mass = domain.comm.allreduce(assemble_scalar(phi_mass_form), op=MPI.SUM)

phi_out = fem.Function(W0, name="phase_field")
mu_out  = fem.Function(W1, name="chemical_potential")

with XDMFFile(domain.comm, "phase_field.xdmf", "w") as xdmf_phi, \
     XDMFFile(domain.comm, "chemical_potential.xdmf", "w") as xdmf_mu:
    xdmf_phi.write_mesh(domain)
    xdmf_mu.write_mesh(domain)

    for step in range(n_steps):
        w_old.x.array[:] = w.x.array[:]
        problem.solve()
        w.x.scatter_forward()

        t = (step + 1) * {dt}
        if step % max(1, n_steps // 10) == 0 or step == n_steps - 1:
            # Extract and write
            phi_out.interpolate(w.sub(0).collapse())
            mu_out.interpolate(w.sub(1).collapse())
            xdmf_phi.write_function(phi_out, t)
            xdmf_mu.write_function(mu_out, t)

            mass = domain.comm.allreduce(assemble_scalar(phi_mass_form), op=MPI.SUM)
            phi_arr = phi_out.x.array
            its = problem.solver.getIterationNumber()
            print(f"Step {{step+1}}/{{n_steps}}, t={{t:.6f}}: "
                  f"phi=[{{phi_arr.min():.4f}}, {{phi_arr.max():.4f}}], "
                  f"mass={{mass:.6e}} (ref={{initial_mass:.6e}}), "
                  f"Newton its={{its}}")

print(f"Cahn-Hilliard: {{n_steps}} steps complete")
print(f"DOFs: {{W.dofmap.index_map.size_global}}")
'''


# ---------------------------------------------------------------------------
# nonlinear_pde
# ---------------------------------------------------------------------------

_NONLINEAR_PDE_KNOWLEDGE = {
    "description": (
        "General nonlinear PDE solved with Newton's method via UFL automatic differentiation. "
        "Template: -div(D(u)*grad(u)) + R(u) = f with user-defined D and R."
    ),
    "weak_form": (
        "F(u; v) = D(u)*grad(u).grad(v)*dx + R(u)*v*dx - f*v*dx = 0; "
        "Jacobian J(u; du, v) = derivative(F, u, du) computed by UFL."
    ),
    "function_space": "Lagrange order 1 or 2 (scalar or vector)",
    "solver": "PETSc SNES (newtonls) with automatic UFL Jacobian",
    "pitfalls": [
        "Jacobian computed via ufl.derivative(F, u, du) — automatic, no hand differentiation needed",
        "Newton convergence: start from a good initial guess (e.g., linear solution or continuation)",
        "For strongly nonlinear D(u): line search in SNES helps (snes_linesearch_type=l2)",
        "Singularity: D(u)=0 can cause indefinite Jacobian; add regularization D_reg = max(D,eps)",
        "p-Laplacian: D(u) = |grad(u)|^(p-2) — singular at grad(u)=0 for p<2",
        "Semilinear: R(u)=u^3 (subcritical) is easy; R(u)=exp(u) can blow up",
        "Monitor convergence with snes_monitor and ksp_monitor PETSc options",
    ],
    "materials": {
        "nonlinearity_exponent": {"range": [1.0, 5.0], "unit": "dimensionless"},
    },
}


def _nonlinear_pde_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx nonlinear PDE script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 32)
    ny = params.get("ny", 32)
    q_exp = params.get("q_exponent", 2.0)
    return f'''\
"""General nonlinear PDE — FEniCSx/dolfinx
-div((1 + u^{q_exp}) * grad(u)) = f on [0,1]^2, u=0 on boundary
Jacobian via UFL automatic differentiation.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

V = fem.functionspace(domain, ("Lagrange", 1))

# Dirichlet BC: u = 0 on all boundaries
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

# Source term
f = fem.Constant(domain, default_scalar_type(1.0))

# Nonlinear diffusion coefficient: D(u) = 1 + u^q
# q_exp controls the nonlinearity strength
q = {q_exp}

# Current solution (nonlinear iterate)
u = fem.Function(V, name="u")
v = ufl.TestFunction(V)

# Nonlinear diffusivity
D_u = 1.0 + u**q

# Residual F(u; v)
F = D_u * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

# Jacobian: computed automatically by UFL via derivative
# du = ufl.TrialFunction(V) is inferred by NonlinearProblem

# Newton solver with PETSc SNES
problem = NonlinearProblem(F, u, bcs=[bc], petsc_options_prefix="nl",
    petsc_options={{
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "snes_monitor": None,
        "snes_linesearch_type": "l2",
    }})

# Solve (initial guess u=0 is fine for moderate nonlinearity)
problem.solve()
its  = problem.solver.getIterationNumber()
reason = problem.solver.getConvergedReason()
print(f"Newton: {{its}} iterations, converged reason = {{reason}}")

# Postprocess: evaluate diffusivity D(u) at solution
V_vis = fem.functionspace(domain, ("DG", 0))
D_expr = fem.Expression(D_u, V_vis.element.interpolation_points)
D_func = fem.Function(V_vis, name="diffusivity")
D_func.interpolate(D_expr)

# Output
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u)

u_arr = u.x.array
D_arr = D_func.x.array
print(f"Nonlinear PDE (q={{q:.1f}}) solved:")
print(f"u: min={{u_arr.min():.6e}}, max={{u_arr.max():.6e}}")
print(f"D(u): min={{D_arr.min():.4f}}, max={{D_arr.max():.4f}}")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''


# ---------------------------------------------------------------------------
# magnetostatics
# ---------------------------------------------------------------------------

_MAGNETOSTATICS_KNOWLEDGE = {
    "description": (
        "Magnetostatics: curl-curl formulation for magnetic vector potential A. "
        "div(B)=0 satisfied via B = curl(A). "
        "In 2D: A is a scalar (Az component), reduces to -div((1/mu)*grad(Az)) = Jz."
    ),
    "weak_form": (
        "2D scalar: (1/mu) * grad(Az) . grad(v) * dx = Jz * v * dx; "
        "3D vector: (1/mu) * curl(A) . curl(v) * dx = J . v * dx "
        "(requires Nedelec H(curl) elements in 3D)"
    ),
    "function_space": (
        "2D: scalar Lagrange P1 for Az component; "
        "3D: Nedelec first kind (H(curl) conforming) — essential for curl-curl!"
    ),
    "solver": {"ksp_type": "preonly", "pc_type": "lu"},
    "pitfalls": [
        "In 2D: scalar Az formulation — standard Lagrange works perfectly",
        "In 3D: MUST use H(curl) Nedelec elements (not Lagrange!) for correct curl",
        "Gauge fixing: add Coulomb gauge div(A)=0 or use tree-cotree gauging in 3D",
        "2D scalar formulation automatically satisfies gauge condition",
        "Flux density: B = curl(A); in 2D: Bx = dAz/dy, By = -dAz/dx",
        "For magnets: J=0 in iron, J=J_coil in coil regions (use markers)",
        "Permeability mu = mu0 * mu_r; air: mu_r=1; iron: mu_r=1000-10000",
        "Nonlinear iron (B-H curve): requires Newton iteration with mu(|B|)",
    ],
    "materials": {
        "mu_r": {"range": [1.0, 10000.0], "unit": "dimensionless (relative permeability)"},
        "J_source": {"range": [1e3, 1e8], "unit": "A/m^2 (current density)"},
    },
}


def _magnetostatics_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx 2D magnetostatics script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 40)
    ny = params.get("ny", 40)
    mu_r = params.get("mu_r", 1000.0)
    J_source = params.get("J_source", 1e6)
    coil_cx = params.get("coil_cx", 0.5)
    coil_cy = params.get("coil_cy", 0.5)
    coil_r = params.get("coil_r", 0.2)
    return f'''\
"""Magnetostatics — 2D scalar Az formulation — FEniCSx/dolfinx
-div((1/mu) * grad(Az)) = Jz on [0,1]^2
Az = 0 on boundary (tangential A = 0 -> no normal flux through boundary)
Current-carrying coil region: circle at ({coil_cx}, {coil_cy}) radius {coil_r}
Iron region (high mu_r={mu_r}) outside coil; air inside coil.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np

MU0 = 4.0 * np.pi * 1e-7  # H/m (permeability of free space)

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

V = fem.functionspace(domain, ("Lagrange", 1))

# Homogeneous Dirichlet BC: Az = 0 on outer boundary
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

# Spatially varying permeability mu(x):
# Coil region (circle): air mu_r = 1
# Surrounding domain: iron mu_r = {mu_r}
x = ufl.SpatialCoordinate(domain)
coil_cx, coil_cy, coil_r = {coil_cx}, {coil_cy}, {coil_r}
in_coil = ufl.conditional(
    (x[0] - coil_cx)**2 + (x[1] - coil_cy)**2 < coil_r**2,
    1.0,   # air: mu_r = 1
    {mu_r} # iron: mu_r = {mu_r}
)
mu_r_field = in_coil
mu = MU0 * mu_r_field

# Current density Jz: only inside coil
Jz = ufl.conditional(
    (x[0] - coil_cx)**2 + (x[1] - coil_cy)**2 < coil_r**2,
    {J_source},
    0.0
)

# Weak form: (1/mu) * grad(Az) . grad(v) = Jz * v
Az = ufl.TrialFunction(V)
v  = ufl.TestFunction(V)
a = (1.0 / mu) * ufl.dot(ufl.grad(Az), ufl.grad(v)) * ufl.dx
L = Jz * v * ufl.dx

# Solve
problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="mag",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu"}})
Az_h = problem.solve()
Az_h.name = "Az"

# Post-process: magnetic flux density B = curl(A) = (dAz/dy, -dAz/dx, 0)
# In 2D, computed on DG0 space
V_vec = fem.functionspace(domain, ("DG", 0, (2,)))
Bx_expr = fem.Expression(Az_h.dx(1),              V_vec.sub(0).collapse()[0].element.interpolation_points)
By_expr = fem.Expression(-Az_h.dx(0),             V_vec.sub(0).collapse()[0].element.interpolation_points)

# Scalar DG0 for each component
V_dg0 = fem.functionspace(domain, ("DG", 0))
Bx = fem.Function(V_dg0, name="Bx")
By = fem.Function(V_dg0, name="By")
Bx.interpolate(fem.Expression(Az_h.dx(1),  V_dg0.element.interpolation_points))
By.interpolate(fem.Expression(-Az_h.dx(0), V_dg0.element.interpolation_points))

# |B| = sqrt(Bx^2 + By^2)
B_mag = np.sqrt(Bx.x.array**2 + By.x.array**2)

# Output
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "Az.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(Az_h)

with XDMFFile(domain.comm, "Bx.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(Bx)

with XDMFFile(domain.comm, "By.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(By)

Az_arr = Az_h.x.array
print(f"Magnetostatics 2D solved:")
print(f"Az: min={{Az_arr.min():.6e}}, max={{Az_arr.max():.6e}} Wb/m")
print(f"|B|: min={{B_mag.min():.6e}}, max={{B_mag.max():.6e}} T")
print(f"mu_r iron = {mu_r}, J_source = {J_source:.2e} A/m^2")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

KNOWLEDGE: dict[str, dict] = {
    "dg_methods":          _DG_KNOWLEDGE,
    "contact":             _CONTACT_KNOWLEDGE,
    "multiphase":          _MULTIPHASE_KNOWLEDGE,
    "time_dependent_heat": _TIME_DEPENDENT_HEAT_KNOWLEDGE,
    "cahn_hilliard":       _CAHN_HILLIARD_KNOWLEDGE,
    "nonlinear_pde":       _NONLINEAR_PDE_KNOWLEDGE,
    "magnetostatics":      _MAGNETOSTATICS_KNOWLEDGE,
}

GENERATORS: dict[str, dict[str, callable]] = {
    "dg_methods":          {"2d": _dg_methods_2d},
    "contact":             {"2d": _contact_2d},
    "multiphase":          {"2d": _multiphase_2d},
    "time_dependent_heat": {"2d": _time_dependent_heat_2d},
    "cahn_hilliard":       {"2d": _cahn_hilliard_2d},
    "nonlinear_pde":       {"2d": _nonlinear_pde_2d},
    "magnetostatics":      {"2d": _magnetostatics_2d},
}


def generate(physics: str, variant: str, params: dict) -> str:
    """Dispatch to the appropriate advanced physics generator.

    Parameters
    ----------
    physics : str
        One of the physics names registered in GENERATORS.
    variant : str
        Variant name, e.g. ``"2d"``.
    params : dict
        Problem-specific parameters (mesh resolution, material constants, etc.).

    Returns
    -------
    str
        A runnable FEniCSx Python script.

    Raises
    ------
    ValueError
        If *physics* or *variant* is unknown.
    """
    physics_gens = GENERATORS.get(physics)
    if physics_gens is None:
        raise ValueError(
            f"Unknown advanced physics: {physics!r}. "
            f"Available: {sorted(GENERATORS)}"
        )
    gen = physics_gens.get(variant)
    if gen is None:
        raise ValueError(
            f"Unknown variant {variant!r} for physics {physics!r}. "
            f"Available: {sorted(physics_gens)}"
        )
    return gen(params)
