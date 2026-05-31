"""Kratos heat conduction generators and knowledge."""


def _heat_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Heat conduction using Kratos ConvectionDiffusion."""
    nx = params.get("nx", 32)
    T_left = params.get("T_left", 100.0)
    T_right = params.get("T_right", 0.0)
    return f'''\
"""Heat conduction — Kratos (manual assembly)"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import json

nx, ny = {nx}, {nx}
nid = 1; node_map = {{}}; coords = {{}}
for j in range(ny+1):
    for i in range(nx+1):
        coords[nid] = (i/nx, j/ny)
        node_map[(i,j)] = nid; nid += 1
n_nodes = nid - 1

elements = []
for j in range(ny):
    for i in range(nx):
        n1,n2,n3,n4 = node_map[(i,j)],node_map[(i+1,j)],node_map[(i+1,j+1)],node_map[(i,j+1)]
        elements.append((n1,n2,n4)); elements.append((n2,n3,n4))

K = lil_matrix((n_nodes, n_nodes))
for tri in elements:
    ids = [t-1 for t in tri]
    x = np.array([coords[t][0] for t in tri])
    y = np.array([coords[t][1] for t in tri])
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    Ke = (1.0/(4.0*area)) * (np.outer(b,b) + np.outer(c,c))
    for a in range(3):
        for b_idx in range(3):
            K[ids[a], ids[b_idx]] += Ke[a, b_idx]
K = K.tocsr()

# Dirichlet BCs — set for your problem
left = {{node_map[(0,j)]-1 for j in range(ny+1)}}
right = {{node_map[(nx,j)]-1 for j in range(ny+1)}}
interior = sorted(set(range(n_nodes)) - left - right)
u = np.zeros(n_nodes)
for n in left: u[n] = {T_left}
for n in right: u[n] = {T_right}
rhs = -K.dot(u)
u[interior] = spsolve(K[np.ix_(interior, interior)], rhs[interior])

print(f"Temperature: max={{u.max():.6f}}")
import meshio
pts = np.array([[coords[i+1][0], coords[i+1][1], 0.0] for i in range(n_nodes)])
cells = np.array([[t-1 for t in tri] for tri in elements])
meshio.Mesh(pts, [("triangle", cells)], point_data={{"temperature": u}}).write("result.vtu")
summary = {{"max_value": float(u.max()), "n_nodes": n_nodes}}
with open("results_summary.json", "w") as _f: json.dump(summary, _f, indent=2)
'''


def _heat_transient_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Transient heat conduction with backward Euler time integration."""
    nx = params.get("nx", 32)
    dt = params.get("dt", 0.001)
    T_end = params.get("T_end", 0.1)
    T_left = params.get("T_left", 100.0)
    T_right = params.get("T_right", 0.0)
    kappa = params.get("conductivity", 1.0)
    rho_cp = params.get("rho_cp", 1.0)
    return f'''\
"""Transient heat conduction — backward Euler — Kratos (manual assembly)"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve, factorized
import json

nx, ny = {nx}, {nx}
nid = 1; node_map = {{}}; coords = {{}}
for j in range(ny+1):
    for i in range(nx+1):
        coords[nid] = (i/nx, j/ny)
        node_map[(i,j)] = nid; nid += 1
n_nodes = nid - 1

elements = []
for j in range(ny):
    for i in range(nx):
        n1,n2,n3,n4 = node_map[(i,j)],node_map[(i+1,j)],node_map[(i+1,j+1)],node_map[(i,j+1)]
        elements.append((n1,n2,n4)); elements.append((n2,n3,n4))

K = lil_matrix((n_nodes, n_nodes))
M = lil_matrix((n_nodes, n_nodes))

for tri in elements:
    ids = [t-1 for t in tri]
    x = np.array([coords[t][0] for t in tri])
    y = np.array([coords[t][1] for t in tri])
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    Ke = {kappa} * (1.0/(4.0*area)) * (np.outer(b,b) + np.outer(c,c))
    # Consistent mass matrix
    Me = {rho_cp} * area / 12.0 * (np.ones((3,3)) + np.eye(3))
    for a in range(3):
        for b_idx in range(3):
            K[ids[a], ids[b_idx]] += Ke[a, b_idx]
            M[ids[a], ids[b_idx]] += Me[a, b_idx]

K = K.tocsr(); M = M.tocsr()

# Dirichlet BCs — set for your problem
left = {{node_map[(0,j)]-1 for j in range(ny+1)}}
right = {{node_map[(nx,j)]-1 for j in range(ny+1)}}
dirichlet = left | right
interior = sorted(set(range(n_nodes)) - dirichlet)

# Backward Euler: (M + dt*K) * T_new = M * T_old
dt = {dt}
A = M + dt * K
solve_A = factorized(A[np.ix_(interior, interior)].tocsc())

# Initial condition: T=0 everywhere
T = np.zeros(n_nodes)
for n in left: T[n] = {T_left}
for n in right: T[n] = {T_right}

# Time stepping
t = 0.0
n_steps = int({T_end} / dt)
for step in range(n_steps):
    rhs = M @ T
    # Apply Dirichlet BCs to RHS
    rhs -= A @ T  # subtract known BC contributions
    rhs[list(dirichlet)] = 0.0
    T_new = T.copy()
    T_new[interior] = solve_A(rhs[interior] + (M @ T)[interior])
    # Re-apply BCs
    for n in left: T_new[n] = {T_left}
    for n in right: T_new[n] = {T_right}
    T = T_new
    t += dt

print(f"Transient heat: t={{t:.4f}}, max(T)={{T.max():.6f}}, min(T)={{T.min():.6f}}")

import meshio
pts = np.array([[coords[i+1][0], coords[i+1][1], 0.0] for i in range(n_nodes)])
cells_arr = np.array([[t_node-1 for t_node in tri] for tri in elements])
meshio.Mesh(pts, [("triangle", cells_arr)], point_data={{"temperature": T}}).write("result.vtu")

summary = {{
    "max_value": float(T.max()), "min_value": float(T.min()),
    "n_nodes": n_nodes, "n_steps": n_steps, "dt": dt, "time": t,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Transient heat solve complete.")
'''


KNOWLEDGE = {
    "heat": {
        "description": "Thermal analysis via ConvectionDiffusionApplication",
        "application": "ConvectionDiffusionApplication",
        "solver_types": ["stationary", "transient"],
        "pitfalls": [
                        '[API] Same field equation as Poisson but with TEMPERATURE as the unknown — TEMPERATURE must be added to ModelPart variables before any Node is created. '
                        "Signal: RuntimeError 'This container only can store the variables specified in its variables list. The variables list doesn't have this variable: TEMPERATURE' from kratos/containers/variables_list_data_value_container at the first GetSolutionStepValue / SetSolutionStepValue on the node. (Verified empirically 2026-06-01 — same wording as the VELOCITY case in fluid#0; prior catalog text 'not found in variables list of ModelPart' + 'from ConvectionDiffusion InitializeSolutionStep' is rearranged and points at the wrong call site.)",
                        '[Syntax] Non-homogeneous Dirichlet: use AssignScalarVariableProcess with constrained=True. Setting constrained=False applies the value but does NOT fix the DOF, so the solver overwrites it. '
                        'Signal: boundary temperatures drift away from the prescribed values during the solve; T_boundary - T_imposed is O(1) instead of O(eps).',
                        '[Syntax] Neumann (heat flux): use ApplyConstantScalarValueProcess on FACE_HEAT_FLUX (not TEMPERATURE). Targeting TEMPERATURE applies a Dirichlet pseudo-flux. '
                        'Signal: the steady-state interior temperature is wrong by a multiplicative factor; integrated flux on the boundary does not match the applied value.',
                    ],
    },
    "heat_transient": {
        "description": "Transient heat conduction via ConvectionDiffusionApplication",
        "application": "ConvectionDiffusionApplication",
        "solver_types": ["transient (theta scheme: 0=FE, 0.5=CN, 1=BE)"],
        "time_integration": {
            "backward_euler": "theta=1.0, unconditionally stable, first-order",
            "crank_nicolson": "theta=0.5, second-order but may oscillate",
            "forward_euler": "theta=0.0, conditionally stable (dt < h^2/(2*kappa))",
        },
        "pitfalls": [
                        '[Numerical] Backward Euler: factor (M + dt*K) once and reuse each step '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Crank-Nicolson: (M + 0.5*dt*K)*T_new = (M - 0.5*dt*K)*T_old '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Physics] For varying BCs in time: update Dirichlet values each step '
                        'Signal: post-processed quantity (max displacement, integrated flux, pressure) disagrees with analytic / textbook reference by 10-100%.',
                        '[Numerical] Consistent mass matrix gives better accuracy than lumped '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
}

GENERATORS = {
    "heat_2d": _heat_2d_kratos,
    "heat_transient_2d": _heat_transient_2d_kratos,
}
