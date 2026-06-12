"""Kratos cosimulation generators and knowledge."""


def _cosimulation_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable program. All parameter defaults are placeholders.

    CoSimulation framework coupling demo: thermal-structural weak coupling."""
    nx = params.get("nx", 20)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    alpha_T = params.get("thermal_expansion", 1e-5)
    T_left = params.get("T_left", 100.0)
    T_right = params.get("T_right", 0.0)
    n_coupling_steps = params.get("n_coupling_steps", 5)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""CoSimulation — thermal-structural weak coupling — Kratos (standalone)"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import json

# Domain and material — set for your problem
nx, ny = {nx}, {nx}
mu_val, lam_val = {mu}, {lam}
alpha_T = {alpha_T}
n_coupling = {n_coupling_steps}

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

left = {{node_map[(0,j)]-1 for j in range(ny+1)}}
right = {{node_map[(nx,j)]-1 for j in range(ny+1)}}

# --- Thermal solver ---
def solve_thermal():
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
    interior = sorted(set(range(n_nodes)) - left - right)
    T = np.zeros(n_nodes)
    for n in left: T[n] = {T_left}
    for n in right: T[n] = {T_right}
    rhs = -K.dot(T)
    T[interior] = spsolve(K[np.ix_(interior, interior)], rhs[interior])
    return T

# --- Structural solver with thermal loading ---
def solve_structural(T_field):
    ndof = 2 * n_nodes
    K = lil_matrix((ndof, ndof))
    F = np.zeros(ndof)
    for tri in elements:
        ids = [t-1 for t in tri]
        x = np.array([coords[t][0] for t in tri])
        y = np.array([coords[t][1] for t in tri])
        area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
        if area < 1e-14:
            continue
        b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]]) / (2*area)
        c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]]) / (2*area)
        B = np.zeros((3, 6))
        for a in range(3):
            B[0, 2*a] = b[a]; B[1, 2*a+1] = c[a]
            B[2, 2*a] = c[a]; B[2, 2*a+1] = b[a]
        D = np.array([[lam_val+2*mu_val, lam_val, 0],
                      [lam_val, lam_val+2*mu_val, 0],
                      [0, 0, mu_val]])
        Ke = area * B.T @ D @ B
        # Thermal strain: eps_T = alpha_T * (T - T_ref) * [1, 1, 0]
        T_avg = np.mean([T_field[t-1] for t in tri])
        eps_thermal = alpha_T * T_avg * np.array([1.0, 1.0, 0.0])
        sigma_thermal = D @ eps_thermal
        fe_thermal = -area * B.T @ sigma_thermal
        dofs = []
        for a in range(3):
            dofs.extend([2*ids[a], 2*ids[a]+1])
        for ii in range(6):
            F[dofs[ii]] += fe_thermal[ii]
            for jj in range(6):
                K[dofs[ii], dofs[jj]] += Ke[ii, jj]
    K = K.tocsr()
    # Fix left edge
    fixed = set()
    for j in range(ny+1):
        n = node_map[(0,j)] - 1
        fixed.add(2*n); fixed.add(2*n+1)
    interior = sorted(set(range(ndof)) - fixed)
    u = np.zeros(ndof)
    u[interior] = spsolve(K[np.ix_(interior, interior)], F[interior])
    return u

# --- Coupling loop (Gauss-Seidel weak coupling) ---
print(f"CoSimulation: {{n_coupling}} coupling iterations")
for coup_step in range(n_coupling):
    T_field = solve_thermal()
    u_field = solve_structural(T_field)
    ux = u_field[0::2]
    uy = u_field[1::2]
    max_disp = np.sqrt(ux**2 + uy**2).max()
    print(f"Coupling step {{coup_step+1}}/{{n_coupling}}: max(T)={{T_field.max():.4f}}, max|u|={{max_disp:.6e}}")

print(f"CoSimulation complete: max(T)={{T_field.max():.4f}}, max|u|={{max_disp:.6e}}")

# Write final output
import meshio
pts = np.array([[coords[i+1][0], coords[i+1][1], 0.0] for i in range(n_nodes)])
cells_arr = np.array([[t-1 for t in tri] for tri in elements])
meshio.Mesh(pts, [("triangle", cells_arr)], point_data={{
    "temperature": T_field,
    "displacement_x": ux,
    "displacement_y": uy,
    "displacement_magnitude": np.sqrt(ux**2 + uy**2),
}}).write("result.vtu")

summary = {{
    "max_temperature": float(T_field.max()),
    "max_displacement": float(max_disp),
    "n_coupling_steps": n_coupling,
    "n_nodes": n_nodes,
    "n_elements": len(elements),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("CoSimulation coupling complete.")
'''


KNOWLEDGE = {
    "cosimulation": {
        "description": "CoSimulation framework for multi-solver coupling (CoSimulationApplication)",
        "application": "CoSimulationApplication (pip install KratosCoSimulationApplication)",
        "coupling_schemes": {
            "weak": ["Gauss-Seidel (sequential)", "Jacobi (parallel)"],
            "strong": ["Gauss-Seidel with convergence check", "Jacobi with convergence check"],
        },
        # Real names (file stems in
        # KratosMultiphysics/CoSimulationApplication/
        # convergence_accelerators/). Verified empirically
        # 2026-06-01.
        "convergence_accelerators": [
            "constant_relaxation (omega=0.5-0.8 typical)",
            "aitken (adaptive relaxation, good starting point)",
            "mvqn (Multi-Vector Quasi-Newton, fastest convergence)",
            "block_mvqn (block-MVQN, partitioned variant)",
            # 'ibqn' alone is NOT a registered key — see pitfall #0.
            "block_ibqnls (block Interface Block Quasi-Newton Least Squares)",
            "iqnils (Interface Quasi-Newton Inverse Least Squares)",
            "anderson (Anderson acceleration)",
        ],
        # Real mapper type names from libKratosMappingCore.so
        # binary scan.
        "data_transfer": [
            "nearest_neighbor",
            "nearest_element",
            "barycentric",
            "coupling_geometry",
            "radial_basis_function",
            # 'kratos_mapping' is the *python wrapper module* name
            # under CoSim/data_transfer_operators/, not a mapper
            # *type*. 'empire_mapping' is NOT registered anywhere
            # in KratosMappingApplication 10.4.2.
        ],
        "solver_wrappers": {
            "internal": "KratosMultiphysics solvers (fluid, structural, thermal, etc.)",
            "external": "CoSimIO for coupling with external codes (C/C++/Python/Fortran API)",
        },
        "pitfalls": [
                        '[API] Catalog had two systematic naming '
                        'errors corrected 2026-06-01:\n'
                        '  (a) Convergence accelerator "ibqn" — '
                        'NOT a registered name. The real file '
                        'under KratosMultiphysics/'
                        'CoSimulationApplication/convergence_'
                        'accelerators/ is block_ibqnls.py (or '
                        'iqnils.py for the inverse-least-squares '
                        'variant). Other registered names: '
                        'aitken, anderson, constant_relaxation, '
                        'mvqn, block_mvqn.\n'
                        '  (b) Mapper type "empire_mapping" does '
                        'NOT exist in libKratosMappingCore.so '
                        '(empire substring 0 hits). Real mapper '
                        'types include nearest_neighbor, '
                        'nearest_element, barycentric, '
                        'coupling_geometry, radial_basis_function. '
                        'Also: "kratos_mapping" in the prior '
                        'catalog refers to the python wrapper '
                        'module name, not a mapper *type*. '
                        'Signal: convergence_accelerator type '
                        '"ibqn" in a CoSim parameters JSON raises '
                        'a Kratos factory ImportError finding '
                        '"ibqn.py" in convergence_accelerators/; '
                        'similarly mapping with "empire_mapping" '
                        'raises a MapperFactory unknown-mapper '
                        'error. (Verified empirically 2026-06-01 '
                        '— Tier-2 fixture cosimulation_accelerator_'
                        'mapper_names in scripts/tier2_fixtures/'
                        'kratos/. KratosCoSimulationApplication '
                        'was also missing from the .venv — '
                        'install via '
                        '"pip install KratosCoSimulationApplication" '
                        'before any CoSim catalog usage.)',
                        '[Numerical] Weak coupling: one pass per time step (fast but may be inaccurate for strong interactions) '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Strong coupling: iterate until interface convergence (required for added-mass instability) '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Aitken relaxation: good default, but MVQN converges faster for large interface problems '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Data mapping: non-matching meshes require interpolation (use RBF for smooth fields) '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] CoSimIO: standalone library for coupling Kratos with any external solver '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
}

GENERATORS = {
    "cosimulation_2d": _cosimulation_2d_kratos,
}
