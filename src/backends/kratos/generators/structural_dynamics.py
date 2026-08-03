"""Kratos structural dynamics generators and knowledge."""


def _structural_dynamics_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Dynamic structural analysis with Newmark time integration."""
    nx = params.get("nx", 20)
    ny = params.get("ny", 4)
    lx = params.get("lx", 10.0)
    ly = params.get("ly", 1.0)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    rho = params.get("density", 1.0)
    dt = params.get("dt", 0.01)
    T_end = params.get("T_end", 1.0)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Dynamic structural analysis — Newmark time integration — Kratos (manual assembly)"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import json

nx, ny, lx, ly = {nx}, {ny}, {lx}, {ly}
nid = 1; node_map = {{}}; coords = {{}}
for j in range(ny+1):
    for i in range(nx+1):
        coords[nid] = (i*lx/nx, j*ly/ny)
        node_map[(i,j)] = nid; nid += 1
n_nodes = nid - 1

elements = []
for j in range(ny):
    for i in range(nx):
        n1,n2,n3,n4 = node_map[(i,j)],node_map[(i+1,j)],node_map[(i+1,j+1)],node_map[(i,j+1)]
        elements.append((n1,n2,n4)); elements.append((n2,n3,n4))

ndof = 2 * n_nodes
K = lil_matrix((ndof, ndof))
M = lil_matrix((ndof, ndof))
mu_val, lam_val, rho_val = {mu}, {lam}, {rho}

for tri in elements:
    ids = [t-1 for t in tri]
    x = np.array([coords[t][0] for t in tri])
    y = np.array([coords[t][1] for t in tri])
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]]) / (2*area)
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]]) / (2*area)

    # Stiffness matrix
    B = np.zeros((3, 6))
    for a in range(3):
        B[0, 2*a] = b[a]; B[1, 2*a+1] = c[a]
        B[2, 2*a] = c[a]; B[2, 2*a+1] = b[a]
    D = np.array([[lam_val+2*mu_val, lam_val, 0], [lam_val, lam_val+2*mu_val, 0], [0, 0, mu_val]])
    Ke = area * B.T @ D @ B

    # Lumped mass matrix
    Me = rho_val * area / 3.0 * np.eye(6)

    dofs = []
    for a in range(3):
        dofs.extend([2*ids[a], 2*ids[a]+1])
    for i_loc in range(6):
        for j_loc in range(6):
            K[dofs[i_loc], dofs[j_loc]] += Ke[i_loc, j_loc]
            M[dofs[i_loc], dofs[j_loc]] += Me[i_loc, j_loc]

K = K.tocsr(); M = M.tocsr()

# Fix left edge
fixed = set()
for j in range(ny+1):
    n = node_map[(0,j)] - 1
    fixed.add(2*n); fixed.add(2*n+1)
interior = sorted(set(range(ndof)) - fixed)

# External force — set for your problem (impulse load on right edge)
F = np.zeros(ndof)
for j in range(ny+1):
    n = node_map[(nx,j)] - 1
    F[2*n+1] = -1.0 / (ny+1)

# Newmark parameters (average acceleration)
beta = 0.25; gamma = 0.5
dt = {dt}

# Effective stiffness: K_eff = K + 1/(beta*dt^2)*M
K_eff = K + (1.0/(beta*dt**2)) * M
from scipy.sparse.linalg import factorized
solve_eff = factorized(K_eff[np.ix_(interior, interior)].tocsc())

# Initial conditions
u = np.zeros(ndof)
v = np.zeros(ndof)
a_vec = np.zeros(ndof)

# Time stepping
t = 0.0
n_steps = int({T_end} / dt)
max_disp = 0.0

for step in range(n_steps):
    # Effective load
    F_eff = F + M @ (1.0/(beta*dt**2)*u + 1.0/(beta*dt)*v + (1.0/(2*beta)-1)*a_vec)

    u_new = np.zeros(ndof)
    u_new[interior] = solve_eff(F_eff[interior])

    # Update acceleration and velocity
    a_new = 1.0/(beta*dt**2)*(u_new - u) - 1.0/(beta*dt)*v - (1.0/(2*beta)-1)*a_vec
    v_new = v + dt*((1-gamma)*a_vec + gamma*a_new)

    u, v, a_vec = u_new, v_new, a_new
    t += dt
    cur_max = np.max(np.abs(u[1::2]))
    if cur_max > max_disp:
        max_disp = cur_max

uy = u[1::2]
print(f"Dynamic analysis: t={{t:.4f}}, max|u_y|={{max_disp:.6f}}, final max|u_y|={{np.max(np.abs(uy)):.6f}}")

import meshio
pts = np.array([[coords[i+1][0], coords[i+1][1], 0.0] for i in range(n_nodes)])
cells_arr = np.array([[t-1 for t in tri] for tri in elements])
meshio.Mesh(pts, [("triangle", cells_arr)], point_data={{
    "displacement_x": u[0::2], "displacement_y": u[1::2]
}}).write("result.vtu")

summary = {{
    "max_displacement_y": float(max_disp),
    "final_displacement_y": float(np.max(np.abs(uy))),
    "n_dofs": ndof, "n_steps": n_steps, "dt": dt, "T_end": t,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Dynamic structural analysis complete.")
'''


KNOWLEDGE = {
    "structural_dynamics": {
        "description": "Dynamic structural analysis via StructuralMechanicsApplication",
        "application": "StructuralMechanicsApplication",
        "solver_types": ["dynamic (Newmark, Bossak-alpha, GeneralizedAlpha)",
                        "explicit (central differences for wave propagation)"],
        "time_integration": {
            "Newmark": "beta=0.25, gamma=0.5 (average acceleration, unconditionally stable)",
            "Bossak": "alpha_m in [-1/3, 0] (numerical damping, unconditionally stable)",
            "GenAlpha": "Generalized-alpha (user-specified spectral radius at infinity)",
            "Explicit": "Central differences (conditionally stable, dt < h/c)",
        },
        "pitfalls": [
                        '[Numerical] Newmark average acceleration uses beta=0.25, gamma=0.5 (no numerical damping). Choosing gamma > 0.5 adds artificial damping; gamma < 0.5 is unconditionally unstable. '
                        'Signal: in a ResidualBasedNewmarkDisplacementScheme run with gamma < 0.5, the DISPLACEMENT amplitude in the VtkOutput grows exponentially across time steps regardless of dt.',
                        "[Syntax] Bossak adds mild numerical damping for high-frequency noise (alpha_m ≈ -0.1). In the Kratos JSON the parameter name is 'damp_factor_m' (NOT alpha_m). A wrong key is NOT silently ignored: Kratos Parameters validation is strict and the run aborts before the first time step. "
                        "Signal: RuntimeError 'Error: The item with name \"alpha_m\" is present in this Parameters but NOT in the default values' from Parameters::ValidateAndAssignDefaults; with 'damp_factor_m' the same case runs to completion. (Re-verified by execution 2026-08-03 on Kratos 10.4.0 — a full StructuralMechanicsAnalysis with solver_type 'Dynamic', scheme_type 'bossak' and the key renamed to alpha_m raises; the earlier catalog text 'Wrong key is silently ignored and the scheme runs without damping' was WRONG. This matters: an unknown key is a loud failure in Kratos, not a silent one, so do not go hunting for a physics explanation.)",
                        '[Numerical] Mass matrix: consistent (default) or lumped (faster for explicit). Lumped on linear tets/quads is OK; lumped on quadratic elements loses accuracy for higher-frequency modes. '
                        'Signal: natural-frequency study with lumped quad8 shows frequency error 2-5% for f_2..f_5 vs <0.5% with consistent mass.',
                        '[Numerical] Effective stiffness K_eff = K + 1/(beta*dt^2)*M — factor it ONCE for linear dynamics, reuse across steps. Re-factorising every step costs O(N^1.5) instead of O(N). '
                        'Signal: per-step wall_time of ResidualBasedBlockBuilderAndSolver scales as N^1.5 with mesh refinement instead of N — set reform_dofs_at_each_step: false in the linear_solver_settings to reuse the factorisation.',
                        '[Numerical] For nonlinear dynamics: tangent must be re-assembled at each Newton iteration (not just each time step). Caching the initial tangent gives modified-Newton with slow convergence. '
                        "Signal: ResidualBasedNewtonRaphsonStrategy iteration log shows the ResidualCriteria ratio decreasing by < 0.5 per iter (should be O(0.01) for full Newton); the strategy saturates at max_iteration without ResidualBasedBlockBuilderAndSolver reaching tolerance.",
                        '[Numerical] ELEMENT SELECTION: Linear hex8 (SmallDisplacementElement3D8N) shear-locks in bending-dominated problems — use quadratic hex20 or hex27. Same applies to linear quad4 in 2D — use quad8/quad9. '
                        'Signal: measured on this install (Kratos 10.4.0, plane-stress cantilever L=10, h=1, E=2e11, nu=0, tip shear 1000 N, LinearElasticPlaneStress2DLaw, sparse_lu): '
                        'SmallDisplacementElement2D4N on a 10x1 grid gives tip uy = -1.340e-5 = 66.6% of the Timoshenko value -2.012e-5 (33% too stiff, i.e. inside the claimed 20-40% band); '
                        'SmallDisplacementElement2D8N on the SAME 10x1 element grid gives -2.010e-5 = 99.9%. Refining quad4 recovers it only slowly: 20x2 -> 88.8%, 40x4 -> 96.9%, 80x8 -> 99.2%. '
                        '(Verified by execution 2026-08-03.)',
                        '[API] For POINT_LOAD application: use AssignVectorVariableProcess with constrained: [false, false, false]. '
                        'The directional process (KratosMultiphysics.assign_vector_by_direction_process, a CORE module — StructuralMechanicsApplication does NOT export a class of that name) tries to fix/free the DOF and therefore cannot be used for load variables. '
                        "Signal: RuntimeError 'Error: Trying to fix/free dof of variable POINT_LOAD_X but this dof does not exist in node #1!'; the same case with assign_vector_variable_process sets POINT_LOAD = [0, -100, 0] cleanly. "
                        'For loads applied on CONDITIONS (the usual case with PointLoadCondition2D1N) assign_vector_by_direction_to_condition_process IS the right process and works. '
                        '(Verified by execution 2026-08-03 on Kratos 10.4.0.)',
                        "[Syntax] problem_data section MUST include the 'echo_level' field (typically 0). Kratos accesses it during stage initialisation without a default. "
                        "Signal: RuntimeError 'Error: Getting a value that does not exist. entry string : echo_level' from the C++ Parameters::GetValue — NOT a Python KeyError, and not from RunSolutionLoop. (Re-verified by execution 2026-08-03: a full StructuralMechanicsAnalysis whose problem_data omits echo_level aborts with exactly that message. The earlier signal text in this entry said \"KeyError 'echo_level' from AnalysisStage.RunSolutionLoop\" and was stale; the linear_elasticity copy of this pitfall already carried the correct wording.)",
                    ],
    },
}

GENERATORS = {
    "structural_dynamics_2d": _structural_dynamics_2d_kratos,
}
