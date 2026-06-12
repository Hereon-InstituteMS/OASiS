"""Kratos shape optimization generators and knowledge."""


def _shape_optimization_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable program. All parameter defaults are placeholders.

    Shape optimization using gradient-based steepest descent."""
    nx = params.get("nx", 20)
    ny = params.get("ny", 10)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    lx = params.get("lx", 2.0)
    ly = params.get("ly", 1.0)
    n_opt_steps = params.get("n_opt_steps", 30)
    step_size = params.get("step_size", 0.01)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Shape optimization — compliance minimization — Kratos (standalone)"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import json

nx, ny = {nx}, {ny}
lx, ly = {lx}, {ly}
mu_val, lam_val = {mu}, {lam}
n_opt_steps = {n_opt_steps}
step_size = {step_size}

nid = 1; node_map = {{}}; coords = {{}}
for j in range(ny+1):
    for i in range(nx+1):
        coords[nid] = np.array([i*lx/nx, j*ly/ny])
        node_map[(i,j)] = nid; nid += 1
n_nodes = nid - 1

elements = []
for j in range(ny):
    for i in range(nx):
        n1,n2,n3,n4 = node_map[(i,j)],node_map[(i+1,j)],node_map[(i+1,j+1)],node_map[(i,j+1)]
        elements.append((n1,n2,n4)); elements.append((n2,n3,n4))

# Identify design boundary (top edge nodes that can move vertically)
design_nodes = [node_map[(i, ny)] for i in range(1, nx)]

def assemble_and_solve(coords_cur):
    ndof = 2 * n_nodes
    K = lil_matrix((ndof, ndof))
    F = np.zeros(ndof)

    for tri in elements:
        ids = [t-1 for t in tri]
        x = np.array([coords_cur[t][0] for t in tri])
        y = np.array([coords_cur[t][1] for t in tri])
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
        dofs = []
        for a in range(3):
            dofs.extend([2*ids[a], 2*ids[a]+1])
        for ii in range(6):
            for jj in range(6):
                K[dofs[ii], dofs[jj]] += Ke[ii, jj]

    # Load — set for your problem (downward force on right edge)
    for j in range(ny+1):
        n = node_map[(nx, j)] - 1
        F[2*n+1] = -1.0 / (ny + 1)

    K = K.tocsr()
    # Fix left edge
    fixed = set()
    for j in range(ny+1):
        n = node_map[(0,j)] - 1
        fixed.add(2*n); fixed.add(2*n+1)
    interior = sorted(set(range(ndof)) - fixed)
    u = np.zeros(ndof)
    u[interior] = spsolve(K[np.ix_(interior, interior)], F[interior])
    compliance = F @ u
    return u, compliance

# Shape optimization loop
coords_opt = {{k: v.copy() for k, v in coords.items()}}
history = []

for opt_step in range(n_opt_steps):
    u, compliance = assemble_and_solve(coords_opt)
    history.append(compliance)

    # Compute shape gradient via finite differences on design boundary
    grad = np.zeros(len(design_nodes))
    eps_fd = ly * 1e-4
    for idx, dn in enumerate(design_nodes):
        coords_pert = {{k: v.copy() for k, v in coords_opt.items()}}
        coords_pert[dn] = coords_pert[dn] + np.array([0.0, eps_fd])
        _, c_pert = assemble_and_solve(coords_pert)
        grad[idx] = (c_pert - compliance) / eps_fd

    # Steepest descent update (move design nodes)
    grad_norm = np.linalg.norm(grad)
    if grad_norm > 1e-14:
        grad /= grad_norm
    for idx, dn in enumerate(design_nodes):
        coords_opt[dn] = coords_opt[dn] + np.array([0.0, -step_size * grad[idx]])

    if opt_step % 5 == 0 or opt_step == n_opt_steps - 1:
        print(f"Opt step {{opt_step}}: compliance = {{compliance:.6e}}, |grad| = {{grad_norm:.6e}}")

print(f"Shape optimization: compliance {{history[0]:.6e}} -> {{history[-1]:.6e}}")

# Write final shape as VTU
import meshio
pts = np.array([[coords_opt[i+1][0], coords_opt[i+1][1], 0.0] for i in range(n_nodes)])
cells_arr = np.array([[t-1 for t in tri] for tri in elements])
uy = u[1::2]
meshio.Mesh(pts, [("triangle", cells_arr)], point_data={{"displacement_y": uy}}).write("result.vtu")

summary = {{
    "initial_compliance": float(history[0]),
    "final_compliance": float(history[-1]),
    "improvement_pct": float((history[0] - history[-1]) / abs(history[0]) * 100),
    "n_opt_steps": n_opt_steps,
    "n_design_vars": len(design_nodes),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Shape optimization complete.")
'''


KNOWLEDGE = {
    "shape_optimization": {
        "description": "Shape optimization via Kratos ShapeOptimizationApplication",
        "application": "ShapeOptimizationApplication (pip install KratosShapeOptimizationApplication)",
        "algorithms": [
            "steepest_descent", "penalized_projection", "trust_region",
            "gradient_projection", "bead_optimization",
        ],
        "objective_types": ["compliance (strain energy)", "stress (max von Mises)",
                           "mass/volume", "eigenfrequency", "custom (user-defined)"],
        "filtering": {
            "vertex_morphing": "Henze-Hinterberger filter for mesh-independent shape update",
            "radius": "Filter radius controls smoothness of shape changes",
            "damping": "Near-boundary damping to prevent mesh distortion",
        },
        "pitfalls": [
                        '[Numerical] Shape gradients require adjoint solve or finite differences '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Mesh quality degrades with large shape changes: use mesh smoothing '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Filter radius should be > 2-3x element edge length '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Constrained optimization: use penalized projection or augmented Lagrangian '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] For manufacturing constraints: use bead optimization or geometric filtering '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
}

GENERATORS = {
    "shape_optimization_2d": _shape_optimization_2d_kratos,
}
