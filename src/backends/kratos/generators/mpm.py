"""Kratos MPM (Material Point Method) generators and knowledge."""


def _mpm_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable program. All parameter defaults are placeholders.

    Material Point Method for large-deformation solid mechanics."""
    n_cells_x = params.get("n_cells_x", 20)
    n_cells_y = params.get("n_cells_y", 20)
    ppc = params.get("particles_per_cell", 4)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    density = params.get("density", 1000.0)
    gravity = params.get("gravity", -9.81)
    dt = params.get("dt", 1e-4)
    T_end = params.get("T_end", 0.5)
    domain_x = params.get("domain_x", 1.0)
    domain_y = params.get("domain_y", 1.0)
    body_x0 = params.get("body_x0", 0.3)
    body_x1 = params.get("body_x1", 0.7)
    body_y0 = params.get("body_y0", 0.5)
    body_y1 = params.get("body_y1", 0.9)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Material Point Method — large-deformation solid — Kratos (standalone)"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import json

# Grid parameters — set for your problem
n_cells_x, n_cells_y = {n_cells_x}, {n_cells_y}
domain_x, domain_y = {domain_x}, {domain_y}
dx = domain_x / n_cells_x
dy = domain_y / n_cells_y
n_nodes_x = n_cells_x + 1
n_nodes_y = n_cells_y + 1
n_nodes = n_nodes_x * n_nodes_y

# Material parameters — set for your problem
mu_val, lam_val = {mu}, {lam}
density = {density}
gravity = np.array([0.0, {gravity}])
dt = {dt}
T_end = {T_end}

def node_id(i, j):
    return j * n_nodes_x + i

def grid_coords(nid):
    j, i = divmod(nid, n_nodes_x)
    return np.array([i * dx, j * dy])

# Generate material points inside body region
ppc = {ppc}
particles_x = []
particles_v = []
particles_vol = []
particles_mass = []
particles_stress = []
particles_F = []

cell_vol = dx * dy / ppc
for cy in range(n_cells_y):
    for cx in range(n_cells_x):
        cell_x0 = cx * dx
        cell_y0 = cy * dy
        cell_cx = cell_x0 + dx / 2
        cell_cy = cell_y0 + dy / 2
        # Check if cell center is inside initial body
        if {body_x0} <= cell_cx <= {body_x1} and {body_y0} <= cell_cy <= {body_y1}:
            # Place particles in a 2x2 grid per cell
            sp = int(np.sqrt(ppc))
            for py in range(sp):
                for px in range(sp):
                    x_p = cell_x0 + (px + 0.5) * dx / sp
                    y_p = cell_y0 + (py + 0.5) * dy / sp
                    particles_x.append(np.array([x_p, y_p]))
                    particles_v.append(np.zeros(2))
                    particles_vol.append(cell_vol)
                    particles_mass.append(density * cell_vol)
                    particles_stress.append(np.zeros((2, 2)))
                    particles_F.append(np.eye(2))

n_particles = len(particles_x)
print(f"MPM: {{n_particles}} particles on {{n_cells_x}}x{{n_cells_y}} grid")

particles_x = np.array(particles_x)
particles_v = np.array(particles_v)
particles_vol = np.array(particles_vol)
particles_mass = np.array(particles_mass)

# Bilinear shape functions
def shape_functions(x_p, cell_i, cell_j):
    x0 = cell_i * dx
    y0 = cell_j * dy
    xi = (x_p[0] - x0) / dx
    eta = (x_p[1] - y0) / dy
    N = np.array([(1 - xi) * (1 - eta), xi * (1 - eta),
                  xi * eta, (1 - xi) * eta])
    dNdx = np.array([[-(1 - eta) / dx, -(1 - xi) / dy],
                      [(1 - eta) / dx, -xi / dy],
                      [eta / dx, xi / dy],
                      [-eta / dx, (1 - xi) / dy]])
    nodes = [node_id(cell_i, cell_j), node_id(cell_i + 1, cell_j),
             node_id(cell_i + 1, cell_j + 1), node_id(cell_i, cell_j + 1)]
    return N, dNdx, nodes

# Time integration — USL (Update Stress Last)
n_steps = int(T_end / dt)
output_interval = max(1, n_steps // 20)
max_disp = 0.0

for step in range(n_steps):
    ndof = 2 * n_nodes
    grid_mass = np.zeros(n_nodes)
    grid_momentum = np.zeros(ndof)
    grid_force = np.zeros(ndof)

    # Particle to grid transfer
    for p in range(n_particles):
        ci = min(int(particles_x[p, 0] / dx), n_cells_x - 1)
        cj = min(int(particles_x[p, 1] / dy), n_cells_y - 1)
        ci = max(0, ci)
        cj = max(0, cj)
        N, dNdx, nodes = shape_functions(particles_x[p], ci, cj)

        for a in range(4):
            nid = nodes[a]
            grid_mass[nid] += N[a] * particles_mass[p]
            grid_momentum[2 * nid] += N[a] * particles_mass[p] * particles_v[p, 0]
            grid_momentum[2 * nid + 1] += N[a] * particles_mass[p] * particles_v[p, 1]
            # Internal force: -sigma * grad(N) * vol
            sigma = particles_stress[p]
            grid_force[2 * nid] -= (sigma[0, 0] * dNdx[a, 0] + sigma[0, 1] * dNdx[a, 1]) * particles_vol[p]
            grid_force[2 * nid + 1] -= (sigma[1, 0] * dNdx[a, 0] + sigma[1, 1] * dNdx[a, 1]) * particles_vol[p]
            # Body force (gravity)
            grid_force[2 * nid + 1] += N[a] * particles_mass[p] * gravity[1]

    # Grid update with boundary conditions
    for nid in range(n_nodes):
        if grid_mass[nid] > 1e-14:
            # Update momentum
            grid_momentum[2 * nid] += dt * grid_force[2 * nid]
            grid_momentum[2 * nid + 1] += dt * grid_force[2 * nid + 1]
        # Floor BC (y=0): zero y-velocity
        j_idx = nid // n_nodes_x
        if j_idx == 0 and grid_mass[nid] > 1e-14:
            grid_momentum[2 * nid + 1] = 0.0

    # Grid to particle transfer and stress update
    for p in range(n_particles):
        ci = min(int(particles_x[p, 0] / dx), n_cells_x - 1)
        cj = min(int(particles_x[p, 1] / dy), n_cells_y - 1)
        ci = max(0, ci)
        cj = max(0, cj)
        N, dNdx, nodes = shape_functions(particles_x[p], ci, cj)

        v_new = np.zeros(2)
        grad_v = np.zeros((2, 2))
        for a in range(4):
            nid = nodes[a]
            if grid_mass[nid] > 1e-14:
                v_node = grid_momentum[2 * nid:2 * nid + 2] / grid_mass[nid]
                v_new += N[a] * v_node
                grad_v += np.outer(v_node, dNdx[a])

        particles_v[p] = v_new
        particles_x[p] += v_new * dt

        # Update deformation gradient
        F_old = particles_F[p]
        particles_F[p] = (np.eye(2) + dt * grad_v) @ F_old

        # Update volume
        J = np.linalg.det(particles_F[p])
        particles_vol[p] = abs(J) * particles_mass[p] / density

        # Neo-Hookean stress update (Cauchy)
        F = particles_F[p]
        J = np.linalg.det(F)
        if J > 0.01:
            b = F @ F.T
            particles_stress[p] = (mu_val / J) * (b - np.eye(2)) + (lam_val * np.log(J) / J) * np.eye(2)

    disp = np.linalg.norm(particles_x - np.array([
        [ci * dx + dx / 2 for ci in range(n_cells_x) for _ in range(ppc)]
        for _ in range(1)
    ]).flatten()[:n_particles] if False else 0.0)
    cur_max = np.max(np.abs(particles_v))
    max_disp = max(max_disp, cur_max)

    if step % output_interval == 0:
        print(f"Step {{step}}/{{n_steps}}, t={{step*dt:.6f}}, max|v|={{cur_max:.6e}}")

print(f"MPM complete: {{n_steps}} steps, {{n_particles}} particles")

# Write VTU output
import meshio
points = np.column_stack([particles_x, np.zeros(n_particles)])
cells = [("vertex", np.arange(n_particles).reshape(-1, 1))]
stress_xx = np.array([particles_stress[p][0, 0] for p in range(n_particles)])
stress_yy = np.array([particles_stress[p][1, 1] for p in range(n_particles)])
point_data = {{
    "velocity_x": particles_v[:, 0],
    "velocity_y": particles_v[:, 1],
    "stress_xx": stress_xx,
    "stress_yy": stress_yy,
    "volume": particles_vol,
}}
meshio.Mesh(points, cells, point_data=point_data).write("result.vtu")

summary = {{
    "n_particles": n_particles,
    "n_steps": n_steps,
    "dt": dt,
    "max_velocity": float(max_disp),
    "grid": f"{{n_cells_x}}x{{n_cells_y}}",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("MPM simulation complete.")
'''


KNOWLEDGE = {
    "mpm": {
        "description": "Material Point Method via Kratos MPMApplication",
        "application": "MPMApplication (pip install KratosMPMApplication)",
        "elements": {
            "2D": [
                "MPMUpdatedLagrangian2D3N",
                "MPMUpdatedLagrangian2D4N",
            ],
            "3D": [
                "MPMUpdatedLagrangian3D4N",
                "MPMUpdatedLagrangian3D8N",
            ],
            "axisymmetric": [
                "MPMUpdatedLagrangianAxisymmetry2D3N",
                "MPMUpdatedLagrangianAxisymmetry2D4N",
            ],
            "PQ_variant": [
                "MPMUpdatedLagrangianPQ",
            ],
            "UP_variant": [
                "MPMUpdatedLagrangianUP",
                "MPMUpdatedLagrangianUP2D3N",
            ],
        },
        "constitutive_laws": [
            "LinearElastic (small strain)", "NeoHookean (finite strain)",
            "HenckyMC (Mohr-Coulomb with Hencky strain)", "HenckyBorjaCamClay (critical state)",
            "Johnson-Cook (rate-dependent plasticity)",
        ],
        "solver_types": ["USL (Update Stress Last)", "USF (Update Stress First)", "MUSL (Modified USL)"],
        "pitfalls": [
                        '[API] Kratos MPM element names ALL start '
                        'with the literal prefix "MPM": '
                        'MPMUpdatedLagrangian2D4N, '
                        'MPMUpdatedLagrangian3D8N, '
                        'MPMUpdatedLagrangianAxisymmetry2D4N, '
                        'MPMUpdatedLagrangianPQ, '
                        'MPMUpdatedLagrangianUP, etc. The prior '
                        'catalog listed UpdatedLagrangianPQ2D / '
                        'UpdatedLagrangianAxisym (without the MPM '
                        'prefix) — none of those are registered. '
                        "Signal: model_part.CreateNewElement(\"UpdatedLagrangianPQ2D\", ...) "
                        "raises 'Error: The Element X is not "
                        "registered!' from "
                        "kratos/python/add_model_part_to_python.cpp; "
                        "prepending MPM makes the call succeed. "
                        "(Verified empirically 2026-06-01 — Tier-2 "
                        "fixture mpm_element_naming_mpm_prefix in "
                        "scripts/tier2_fixtures/kratos/.)",
                        '[Numerical] Background grid must cover entire domain of particle motion '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Cell crossing instability: use GIMP or CPDI shape functions for stability '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Particles per cell: 4-16 typical (2x2 or 4x4 per cell in 2D) '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Time step: dt < h/c where h=cell size, c=wave speed '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Zero-energy modes possible with linear elements: use stabilization '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Physics] For free surface: particles leaving grid are lost (expand grid or use remeshing) '
                        'Signal: post-processed quantity (max displacement, integrated flux, pressure) disagrees with analytic / textbook reference by 10-100%.',
                    ],
    },
}

GENERATORS = {
    "mpm_2d": _mpm_2d_kratos,
}
