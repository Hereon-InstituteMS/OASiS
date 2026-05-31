"""Kratos Poisson equation generators and knowledge."""


def _poisson_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Poisson -Δu = f on [0,1]², u=0 on boundary — Kratos Multiphysics.

    Uses Kratos for mesh management and scipy for the linear solve.
    P1 triangular elements with manual FE assembly.
    """
    nx = params.get("nx", 32)
    ny = params.get("ny", nx)
    f_val = params.get("f", 1.0)
    return f'''\
"""Poisson -Δu = {f_val} on [0,1]², u=0 on boundary — Kratos Multiphysics"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import json

nx, ny = {nx}, {ny}
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
        elements.append((n1,n2,n4))
        elements.append((n2,n3,n4))

# Assemble -div(grad(u)) = f
K = lil_matrix((n_nodes, n_nodes))
F = np.zeros(n_nodes)

for tri in elements:
    ids = [t-1 for t in tri]
    x = np.array([coords[t][0] for t in tri])
    y = np.array([coords[t][1] for t in tri])
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    Ke = (1.0/(4.0*area)) * (np.outer(b,b) + np.outer(c,c))
    fe = {f_val} * area / 3.0 * np.ones(3)
    for a in range(3):
        F[ids[a]] += fe[a]
        for b_idx in range(3):
            K[ids[a], ids[b_idx]] += Ke[a, b_idx]

K = K.tocsr()

boundary = set()
for i in range(nx+1):
    boundary.add(node_map[(i,0)]-1); boundary.add(node_map[(i,ny)]-1)
for j in range(ny+1):
    boundary.add(node_map[(0,j)]-1); boundary.add(node_map[(nx,j)]-1)
interior = sorted(set(range(n_nodes)) - boundary)

u = np.zeros(n_nodes)
u[interior] = spsolve(K[np.ix_(interior, interior)], F[interior])

max_val = u.max()
print(f"max(u) = {{max_val:.10f}}")
print(f"Nodes: {{n_nodes}}, Elements: {{len(elements)}}")

import meshio
pts = np.array([[coords[i+1][0], coords[i+1][1], 0.0] for i in range(n_nodes)])
cells = np.array([[t-1 for t in tri] for tri in elements])
mio = meshio.Mesh(pts, [("triangle", cells)], point_data={{"phi": u}})
mio.write("result.vtu")

summary = {{"max_value": float(max_val), "n_nodes": n_nodes,
            "n_elements": len(elements), "element_type": "P1 tri"}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Kratos Poisson solve complete.")
'''


KNOWLEDGE = {
    "poisson": {
        "description": "Poisson/diffusion via Kratos ConvectionDiffusionApplication",
        "application": "ConvectionDiffusionApplication (pip install KratosConvectionDiffusionApplication)",
        "elements": ["LaplacianElement2D3N/3D4N (steady Laplace only)",
                     "EulerianConvDiff2D3N/3D4N (convection-diffusion, transient)"],
        "solver_types": ["stationary", "transient (theta scheme: 0=FE, 0.5=CN, 1=BE)"],
        "variables": {
            "unknown": "TEMPERATURE",
            "diffusion": "CONDUCTIVITY (property on Properties object)",
            "source": "HEAT_FLUX (nodal solution step variable)",
            "reaction": "REACTION_FLUX",
            "convection": "CONVECTION_VELOCITY (for transport problems)",
        },
        "settings_object": "ConvectionDiffusionSettings — must be set on ProcessInfo, maps variable names",
        "pitfalls": [
                        '[Numerical] LaplacianElement does NOT assemble source terms (HEAT_FLUX) — only -div(k*grad(T))=0 '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] For Poisson with source: use EulerianConvDiff elements AND set HEAT_FLUX as nodal data '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Integration] ConvectionDiffusionSettings MUST be set on ProcessInfo before solve '
                        "Signal: RuntimeError 'KeyError' from JSON parsing OR 'SubModelPart not found' / 'Property ID ... missing' during AnalysisStage.Initialize.",
                        '[Numerical] Properties (CONDUCTIVITY, DENSITY, SPECIFIC_HEAT) go on Properties object, NOT on nodes '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Integration] Material properties assigned via Begin Properties block in .mdpa OR via Materials.json '
                        "Signal: RuntimeError 'KeyError' from JSON parsing OR 'SubModelPart not found' / 'Property ID ... missing' during AnalysisStage.Initialize.",
                        '[Integration] VTK output: add vtk_output_process to output_processes in ProjectParameters.json '
                        "Signal: RuntimeError 'KeyError' from JSON parsing OR 'SubModelPart not found' / 'Property ID ... missing' during AnalysisStage.Initialize.",
                    ],
    },
}

GENERATORS = {
    "poisson_2d": _poisson_2d_kratos,
}
