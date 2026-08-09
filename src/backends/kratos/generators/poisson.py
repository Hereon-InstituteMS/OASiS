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
"""Poisson -Δu = {f_val} on [0,1]², u=0 on boundary — Kratos (manual assembly)

This script does NOT use KratosMultiphysics: it assembles and solves the system
directly with numpy/scipy. The docstring previously read "Kratos Multiphysics"
with no qualifier, which told a reader the opposite of the truth — every sibling
template in this backend says "(manual assembly)" or "(standalone)", and this one
was the only one that did not. A weak model reads the first line and stops.

Run it with `run_simulation` rather than `run_with_generator`, per the server
instructions: it is a standalone Python script, not a Kratos input deck."""
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
            "[API] ConvectionDiffusionApplication element names \u2014 LaplacianElement2D3N / LaplacianElement3D4N / EulerianConvDiff2D3N / EulerianConvDiff3D4N \u2014 are C++-registered Kratos elements accessible ONLY through the string-typed factory call: model_part.CreateNewElement(\"LaplacianElement2D3N\", id, node_id_list, properties). They are NOT exposed as Python attributes on KratosMultiphysics.ConvectionDiffusionApplication, so CDA.LaplacianElement2D3N raises AttributeError. Wrong-named strings (e.g. \"ConvDiff2D3N\" without the \"Eulerian\" prefix) are rejected at CreateNewElement with \"The Element 'X' is not registered!\". Also: in a fresh install of KratosMultiphysics 10.4.2 the CDA sub-application is NOT included by default \u2014 pip install KratosConvectionDiffusionApplication is the separate package needed before this element family is usable. Signal: hasattr(CDA, \"LaplacianElement2D3N\") is False; mp.CreateNewElement(\"LaplacianElement2D3N\", ...) returns a Kratos Element; mp.CreateNewElement(\"ConvDiff2D3N\", ...) raises with \"is not registered\". (Verified empirically 2026-06-01 \u2014 Tier-2 fixture poisson_cda_element_string_factory in scripts/tier2_fixtures/kratos/.)",
            "[Numerical] LaplacianElement2D3N / LaplacianElement3D4N DO assemble the HEAT_FLUX volumetric source term when the ConvectionDiffusionSettings on ProcessInfo declares HEAT_FLUX as the VolumeSourceVariable. On a P1 unit-right triangle, LaplacianElement2D3N.CalculateRightHandSide with HEAT_FLUX=10 set on all 3 nodes returns the consistent load RHS_i = 10 * area / 3 = 1.66667 on every node \u2014 classic linear-shape-function integration of a constant source. (Catalog falsification verified empirically 2026-06-01 \u2014 Tier-2 fixture poisson_laplacian_element_assembles_heat_flux. The prior catalog claim that this element \"does NOT assemble HEAT_FLUX\" was WRONG and has been corrected.) Signal: with ConvectionDiffusionSettings.SetVolumeSourceVariable(HEAT_FLUX) and HEAT_FLUX set on nodes, RHS node values equal source * triangle_area / 3 for LaplacianElement2D3N; with HEAT_FLUX=0 the RHS is exactly zero.",
            "[Numerical] LaplacianElement2D3N and EulerianConvDiff2D3N are NOT interchangeable, not even with zero velocity in a stationary solve \u2014 swapping them silently destroys the solution. Both assemble the same HEAT_FLUX volume source, but only LaplacianElement assembles a DIFFUSION stiffness. Measured element-level on a unit right triangle (nodal and Properties CONDUCTIVITY = 1, zero velocity, DELTA_TIME = 1): LaplacianElement2D3N LHS = [[1,-0.5,-0.5],[-0.5,0.5,0],[-0.5,0,0.5]] (the P1 stiffness matrix), EulerianConvDiff2D3N LHS = [[0.0833,0.0417,0.0417],[...]] = the consistent MASS matrix with NO conductivity term; both RHS = 1.6667. Consequence at system level, 12x12 P1 unit square, f = 2*pi^2*sin(pi x)sin(pi y), u = 0 on the boundary, ResidualBasedLinearStrategy: LaplacianElement2D3N gives max T = 0.9755 (exact 1.0), while EulerianConvDiff2D3N returns a field with NO diffusion in it. WHAT that wrong field looks like depends on DELTA_TIME, so do NOT use \"all zeros\" as the test: with DELTA_TIME = 1.0 the same case gives max T = 19.7392 = 2*pi^2, i.e. T reproduces the SOURCE field (LHS = M/dt, so T = M^-1 M f = f); with DELTA_TIME = 0 (easy to hit by setting ProcessInfo[TIME] and then calling CloneTimeStep with the same value) the system degenerates, Kratos prints \"[WARNING] ResidualBasedBlockBuilderAndSolver: ATTENTION! setting the RHS to zero!\" and T comes out 0.0 everywhere. Use EulerianConvDiff only inside a TRANSIENT convection-diffusion solver that supplies the time-integration context it expects. Signal: check the ELEMENT, not the field \u2014 CalculateLocalSystem on one triangle returns the consistent mass matrix (0.0833/0.0417 for a unit right triangle) instead of the P1 stiffness (1/-0.5), and the LHS scales with 1/dt instead of being dt-independent. (Verified by execution 2026-08-03 on Kratos 10.4.0; system-level wording re-derived 2026-08-03 (re-audit) \u2014 the prior catalog claim that the two elements \"differ by less than 1e-12 relative norm\" with zero convection was WRONG, and the 2026-08-03 replacement text \"TEMPERATURE == 0.0 at EVERY node \u2014 no exception, no warning\" was ALSO wrong: it reported a DELTA_TIME = 0 artifact of its own fixture, and a warning IS printed.)",
            "[Integration] ConvectionDiffusionSettings MUST be set on ProcessInfo before solve Signal: the omission is SILENT, not an exception. On a ModelPart that never had the settings assigned, ProcessInfo[CONVECTION_DIFFUSION_SETTINGS] returns None instead of raising \u2014 and that read itself inserts the key, so a subsequent ProcessInfo.Has(...) reports True. Guarding with Has() after a read therefore passes on a model that has no settings.",
            "[Numerical] CONDUCTIVITY for LaplacianElement2D3N is read NODALLY (via ConvectionDiffusionSettings.GetDiffusionVariable), NOT from the Properties object. Swap test on a unit right triangle, reading LHS[0][0]: nodal 1 + Properties 1 -> 1.0; nodal 1 + Properties 999 -> 1.0 (Properties IGNORED); nodal 999 + Properties 1 -> 999.0. So SetSolutionStepValue(CONDUCTIVITY, k) on every node is mandatory; setting it only on Properties gives a zero-diffusivity (singular) system. DENSITY / SPECIFIC_HEAT follow the same settings-driven lookup. Signal: the solution scales with the nodal value and is unaffected by the Properties value. (Verified by execution 2026-08-03 on Kratos 10.4.0 \u2014 the prior catalog text \"Properties (CONDUCTIVITY, DENSITY, SPECIFIC_HEAT) go on Properties object, NOT on nodes\" had it exactly backwards for this element, and contradicted KNOWLEDGE['curved_mms'] pitfall #1, which was right.)",
            "[Integration] Material properties assigned via Begin Properties block in .mdpa OR via Materials.json Signal: which route is authoritative is ELEMENT-dependent, and the mismatch is silent: LaplacianElement2D3N ignores the CONDUCTIVITY on the Properties object entirely and reads the diffusion variable nominated by ConvectionDiffusionSettings off the NODES, so a Properties-only or Materials.json-only assignment yields a zero-diffusivity system with no error.",
            "[Integration] VTK output: add vtk_output_process to output_processes in ProjectParameters.json Signal: the output block resolves the core module KratosMultiphysics.vtk_output_process and its Factory, wrapping the core KM.VtkOutput class \u2014 neither is an attribute of ConvectionDiffusionApplication. A wrong module name in output_processes fails at process construction, before the solve.",
        ],
    },
}

GENERATORS = {
    "poisson_2d": _poisson_2d_kratos,
}
