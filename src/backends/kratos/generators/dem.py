"""Kratos DEM (Discrete Element Method) generators and knowledge.

Uses the REAL Kratos DEMApplication — NOT standalone numpy/scipy.
Generates ProjectParametersDEM.json + .mdpa + MaterialsDEM.json + input.py.
"""

import json


def _dem_2d_kratos(params: dict) -> str:
    """Generator that produces real Kratos DEM input files.

    Creates: ProjectParametersDEM.json, particles.mdpa, MaterialsDEM.json, input.py
    The run_with_generator pipeline will find input.py and execute it via Kratos.
    """
    n_particles = params.get("n_particles", 200)
    radius = params.get("radius", 0.01)
    density = params.get("density", 2650.0)
    gravity_y = params.get("gravity", -9.81)
    dt = params.get("dt", 1e-5)
    T_end = params.get("T_end", 0.5)
    young_modulus = params.get("E", 1e7)
    restitution = params.get("restitution", 0.5)
    friction = params.get("friction", 0.4)
    domain_x = params.get("domain_x", 0.5)
    domain_y = params.get("domain_y", 1.0)

    return f'''\
"""Generator: creates real Kratos DEM input files and runs the analysis."""
import os
import json
import math
import numpy as np

# === Parameters ===
n_particles = {n_particles}
radius = {radius}
density = {density}
gravity_y = {gravity_y}
dt = {dt}
T_end = {T_end}
E_contact = {young_modulus}
restitution = {restitution}
friction = {friction}
domain_x = {domain_x}
domain_y = {domain_y}

np.random.seed(42)

# === Generate particle positions ===
cols = int(np.ceil(np.sqrt(n_particles)))
spacing = 2.5 * radius
positions = []
for i in range(n_particles):
    row, col = divmod(i, cols)
    x = (col + 1) * spacing + np.random.uniform(-radius*0.1, radius*0.1)
    y = domain_y * 0.5 + (row + 1) * spacing
    x = min(max(x, radius), domain_x - radius)
    positions.append((x, y, 0.0))

# === Write MDPA ===
mdpa = "Begin ModelPartData\\nEnd ModelPartData\\n\\n"
mdpa += "Begin Properties 1\\n"
mdpa += f"  PARTICLE_DENSITY {{density}}\\n"
mdpa += f"  YOUNG_MODULUS {{E_contact}}\\n"
mdpa += f"  POISSON_RATIO 0.25\\n"
mdpa += f"  PARTICLE_FRICTION {{friction}}\\n"
mdpa += f"  COEFFICIENT_OF_RESTITUTION {{restitution}}\\n"
mdpa += f"  PARTICLE_MATERIAL 1\\n"
mdpa += f"  ROLLING_FRICTION 0.01\\n"
mdpa += "End Properties\\n\\n"

mdpa += "Begin Nodes\\n"
for i, (x, y, z) in enumerate(positions, 1):
    mdpa += f"  {{i}} {{x:.8f}} {{y:.8f}} {{z:.8f}}\\n"
mdpa += "End Nodes\\n\\n"

mdpa += "Begin Elements SphericParticle3D\\n"
for i in range(1, n_particles + 1):
    mdpa += f"  {{i}} 1 {{i}}\\n"
mdpa += "End Elements\\n\\n"

# All particles in one sub model part
mdpa += "Begin SubModelPart DEMParts_particles\\n"
mdpa += "  Begin SubModelPartData\\n"
mdpa += "  End SubModelPartData\\n"
mdpa += "  Begin SubModelPartNodes\\n"
for i in range(1, n_particles + 1):
    mdpa += f"    {{i}}\\n"
mdpa += "  End SubModelPartNodes\\n"
mdpa += "End SubModelPart\\n"

with open("particles.mdpa", "w") as f:
    f.write(mdpa)

# === Write MaterialsDEM.json ===
materials = {{
    "properties": [{{
        "model_part_name": "SpheresPart",
        "properties_id": 1,
        "hydrodynamic_law_parameters": {{}},
        "Material": {{
            "Variables": {{
                "PARTICLE_DENSITY": density,
                "YOUNG_MODULUS": E_contact,
                "POISSON_RATIO": 0.25,
                "PARTICLE_FRICTION": friction,
                "COEFFICIENT_OF_RESTITUTION": restitution,
                "PARTICLE_MATERIAL": 1,
                "ROLLING_FRICTION": 0.01,
                "ROLLING_FRICTION_WITH_WALLS": 0.01,
            }},
            "constitutive_law": {{
                "name": "DEM_D_Hertz_viscous_Coulomb"
            }}
        }}
    }}]
}}
with open("MaterialsDEM.json", "w") as f:
    json.dump(materials, f, indent=2)

# === Write ProjectParametersDEM.json ===
n_steps = int(T_end / dt)
output_interval = max(1, n_steps // 20)

project_params = {{
    "problem_data": {{
        "problem_name": "particles",
        "parallel_type": "OpenMP",
        "echo_level": 0,
        "start_time": 0.0,
        "end_time": T_end
    }},
    "solver_settings": {{
        "solver_type": "dem_solver",
        "model_part_name": "SpheresPart",
        "domain_size": 3,
        "model_import_settings": {{
            "input_type": "mdpa",
            "input_filename": "particles"
        }},
        "material_import_settings": {{
            "materials_filename": "MaterialsDEM.json"
        }},
        "time_stepping": {{
            "time_step": dt
        }},
        "body_force_per_unit_mass": [0.0, gravity_y, 0.0],
        "DEM_timestep_safety_factor": 0.5,
        "MaxAmplificationRatioOfSearchRadius": 0.0,
        "BoundingBoxMaxX": domain_x,
        "BoundingBoxMaxY": domain_y * 2,
        "BoundingBoxMaxZ": 0.1,
        "BoundingBoxMinX": 0.0,
        "BoundingBoxMinY": 0.0,
        "BoundingBoxMinZ": -0.1,
        "BoundingBoxEnlargementFactor": 1.1,
        "AutomaticBoundingBoxOption": False,
        "ContactMeshOption": "use_particles"
    }},
    "output_processes": {{
        "vtk_output": [{{
            "python_module": "vtk_output_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "VtkOutputProcess",
            "Parameters": {{
                "model_part_name": "SpheresPart",
                "output_control_type": "step",
                "output_interval": output_interval,
                "file_format": "ascii",
                "output_path": "vtk_output",
                "nodal_solution_step_data_variables": ["VELOCITY", "TOTAL_FORCES"]
            }}
        }}]
    }},
    "processes": {{
        "walls_process_list": [{{
            "python_module": "assign_vector_variable_process",
            "kratos_module": "KratosMultiphysics",
            "process_name": "AssignVectorVariableProcess",
            "Parameters": {{
                "model_part_name": "SpheresPart.DEMParts_particles",
                "variable_name": "RADIUS",
                "value": [radius, 0.0, 0.0],
                "constrained": [False, False, False],
                "interval": [0, "End"]
            }}
        }}]
    }}
}}

with open("ProjectParametersDEM.json", "w") as f:
    json.dump(project_params, f, indent=2)

# === Write input.py (the entry point) ===
input_py = """import KratosMultiphysics
from KratosMultiphysics.DEMApplication.DEM_analysis_stage import DEMAnalysisStage

if __name__ == "__main__":
    with open("ProjectParametersDEM.json", "r") as f:
        project_parameters = KratosMultiphysics.Parameters(f.read())
    model = KratosMultiphysics.Model()
    DEMAnalysisStage(model, project_parameters).Run()
"""

with open("input.py", "w") as f:
    f.write(input_py)

print(f"Generated Kratos DEM input files:")
print(f"  particles.mdpa: {{n_particles}} particles")
print(f"  ProjectParametersDEM.json: dt={{dt}}, T={{T_end}}")
print(f"  MaterialsDEM.json: Hertz-viscous-Coulomb")
print(f"  input.py: DEMAnalysisStage entry point")
'''


KNOWLEDGE = {
    "dem": {
        "description": "Discrete Element Method via Kratos DEMApplication",
        "application": "DEMApplication (pip install KratosDEMApplication)",
        "workflow": (
            "Kratos DEM uses ProjectParametersDEM.json + .mdpa + MaterialsDEM.json. "
            "The entry point is DEMAnalysisStage(model, parameters).Run(). "
            "Use run_with_generator where the generator creates all input files "
            "and writes input.py as the execution script."
        ),
        "elements": {
            "3D": ["SphericParticle3D (sphere-sphere contact)", "SphericContinuumParticle3D (bonded)"],
            "2D": ["CylinderParticle2D (disk in 2D)", "CylinderContinuumParticle2D"],
            "cluster": ["Cluster3D (rigid body composed of multiple spheres)"],
        },
        "contact_models": {
            "normal": ["DEM_D_Hertz_viscous_Coulomb (recommended)", "DEM_D_Linear_viscous_Coulomb",
                       "DEM_D_Hertz_viscous_Coulomb_JKR (adhesive)", "DEM_D_Hertz_viscous_Coulomb_DMT"],
            "tangential": ["Coulomb friction (built into normal law)"],
            "rolling": ["DEMRollingFrictionModelConstantTorque", "DEMRollingFrictionModelViscousTorque"],
            "bonding": ["DEM_KDEM (parallel bond)", "DEM_parallel_bond", "DEM_Dempack"],
        },
        "solver_types": ["explicit (velocity Verlet, default)", "explicit (symplectic Euler)"],
        "pitfalls": [
                        '[Numerical] Time step must satisfy CFL: dt < min(2*sqrt(m/k)) — use DEM_timestep_safety_factor: 0.5 '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Numerical] Bounding box must enclose all particles at all times (AutomaticBoundingBoxOption or manual) '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        "[Integration] MaterialsDEM.json constitutive_law name must match exactly (e.g. 'DEM_D_Hertz_viscous_Coulomb') "
                        "Signal: RuntimeError 'KeyError' from JSON parsing OR 'SubModelPart not found' / 'Property ID ... missing' during AnalysisStage.Initialize.",
                        '[Integration] MDPA uses SphericParticle3D even for 2D problems (Kratos DEM is always 3D internally) '
                        "Signal: RuntimeError 'KeyError' from JSON parsing OR 'SubModelPart not found' / 'Property ID ... missing' during AnalysisStage.Initialize.",
                        '[Integration] Wall geometry: use RigidFace3D3N elements or the walls_process_list in ProjectParameters '
                        "Signal: RuntimeError 'KeyError' from JSON parsing OR 'SubModelPart not found' / 'Property ID ... missing' during AnalysisStage.Initialize.",
                        '[Numerical] For >10k particles: enable MPI via parallel_type: MPI '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                        '[Integration] RADIUS must be set per-particle in the MDPA or via a process '
                        "Signal: RuntimeError 'KeyError' from JSON parsing OR 'SubModelPart not found' / 'Property ID ... missing' during AnalysisStage.Initialize.",
                        '[Numerical] Output: VTK with sphere glyphs, or GiD .post.bin format '
                        "Signal: solver reports 'Convergence is not achieved' / 'iteration count exceeded' / oscillating residual; reported quantity disagrees with analytic reference by an order-of-magnitude factor.",
                    ],
    },
}

GENERATORS = {
    "dem_2d": _dem_2d_kratos,
}
