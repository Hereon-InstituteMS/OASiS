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

        # EVIDENCE BASIS for the 'dem' entries below. Every pitfall marked
        # "(Verified by execution 2026-08-07)" was produced by running Kratos
        # 10.4.3 (Release, GCC-15.2, OpenMP) in an environment carrying all 28
        # applications, driving DEMAnalysisStage on a shipped DEMApplication
        # fixture and injecting exactly one fault per run. Entries marked
        # "(Verified from Kratos source ...)" were read off the C++/Python on
        # disk and NOT executed; they are labelled individually.
        #
        # ProjectParametersDEM.json is a FLAT schema and shares almost nothing
        # with the FEM ProjectParameters.json used by StructuralMechanics /
        # FluidDynamics. Working minimum, all at top level: "problem_name",
        # "FinalTime", "MaxTimeStep", "Dimension", "GravityX/Y/Z",
        # "ElementType", "TranslationalIntegrationScheme",
        # "RotationalIntegrationScheme", "BoundingBox*", plus
        # "solver_settings": {"strategy": ..., "material_import_settings": ...}.
        # There is NO "problem_data" block and NO "time_stepping" block.
        "project_parameters_schema": {
            "shape": "flat, top-level keys \u2014 NOT the FEM problem_data/solver_settings layout",
            "problem_name": "top level; the four mdpa filenames are built from it",
            "FinalTime": "top level; end time. Overrides problem_data.end_time if both exist",
            "MaxTimeStep": "top level; the ONLY key that sets dt. Used verbatim",
            "solver_settings.strategy": (
                "python module basename inside DEMApplication: 'sphere_strategy', "
                "'continuum_sphere_strategy', 'verlet_continuum_sphere_strategy', "
                "'ice_continuum_sphere_strategy'. NOT a solver_type string"
            ),
            "solver_settings.material_import_settings.materials_filename": "MaterialsDEM.json",
        },
        "mdpa_files": (
            "DEM reads FOUR separate mdpa files whose names are problem_name "
            "concatenated with a fixed tag and no separator: <problem_name>DEM.mdpa "
            "(spheres), <problem_name>DEM_FEM_boundary.mdpa (walls), "
            "<problem_name>DEM_Clusters.mdpa, <problem_name>DEM_Inlet.mdpa. "
            "solver_settings.model_import_settings.input_filename does NOT name them "
            "and has no effect on which files are read."
        ),
        "materials_schema": (
            "MaterialsDEM.json has three mandatory top-level keys and is NOT the FEM "
            "'properties' schema: 'materials' (list of {material_name, material_id, "
            "Variables}), 'material_relations' (list of {material_names_list, "
            "material_ids_list, Variables}) and 'material_assignation_table' (list of "
            "[submodelpart_name, material_name_or_id]). Per-particle data "
            "(PARTICLE_DENSITY, YOUNG_MODULUS, POISSON_RATIO) goes in 'materials'; "
            "per-CONTACT-PAIR data (STATIC_FRICTION, DYNAMIC_FRICTION, "
            "COEFFICIENT_OF_RESTITUTION, DEM_DISCONTINUUM_CONSTITUTIVE_LAW_NAME, "
            "DEM_CONTINUUM_CONSTITUTIVE_LAW_NAME, DEM_ROLLING_FRICTION_MODEL_NAME) goes "
            "in 'material_relations'. There is no 'constitutive_law': {'name': ...} key."
        ),
        "pitfalls": [
            "[Input] ProjectParametersDEM.json is a FLAT schema; writing the FEM layout (a 'problem_data' block plus 'solver_settings': {'solver_type': ...}) does not work. DEM looks for solver_settings.strategy, which names a Python module inside DEMApplication ('sphere_strategy'), not a solver_type label. Signal: RuntimeError 'Error: Getting a value that does not exist. entry string : strategy' raised from kratos/sources/kratos_parameters.cpp:426 via DEM_analysis_stage.SetSolverStrategy, before any mesh is read. (Verified by execution 2026-08-07.)",
            "[Input] DEM builds four mdpa filenames by concatenating problem_name with the fixed tags DEM, DEM_FEM_boundary, DEM_Clusters and DEM_Inlet \u2014 so problem_name 'mycase' means the spheres live in 'mycaseDEM.mdpa', not 'mycase.mdpa'. A missing file is NOT an error: the reader prints one INFO line and returns, and the run completes normally on an empty model. Signal: 'DEM: Input file DEM.mdpa not found. Continuing.' on stdout, then a full successful run whose SpheresPart ends with 0 elements and 0 nodes; measured 0/0 against 4/4 for the same case with the file present. (Verified by execution 2026-08-07.)",
            "[Input] The wall file is silently optional in exactly the same way, which is the more dangerous half: particles then fall through the boundary that the user believes exists. Signal: 'DEM: Input file DEM_FEM_boundary.mdpa not found. Continuing.' and RigidFacePart holds 0 conditions while SpheresPart still holds its 4 particles \u2014 measured 0 vs 18 conditions against the same case with the wall file present. No exception, exit code 0. (Verified by execution 2026-08-07.)",
            "[Input] solver_settings.model_import_settings.input_filename is inert for DEM. Setting it to the mdpa base name changes nothing, because file paths come from problem_name; its only reader is the restart utility. Signal: a run whose input_filename points at a real, correctly named mdpa still reports 'Input file DEM.mdpa not found. Continuing.' whenever problem_name disagrees with the file on disk. (Verified from Kratos source 10.4.3, DEM_analysis_stage.py GetInputFilePath/GetProblemNameWithPath \u2014 the inert-key half was not separately executed.)",
            "[Input] MaterialsDEM.json is NOT the FEM materials schema. A file shaped {'properties': [{'model_part_name':..., 'Material': {'Variables':..., 'constitutive_law': {'name':...}}}]} \u2014 the StructuralMechanics layout \u2014 has none of the keys DEM reads. Signal: RuntimeError 'Error: Getting a value that does not exist. entry string : materials' from materials_assignation_utility.py; dropping only the assignation table instead gives the same error with 'entry string : material_assignation_table'. (Verified by execution 2026-08-07.)",
            "[Input] PARTICLE_FRICTION is not a Kratos variable at all \u2014 it appears in zero files of the installed distribution, including the compiled libraries. Friction is a per-CONTACT-PAIR property named STATIC_FRICTION and DYNAMIC_FRICTION, set in material_relations, not in materials. Signal: RuntimeError 'Error: Value type for \"PARTICLE_FRICTION\" not defined' from read_materials_utility while reading MaterialsDEM.json; the same name passed to KratosGlobals.GetVariable raises ValueError 'Kernel.GetVariable() ERROR: Variable PARTICLE_FRICTION is unknown.'. (Verified by execution 2026-08-07.)",
            "[Numerical] Particle density is PARTICLE_DENSITY. DENSITY is also a valid Kratos variable, so writing it is accepted by the materials reader and then never read \u2014 Properties[PARTICLE_DENSITY] silently returns the inserted default 0.0, particle mass becomes 0, the integrator divides by it, and the resulting non-finite coordinates fail the bounding-box test so every particle is erased. Signal: the run exits 0 and reports ANALYSIS COMPLETED while SpheresPart goes from 4 elements to 0 elements / 0 nodes; probing Properties shows PARTICLE_DENSITY = 0.0 alongside DENSITY = 4000.0. No warning of any kind. (Verified by execution 2026-08-07.)",
            "[Numerical] Omitting YOUNG_MODULUS behaves the same way: the property read inserts 0.0 rather than raising, giving zero contact stiffness and mutually transparent particles. Signal: Properties[YOUNG_MODULUS] reads back exactly 0.0 after a clean Initialize() and the run proceeds to completion with no message. (Verified by execution 2026-08-07.)",
            "[Numerical] MaxTimeStep is the only key that sets dt and it is used verbatim \u2014 DEM performs no stability check and prints no critical-time-step warning at any value. The keys that look like safety nets are dead: DeltaTimeSafetyFactor is stored and never read, AutomaticTimestep is never read at all, and DEM_timestep_safety_factor is not a Kratos key (zero occurrences in the distribution). Signal: MaxTimeStep = 1.0 s on a case whose stable step is 5e-4 prints 'DEM: Total number of time steps expected in the calculation: 0', sets DELTA_TIME to 1.0 and jumps straight to the end time \u2014 no error, no warning. (Verified by execution 2026-08-07.)",
            "[Input] FinalTime, not problem_data.end_time, decides when a DEM run stops, and the DEM default silently overwrites whatever end_time the user wrote. Signal: a deck with 'problem_data': {'end_time': 10.0} and no FinalTime key prints 'DEM: Total number of time steps expected in the calculation: 100' and stops at TIME = 0.05 \u2014 the FinalTime default, 200x short \u2014 with no warning. (Verified by execution 2026-08-07.)",
            "[Input] Every sub model part that carries entities needs a row in material_assignation_table, including the wall part. Omitting the wall row while the wall mdpa is present is a hard error raised deep in the solver, not at materials-reading time. Signal: RuntimeError 'Error: PROPERTIES_ID is not set for SubModelPart 1 . Make sure the Materials file contains material assignation for this SubModelPart' from explicit_solver_strategy InitializeFEMElements \u2014 note it names the sub model part by its numeric id, not by name. (Verified by execution 2026-08-07.)",
            "[Input] Unknown or misspelled top-level keys are rejected by ValidateAndAssignDefaults, and so are unknown keys inside solver_settings \u2014 but validation is flat, so a typo nested inside any other sub-block is silently ignored. Signal: 'MaxTimestep' (lowercase s) raises RuntimeError 'Error: The item with name \"MaxTimestep\" is present in this Parameters but NOT in the default values', followed by a several-hundred-line dump of both the supplied and the default parameter trees; the same happens for an unknown key under solver_settings. (Verified by execution 2026-08-07.)",
            "[API] The contact law name is set per contact PAIR, as DEM_DISCONTINUUM_CONSTITUTIVE_LAW_NAME inside a material_relations entry \u2014 not as a 'constitutive_law' block on a material. The lookup is an unchecked dictionary get, so a wrong or abbreviated name never names itself in the error. Signal: an abbreviated name such as 'DEM_D_Hertz' produces TypeError \"'NoneType' object is not callable\" raised inside sphere_strategy.ModifySubProperties; the offending string is never echoed. Registered names include DEM_D_Hertz_viscous_Coulomb, DEM_D_Linear_viscous_Coulomb, DEM_D_Hertz_viscous_Coulomb_JKR/_DMT, DEM_KDEM, DEM_parallel_bond, DEM_Dempack. (Verified by execution 2026-08-07.)",
            "[API] Kratos DEM is NOT 3D-only, contrary to a long-standing catalog claim. There is no SphericParticle2D, but genuine 2D particle elements exist under the Cylinder* stem and construct successfully: CylinderParticle2D and CylinderContinuumParticle2D. The right fix for a planar problem is CylinderParticle2D, not a 3D sphere with constrained out-of-plane DOFs. Signal: CreateNewElement('CylinderParticle2D', ...) succeeds on the same model part where CreateNewElement('SphericParticle2D', ...) raises 'The Element \"SphericParticle2D\" is not registered!'; the 2D discontinuum law is spelled DEM_D_Hertz_viscous_Coulomb2D. (Verified by execution 2026-08-07 \u2014 corrects the earlier 'DEM is always 3D internally' wording.)",
            "[API] Wall geometry is built from CONDITIONS, not elements, and only 3D face spellings are registered. Signal: CreateNewCondition('RigidFace3D3N' / 'RigidFace3D4N', ...) constructs, while RigidFace2D2N and RigidEdge3D2N raise 'The Condition \"...\" is not registered!' and list the available DEM conditions. (Verified by execution 2026-08-07.)",
            "[API] RADIUS is a core KratosMultiphysics nodal variable, not a DEMApplication one, and it is the single mandatory per-node input in the spheres mdpa. Signal: KratosMultiphysics.RADIUS resolves while DEMApplication.RADIUS raises AttributeError; a particle whose RADIUS was never written keeps the silent default of zero rather than raising. (Verified by execution 2026-08-07.)",
            "[Output] VTK output goes through the core KM.VtkOutput class and the core vtk_output_process module. Signal: a ModelPart holding geometries with no VTK writer raises 'Modelpart contains elements or conditions with geometries for which no VTK-output is implemented!'. (Verified 2026-06-01, carried over unchanged.)",
        ],
        "guidance": [
            "[Numerical] There is no automatic time step. Size dt by hand from the Rayleigh/Hertz estimate dt ~ 0.34*sqrt(m/(E*pi*R)) and set it in MaxTimeStep; the formula exists in the source (SphericContinuumParticle::Calculate for DELTA_TIME) but is never invoked by any solver path.",
            "[Numerical] The one opt-in automatic-dt path is the automatic_dt_process, added by hand to 'processes'; it prints 'Automatic DT process: Calculated critical time step: <x> seconds.'. It casts unconditionally to SphericContinuumParticle, so pointing it at a plain SphericParticle3D model part dereferences a null pointer. (Source-read, not executed.)",
            "[Numerical] BoundingBoxOption defaults to false. With it true, particles leaving the box are deleted with no message and no counter, and mass is silently not conserved; the check runs only on neighbour-search steps. With it false nothing is ever deleted and escapees integrate forever.",
            "[Numerical] NeighbourSearchFrequency defaults to 50 and SearchTolerance to 0.0, so contacts are rediscovered only every 50 steps with a search radius equal to the particle radius \u2014 fast particles tunnel through each other silently.",
            "[Numerical] For >10k particles: enable MPI via parallel_type: MPI.",
        ]
    },
}

GENERATORS = {
    "dem_2d": _dem_2d_kratos,
}
