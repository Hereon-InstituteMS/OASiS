"""Tier-2: an unknown key in Kratos Parameters is a LOUD failure, not a silent one.

Catalog claim falsified (KNOWLEDGE['structural_dynamics'], pre-2026-08-03):

    "In the Kratos JSON the parameter name is 'damp_factor_m' (NOT alpha_m).
     Wrong key is silently ignored and the scheme runs without damping."

Kratos Parameters::ValidateAndAssignDefaults rejects unknown keys, so a full
StructuralMechanicsAnalysis with scheme_type 'bossak' and the key renamed to
'alpha_m' aborts before the first time step. The correct key runs to completion.

This matters operationally: an agent told "the wrong key is silently ignored"
will look for a PHYSICS explanation of a missing-damping symptom that never
occurs, instead of reading the exception.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import KratosMultiphysics as KM
from KratosMultiphysics.StructuralMechanicsApplication.structural_mechanics_analysis import (
    StructuralMechanicsAnalysis)

# ---- level 1: the validation primitive itself -----------------------------
defaults = KM.Parameters('{"scheme_type":"bossak","damp_factor_m":-0.3}')
wrong = KM.Parameters('{"scheme_type":"bossak","alpha_m":-0.3}')
try:
    wrong.ValidateAndAssignDefaults(defaults)
    validate_msg = "SILENTLY ACCEPTED"
except Exception as exc:  # noqa: BLE001
    validate_msg = str(exc).splitlines()[0]
print(f"validate_unknown_key_raises={'SILENTLY' not in validate_msg}")
print(f"validate_message_mentions_not_in_default_values="
      f"{'NOT in the default values' in validate_msg}")

# ---- level 2: a real dynamic analysis -------------------------------------
WORK = tempfile.mkdtemp(prefix="kratos_bossak_")
CWD = os.getcwd()
os.chdir(WORK)

NX, NY, LX, LY = 6, 1, 6.0, 1.0
nid, nodes, k = {}, [], 1
for j in range(NY + 1):
    for i in range(NX + 1):
        nid[(i, j)] = k
        nodes.append(f"  {k}  {i * LX / NX:.10f}  {j * LY / NY:.10f}  0.0")
        k += 1
elems, eid = [], 1
for j in range(NY):
    for i in range(NX):
        a, b, c, d = nid[(i, j)], nid[(i + 1, j)], nid[(i + 1, j + 1)], nid[(i, j + 1)]
        elems.append(f"  {eid} 1 {a} {b} {d}"); eid += 1
        elems.append(f"  {eid} 1 {b} {c} {d}"); eid += 1
left = [nid[(0, j)] for j in range(NY + 1)]
tip = [nid[(NX, j)] for j in range(NY + 1)]

mdpa = "\n".join([
    "Begin Properties 1\nEnd Properties\n",
    "Begin Nodes\n" + "\n".join(nodes) + "\nEnd Nodes\n",
    "Begin Elements SmallDisplacementElement2D3N\n" + "\n".join(elems) + "\nEnd Elements\n",
    "Begin Conditions PointLoadCondition2D1N\n"
    + "\n".join(f"  {i + 1} 1 {n}" for i, n in enumerate(tip)) + "\nEnd Conditions\n",
    "Begin SubModelPart Parts_Body\nBegin SubModelPartNodes\n"
    + "\n".join(f"  {n}" for n in range(1, k)) + "\nEnd SubModelPartNodes\n"
    + "Begin SubModelPartElements\n" + "\n".join(f"  {e}" for e in range(1, eid))
    + "\nEnd SubModelPartElements\nEnd SubModelPart\n",
    "Begin SubModelPart DISPLACEMENT_Fixed\nBegin SubModelPartNodes\n"
    + "\n".join(f"  {n}" for n in left) + "\nEnd SubModelPartNodes\nEnd SubModelPart\n",
    "Begin SubModelPart PointLoad_Tip\nBegin SubModelPartNodes\n"
    + "\n".join(f"  {n}" for n in tip) + "\nEnd SubModelPartNodes\n"
    + "Begin SubModelPartConditions\n"
    + "\n".join(f"  {i + 1}" for i in range(len(tip))) + "\nEnd SubModelPartConditions\n"
    + "End SubModelPart\n",
])
open("beam.mdpa", "w").write(mdpa)
open("StructuralMaterials.json", "w").write(json.dumps({"properties": [{
    "model_part_name": "Structure.Parts_Body", "properties_id": 1,
    "Material": {"constitutive_law": {"name": "LinearElasticPlaneStress2DLaw"},
                 "Variables": {"DENSITY": 7850.0, "YOUNG_MODULUS": 2.0e11,
                               "POISSON_RATIO": 0.0, "THICKNESS": 1.0},
                 "Tables": {}}}]}))


def project_parameters(damping_key: str) -> dict:
    return {
        "problem_data": {"problem_name": "beam", "parallel_type": "OpenMP",
                         "echo_level": 0, "start_time": 0.0, "end_time": 0.02},
        "solver_settings": {
            "solver_type": "Dynamic", "model_part_name": "Structure", "domain_size": 2,
            "echo_level": 0, "analysis_type": "linear",
            "time_integration_method": "implicit", "scheme_type": "bossak",
            damping_key: -0.3,
            "model_import_settings": {"input_type": "mdpa", "input_filename": "beam"},
            "material_import_settings": {"materials_filename": "StructuralMaterials.json"},
            "time_stepping": {"time_step": 0.01}, "rotation_dofs": False,
            "linear_solver_settings": {"solver_type": "LinearSolversApplication.sparse_lu"}},
        "processes": {
            "constraints_process_list": [{
                "python_module": "assign_vector_variable_process",
                "kratos_module": "KratosMultiphysics",
                "process_name": "AssignVectorVariableProcess",
                "Parameters": {"model_part_name": "Structure.DISPLACEMENT_Fixed",
                               "variable_name": "DISPLACEMENT",
                               "constrained": [True, True, True],
                               "value": [0.0, 0.0, 0.0], "interval": [0.0, "End"]}}],
            "loads_process_list": [{
                "python_module": "assign_vector_by_direction_to_condition_process",
                "kratos_module": "KratosMultiphysics",
                "process_name": "AssignVectorByDirectionToConditionProcess",
                "Parameters": {"model_part_name": "Structure.PointLoad_Tip",
                               "variable_name": "POINT_LOAD", "modulus": 500.0,
                               "direction": [0.0, -1.0, 0.0], "interval": [0.0, "End"]}}],
            "list_other_processes": []},
        "output_processes": {},
    }


def run(damping_key: str) -> str:
    params = KM.Parameters(json.dumps(project_parameters(damping_key)))
    try:
        StructuralMechanicsAnalysis(KM.Model(), params).Run()
        return "OK"
    except Exception as exc:  # noqa: BLE001
        return "EXC: " + str(exc).splitlines()[0]


wrong_run = run("alpha_m")
right_run = run("damp_factor_m")
os.chdir(CWD)
shutil.rmtree(WORK, ignore_errors=True)

print(f"analysis_with_alpha_m={wrong_run[:120]}")
print(f"analysis_with_damp_factor_m={right_run[:120]}")
print(f"wrong_key_aborts_the_run={wrong_run.startswith('EXC')}")
print(f"wrong_key_is_silently_ignored={wrong_run == 'OK'}")
print(f"correct_key_runs={right_run == 'OK'}")

if not (wrong_run.startswith("EXC") and right_run == "OK"):
    print("FAIL: fixture expectations not met")
    sys.exit(1)
