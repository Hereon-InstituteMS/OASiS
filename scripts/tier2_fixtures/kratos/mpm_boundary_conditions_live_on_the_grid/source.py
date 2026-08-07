"""Tier-2: MPM boundary conditions attach to Background_Grid, never to MPM_Material.

Pitfall (kratos.mpm): in MPM the material points move and the grid does not, so
a constrained region has to be a GRID region. A constraints_process_list entry
pointed at 'MPM_Material.<name>' -- the spelling a user coming from
StructuralMechanics writes, since model_part_name in solver_settings is
"MPM_Material" -- fails at process construction:

  RuntimeError: Error: There is no sub model part with name
  "DISPLACEMENT_fix" in model part "MPM_Material"

The same block with 'Background_Grid.<name>' initialises. Materials are the
mirror image: they address the body as 'Initial_MPM_Material.<name>'.

The fixture writes its own two-mdpa deck, so it needs no data files.

Mutation control: T2_MUTATE=1 SWAPS the two model-part roots, so the deck that
is supposed to initialise points its constraint at MPM_Material and the deck
that is supposed to fail points at Background_Grid. The assertion machinery is
untouched; only the deck changes. Mutated, both initialises[...] lines flip and
the process exits non-zero.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import KratosMultiphysics as KM
from KratosMultiphysics.MPMApplication.mpm_analysis import MpmAnalysis

MUTATE = os.environ.get("T2_MUTATE") == "1"

GRID_MDPA = """Begin Properties 0
End Properties
Begin Nodes
1 1.0 0.0 0.0
2 0.0 0.0 0.0
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 1.0 2.0 0.0
6 0.0 2.0 0.0
End Nodes
Begin Elements Element2D4N
1 0 1 3 4 2
2 0 3 5 6 4
End Elements
Begin SubModelPart Parts_Grid
    Begin SubModelPartNodes
1
2
3
4
5
6
    End SubModelPartNodes
    Begin SubModelPartElements
1
2
    End SubModelPartElements
End SubModelPart
Begin SubModelPart DISPLACEMENT_fix
    Begin SubModelPartNodes
1
2
    End SubModelPartNodes
End SubModelPart
"""

BODY_MDPA = """Begin Properties 0
End Properties
Begin Nodes
3 1.0 1.0 0.0
4 0.0 1.0 0.0
5 1.0 2.0 0.0
6 0.0 2.0 0.0
End Nodes
Begin Elements MPMUpdatedLagrangian2D4N
11 0 3 5 6 4
End Elements
Begin SubModelPart Parts_Body
    Begin SubModelPartNodes
3
4
5
6
    End SubModelPartNodes
    Begin SubModelPartElements
11
    End SubModelPartElements
End SubModelPart
"""

# root used by the deck that must work, and by the deck that must fail
GOOD_ROOT, BAD_ROOT = "Background_Grid", "MPM_Material"
if MUTATE:
    print("mutation=constraint_model_part_roots_swapped")
    GOOD_ROOT, BAD_ROOT = BAD_ROOT, GOOD_ROOT


def initialises(root: str) -> tuple[bool, str]:
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as workdir:
        try:
            os.chdir(workdir)
            with open("grid.mdpa", "w") as f:
                f.write(GRID_MDPA)
            with open("body.mdpa", "w") as f:
                f.write(BODY_MDPA)
            with open("materials.json", "w") as f:
                json.dump({"properties": [{
                    "model_part_name": "Initial_MPM_Material.Parts_Body",
                    "properties_id": 1,
                    "Material": {
                        "constitutive_law":
                            {"name": "LinearElasticIsotropicPlaneStrain2DLaw"},
                        "Variables": {"THICKNESS": 1.0, "DENSITY": 7850.0,
                                      "YOUNG_MODULUS": 2.069e11,
                                      "POISSON_RATIO": 0.29,
                                      "MATERIAL_POINTS_PER_ELEMENT": 1},
                        "Tables": {}}}]}, f)
            params = {
                "problem_data": {"problem_name": "t", "parallel_type": "OpenMP",
                                 "start_time": 0.0, "end_time": 0.05,
                                 "echo_level": 0},
                "solver_settings": {
                    "solver_type": "Dynamic", "model_part_name": "MPM_Material",
                    "domain_size": 2, "echo_level": 0,
                    "analysis_type": "non_linear",
                    "time_integration_method": "implicit",
                    "scheme_type": "newmark",
                    "model_import_settings": {"input_type": "mdpa",
                                              "input_filename": "body"},
                    "material_import_settings":
                        {"materials_filename": "materials.json"},
                    "time_stepping": {"time_step": 0.05},
                    "convergence_criterion": "residual_criterion",
                    "max_iteration": 10,
                    "problem_domain_sub_model_part_list":
                        ["Parts_Grid", "Parts_Body"],
                    "processes_sub_model_part_list": ["DISPLACEMENT_fix"],
                    "grid_model_import_settings": {"input_type": "mdpa",
                                                   "input_filename": "grid"},
                    "pressure_dofs": False,
                },
                "processes": {"constraints_process_list": [{
                    "python_module": "assign_vector_variable_process",
                    "kratos_module": "KratosMultiphysics",
                    "Parameters": {
                        "model_part_name": f"{root}.DISPLACEMENT_fix",
                        "variable_name": "DISPLACEMENT",
                        "constrained": [True, True, True],
                        "value": [0.0, 0.0, 0.0],
                        "interval": [0.0, "End"]}}]},
                "output_processes": {},
            }
            MpmAnalysis(KM.Model(), KM.Parameters(json.dumps(params))).Initialize()
            return True, ""
        except Exception as exc:  # noqa: BLE001 - classifying, not handling
            return False, str(exc).replace("\n", " ")[:170]
        finally:
            os.chdir(cwd)


def main() -> int:
    mismatches = 0
    for root, must in ((GOOD_ROOT, True), (BAD_ROOT, False)):
        got, msg = initialises(root)
        print()
        print(f"initialises[{root}]={got}_expected={must}")
        if msg:
            print(f"message[{root}]={msg}")
        if got != must:
            mismatches += 1
            print(f"MISMATCH: root {root} initialises={got} expected={must}",
                  file=sys.stderr)

    print(f"constraint_root_mismatches={mismatches}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
