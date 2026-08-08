"""Tier-2: MATERIAL_POINTS_PER_ELEMENT is mandatory, and lives in the MATERIALS json.

Pitfall (kratos.mpm): the key that decides how many material points each body
element is seeded with is an int in the materials file, under
properties[i].Material.Variables -- not in ProjectParameters. On the installed
10.4.3 build its absence is a hard error:

  RuntimeError: Error: "MATERIAL_POINTS_PER_ELEMENT" is not specified in Properties

and an unsupported count is rejected with a message that names the GRID geometry:

  RuntimeError: Error: The input number of MATERIAL_POINTS_PER_ELEMENT (5)
  is not available for Quadrilateral elements

(Allowed: 1/4/9/16 for quad+hex, 1/3/6/12/16/33 for tri+tet.)

The fixture writes its own two-mdpa deck -- a 1x2 column of quadrilateral grid
cells with a single body element in the top cell -- so it needs no data files.
It asserts three verdicts: 1 is accepted, omission raises, and 5 raises.

Mutation control: T2_MUTATE=1 INVERTS all three expectations -- it claims 1
must fail while omission and 5 must succeed. The solver invocation is
untouched. Mutated, every accepted[...] line disagrees with itself and
material_point_count_mismatches goes 0 -> 3.
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


def build(workdir: str, mpp) -> dict:
    """Write grid/body/materials/parameters. mpp=None omits the key."""
    with open(os.path.join(workdir, "grid.mdpa"), "w") as f:
        f.write(GRID_MDPA)
    with open(os.path.join(workdir, "body.mdpa"), "w") as f:
        f.write(BODY_MDPA)

    variables = {"THICKNESS": 1.0, "DENSITY": 7850.0,
                 "YOUNG_MODULUS": 2.069e11, "POISSON_RATIO": 0.29}
    if mpp is not None:
        variables["MATERIAL_POINTS_PER_ELEMENT"] = mpp
    materials = {"properties": [{
        "model_part_name": "Initial_MPM_Material.Parts_Body",
        "properties_id": 1,
        "Material": {
            "constitutive_law": {"name": "LinearElasticIsotropicPlaneStrain2DLaw"},
            "Variables": variables,
            "Tables": {},
        },
    }]}
    with open(os.path.join(workdir, "materials.json"), "w") as f:
        json.dump(materials, f)

    return {
        "problem_data": {"problem_name": "t", "parallel_type": "OpenMP",
                         "start_time": 0.0, "end_time": 0.05, "echo_level": 0},
        "solver_settings": {
            "solver_type": "Dynamic", "model_part_name": "MPM_Material",
            "domain_size": 2, "echo_level": 0, "analysis_type": "non_linear",
            "time_integration_method": "implicit", "scheme_type": "newmark",
            "model_import_settings": {"input_type": "mdpa", "input_filename": "body"},
            "material_import_settings": {"materials_filename": "materials.json"},
            "time_stepping": {"time_step": 0.05},
            "convergence_criterion": "residual_criterion",
            "max_iteration": 10,
            "problem_domain_sub_model_part_list": ["Parts_Grid", "Parts_Body"],
            "processes_sub_model_part_list": ["DISPLACEMENT_fix"],
            "grid_model_import_settings": {"input_type": "mdpa",
                                           "input_filename": "grid"},
            "pressure_dofs": False,
        },
        "processes": {"constraints_process_list": [{
            "python_module": "assign_vector_variable_process",
            "kratos_module": "KratosMultiphysics",
            "Parameters": {"model_part_name": "Background_Grid.DISPLACEMENT_fix",
                           "variable_name": "DISPLACEMENT",
                           "constrained": [True, True, True],
                           "value": [0.0, 0.0, 0.0], "interval": [0.0, "End"]}}]},
        "output_processes": {},
    }


def initialises(mpp) -> tuple[bool, str]:
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as workdir:
        try:
            os.chdir(workdir)
            params = build(workdir, mpp)
            MpmAnalysis(KM.Model(), KM.Parameters(json.dumps(params))).Initialize()
            return True, ""
        except Exception as exc:  # noqa: BLE001 - classifying, not handling
            return False, str(exc).replace("\n", " ")[:150]
        finally:
            os.chdir(cwd)


# label -> (value, must Initialize() succeed?)
CASES = [("mpp=1", 1, True), ("mpp=omitted", None, False), ("mpp=5", 5, False)]
if MUTATE:
    print("mutation=material_point_count_expectations_inverted")
    CASES = [(label, val, not ok) for label, val, ok in CASES]


def main() -> int:
    mismatches = 0
    for label, value, must in CASES:
        got, msg = initialises(value)
        print()
        print(f"accepted[{label}]={got}_expected={must}")
        if msg:
            print(f"message[{label}]={msg}")
        if got != must:
            mismatches += 1
            print(f"MISMATCH: {label} accepted={got} expected={must}", file=sys.stderr)

    print(f"material_point_count_mismatches={mismatches}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
