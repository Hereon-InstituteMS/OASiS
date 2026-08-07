"""Tier-2: MPM material points that leave the background grid are ERASED.

Pitfall (kratos.mpm): the background grid must cover the whole TRAJECTORY of the
body, not just its initial position. A material point whose search fails has its
geometry cleared and is then deleted by MaterialPointEraseProcess, which runs by
default (element_search_settings.remove_entities_not_found defaults to true).
The mass it carried leaves the simulation. The receipt is two log lines, not an
error, so a partially-escaped body silently loses mass:

  MPMSearchElementUtility: WARNING: Search Element for Material Point: <id>
  is failed. Geometry is cleared.
  [WARNING] MaterialPointEraseProcess: 1 particle elements have been erased.

Only once the LAST point is gone does the run stop, and then with a message that
does not mention material points at all:

  RuntimeError: Error: No degrees of freedom in model part: MPM_Material

This fixture drops a free-falling body out of the bottom of a short grid and
asserts the element count goes to zero. It writes its own deck.

Mutation control: T2_MUTATE=1 EXTENDS the grid downward so the body stays inside
for the whole run, without touching the body, the gravity or the assertions.
The point then survives, so surviving_material_points goes 0 -> 1 and
material_points_were_erased flips to False, and the process exits non-zero.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import KratosMultiphysics as KM
from KratosMultiphysics.MPMApplication.mpm_analysis import MpmAnalysis

MUTATE = os.environ.get("T2_MUTATE") == "1"

# The grid spans y in [GRID_BOTTOM, 2]; the body sits in y in [1, 2] and falls.
GRID_BOTTOM = 0.0
if MUTATE:
    print("mutation=grid_extended_downward_so_the_body_stays_inside")
    GRID_BOTTOM = -400.0


def grid_mdpa() -> str:
    return f"""Begin Properties 0
End Properties
Begin Nodes
1 1.0 {GRID_BOTTOM} 0.0
2 0.0 {GRID_BOTTOM} 0.0
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


def run() -> tuple[int, int, str]:
    """Return (initial material points, surviving material points, message)."""
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as workdir:
        try:
            os.chdir(workdir)
            with open("grid.mdpa", "w") as f:
                f.write(grid_mdpa())
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
                                 "start_time": 0.0, "end_time": 2.0,
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
                    "time_stepping": {"time_step": 0.1},
                    "convergence_criterion": "residual_criterion",
                    "max_iteration": 10,
                    "problem_domain_sub_model_part_list":
                        ["Parts_Grid", "Parts_Body"],
                    "processes_sub_model_part_list": [],
                    "grid_model_import_settings": {"input_type": "mdpa",
                                                   "input_filename": "grid"},
                    "pressure_dofs": False,
                },
                "processes": {"gravity": [{
                    "python_module": "assign_gravity_to_material_point_process",
                    "kratos_module": "KratosMultiphysics.MPMApplication",
                    "Parameters": {"model_part_name": "MPM_Material",
                                   "modulus": 9.81,
                                   "direction": [0.0, -1.0, 0.0]}}]},
                "output_processes": {},
            }
            model = KM.Model()
            sim = MpmAnalysis(model, KM.Parameters(json.dumps(params)))
            sim.Initialize()
            mp = model.GetModelPart("MPM_Material")
            before = len(mp.Elements)
            msg = ""
            try:
                sim.RunSolutionLoop()
            except Exception as exc:  # noqa: BLE001 - the empty-model stop is expected
                msg = str(exc).replace("\n", " ")[:130]
            return before, len(mp.Elements), msg
        finally:
            os.chdir(cwd)


def main() -> int:
    before, after, msg = run()
    erased = after == 0 and before > 0
    print(f"initial_material_points={before}")
    print(f"surviving_material_points={after}")
    print(f"material_points_were_erased={erased}_expected=True")
    if msg:
        print(f"stop_message={msg}")

    if not erased:
        print(f"MISMATCH: expected every material point to be erased, "
              f"{after} of {before} survived", file=sys.stderr)
        return 1

    print("mpm_grid_escape_check=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
