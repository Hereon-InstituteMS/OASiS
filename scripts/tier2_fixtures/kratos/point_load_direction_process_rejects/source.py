"""Tier-2: which process may be used to apply POINT_LOAD.

Two catalog entries disagreed before 2026-08-03:

  structural_dynamics: "AssignVectorByDirectionProcess crashes for load
     variables because it tries to fix/free DOFs.  Signal: RuntimeError
     'Trying to fix DOF of non-existing variable' or segfault"
  linear_elasticity:  "the directional-magnitude process the agent might be
     tempted to use does not exist in current StructuralMechanicsApplication
     ... the prior claim 'crashes for load variables' was misleading because
     the named class is not available to crash"

Both are half right. Measured on Kratos 10.4.0:
  * StructuralMechanicsApplication really has no attribute of that name, AND
  * the CORE module KratosMultiphysics.assign_vector_by_direction_process
    really does crash on POINT_LOAD, with the originally-documented message
    "Trying to fix/free dof of variable POINT_LOAD_X but this dof does not
    exist in node #1!"
  * assign_vector_variable_process sets POINT_LOAD = [0, -100, 0] cleanly.
  * assign_vector_by_direction_to_condition_process is the right directional
    process for loads carried by Conditions.
"""
from __future__ import annotations

import json
import sys

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
from KratosMultiphysics import assign_vector_by_direction_process as avbd
from KratosMultiphysics import assign_vector_variable_process as avv


def apply(mod, extra):
    model = KM.Model()
    mp = model.CreateModelPart("Structure")
    mp.AddNodalSolutionStepVariable(SMA.POINT_LOAD)
    mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
    sub = mp.CreateSubModelPart("Load")
    node = mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    sub.AddNodes([1])
    par = {"model_part_name": "Structure.Load", "variable_name": "POINT_LOAD",
           "interval": [0.0, 1e30]}
    par.update(extra)
    try:
        proc = mod.Factory(KM.Parameters(json.dumps({"Parameters": par})), model)
        proc.ExecuteInitialize()
        proc.ExecuteInitializeSolutionStep()
        v = node.GetSolutionStepValue(SMA.POINT_LOAD)
        return "OK [%.1f, %.1f, %.1f]" % (v[0], v[1], v[2])
    except Exception as exc:  # noqa: BLE001
        return "ERR " + str(exc).splitlines()[0]


by_dir = apply(avbd, {"modulus": 100.0, "direction": [0.0, -1.0, 0.0]})
by_val = apply(avv, {"value": [0.0, -100.0, 0.0], "constrained": [False, False, False]})

print(f"sma_has_AssignVectorByDirectionProcess={hasattr(SMA, 'AssignVectorByDirectionProcess')}")
print(f"core_by_direction_result={by_dir[:150]}")
print(f"assign_vector_variable_result={by_val[:150]}")
print(f"core_by_direction_raises={by_dir.startswith('ERR')}")
print(f"raises_about_fix_free_dof="
      f"{'fix/free dof of variable POINT_LOAD_X' in by_dir}")
print(f"assign_vector_variable_works={by_val.startswith('OK')}")

ok = (not hasattr(SMA, "AssignVectorByDirectionProcess")
      and by_dir.startswith("ERR")
      and "fix/free dof of variable POINT_LOAD_X" in by_dir
      and by_val.startswith("OK"))
if not ok:
    print("FAIL: fixture expectations not met")
    sys.exit(1)
