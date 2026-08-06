"""Tier-2: non-homogeneous Dirichlet needs constrained=True.

Pitfall (kratos.heat #1). With constrained=False the process
assigns the value but does NOT fix the DOF, so a solver is free
to overwrite it. The discriminator is Node.IsFixed().
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401
from KratosMultiphysics import assign_scalar_variable_process as asvp


def run(constrained: bool):
    model = KM.Model()
    mp = model.CreateModelPart("h" + str(constrained))
    mp.AddNodalSolutionStepVariable(KM.TEMPERATURE)
    mp.SetBufferSize(2)
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    mp.CreateNewNode(2, 1.0, 0.0, 0.0)
    mp.CreateNewNode(3, 1.0, 1.0, 0.0)
    for node in mp.Nodes:
        node.AddDof(KM.TEMPERATURE)
    sub = mp.CreateSubModelPart("bnd")
    sub.AddNodes([1, 2])
    settings = KM.Parameters("""{
        "Parameters": {
            "model_part_name": "%s.bnd",
            "variable_name": "TEMPERATURE",
            "value": 50.0,
            "constrained": %s,
            "interval": [0.0, 1e30]
        }
    }""" % (mp.Name, "true" if constrained else "false"))
    proc = asvp.Factory(settings, model)
    proc.ExecuteInitialize()
    proc.ExecuteInitializeSolutionStep()
    node = mp.GetNode(1)
    return node.IsFixed(KM.TEMPERATURE), node.GetSolutionStepValue(KM.TEMPERATURE)


def main() -> int:
    bad = 0
    fixed_f, val_f = run(constrained=False)
    fixed_t, val_t = run(constrained=True)
    print(f"constrained_false_isfixed={fixed_f}")
    print(f"constrained_false_value={val_f}")
    print(f"constrained_true_isfixed={fixed_t}")
    print(f"constrained_true_value={val_t}")
    if fixed_f is not False:
        print("FAIL: constrained=false fixed the DOF anyway", file=sys.stderr)
        bad += 1
    if fixed_t is not True:
        print("FAIL: constrained=true did NOT fix the DOF", file=sys.stderr)
        bad += 1
    if val_f != 50.0 or val_t != 50.0:
        print(f"FAIL: value not applied in both cases ({val_f}, {val_t})",
              file=sys.stderr)
        bad += 1
    print(f"dirichlet_constrained_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
