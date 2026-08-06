"""Tier-2: two CableNet Python process wrappers, exercised.

empirical_spring_element_process is claimed to fail in numpy
before its C++ check can report a misleading message.
sliding_edge_process is claimed to be unusable through every
path. Both are instantiated here and the real failure recorded.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "2")

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.CableNetApplication as CN



def main() -> int:
    bad = 0

    # ---- empirical_spring_element_process: mismatched data arrays ----
    import KratosMultiphysics.CableNetApplication.empirical_spring_element_process as esp

    model = KM.Model()
    mp = model.CreateModelPart("Structure")
    mp.CreateNewNode(1, 0.0, 0.0, 0.0)
    mp.CreateNewNode(2, 1.0, 0.0, 0.0)
    params = KM.Parameters("""{
        "Parameters": {
            "model_part_name": "Structure",
            "computing_model_part_name": "Structure",
            "node_ids": [1, 2],
            "element_id": 1,
            "property_id": 1,
            "displacement_data": [0.0, 1.0, 2.0],
            "force_data": [0.0, 1.0],
            "polynomial_order": 1
        }
    }""")
    try:
        esp.Factory(params, model)
        print("mismatched_arrays_raised=False")
        print("FAIL: a displacement/force length mismatch was accepted",
              file=sys.stderr)
        bad += 1
    except Exception as exc:
        msg = str(exc).strip().splitlines()[-1]
        print("mismatched_arrays_raised=True")
        print(f"raiser={type(exc).__name__}")
        if "same length" in msg:
            print("observed=numpy_polyfit_same_length")
        else:
            print(f"FAIL: expected the numpy polyfit length error, got "
                  f"{msg[:140]}", file=sys.stderr)
            bad += 1

    # ---- sliding_edge_process: broken through the wrapper ----
    import KratosMultiphysics.CableNetApplication.sliding_edge_process as sep

    model2 = KM.Model()
    s = model2.CreateModelPart("Structure")
    s.CreateSubModelPart("m")
    s.CreateSubModelPart("s")
    p2 = KM.Parameters("""{
        "Parameters": {
            "master_sub_model_part_name": "m",
            "slave_sub_model_part_name": "s",
            "variable_names": ["DISPLACEMENT_X"]
        }
    }""")
    try:
        sep.Factory(p2, model2)
        print("sliding_edge_wrapper_raised=False")
        print("FAIL: sliding_edge_process instantiated cleanly, so the claim "
              "that it is broken through every user path would be false",
              file=sys.stderr)
        bad += 1
    except Exception as exc:
        msg = str(exc).replace("\n", " ")
        print("sliding_edge_wrapper_raised=True")
        if "NOT in the default values" in msg:
            print("observed=validate_and_assign_defaults_rejects_own_key")
        else:
            print(f"FAIL: unexpected sliding_edge failure {msg[:150]}",
                  file=sys.stderr)
            bad += 1

    print(f"cablenet_process_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
