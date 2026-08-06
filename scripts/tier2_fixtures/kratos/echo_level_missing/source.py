"""Tier-2: Kratos Parameters access without echo_level field.

Pitfall (Kratos linear_elasticity#7): problem_data section in
ProjectParameters.json MUST include 'echo_level'. Kratos
accesses it during stage initialisation without a default.
Calling Parameters['problem_data']['echo_level'].GetInt() on
a Parameters object without that field raises:

  RuntimeError: Error: Getting a value that does not exist.
  entry string : echo_level
  ... in kratos/sources/kratos_parameters.cpp:426
  Parameters Parameters::GetValue(const string&)

Mutation control: T2_MUTATE=1 puts an echo_level field into the problem_data block, so the lookup finds it and GetInt succeeds; the RuntimeError about getting a value that does not exist disappears.
"""
from __future__ import annotations

import os
import sys
import traceback

import KratosMultiphysics as KM

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=the_absent_field_is_now_present_in_problem_data")


def main() -> int:
    params = KM.Parameters('''
    {
        "problem_data": {
            "problem_name": "test",''' + ('''
            "echo_level": 0,''' if MUTATE else '') + '''
            "parallel_type": "OpenMP",
            "start_time": 0.0,
            "end_time": 1.0
        }
    }
    ''')
    pd = params["problem_data"]
    try:
        pd["echo_level"].GetInt()
    except Exception:
        traceback.print_exc()
        return 1
    print("ERROR: GetValue returned echo_level despite the field "
          "being absent (catalog claim wrong)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
