"""Tier-2: MPM solver_type takes a short label, not a solver class name.

Pitfall (kratos.mpm): the catalog served MPMStaticSolver / MPMQuasiStaticSolver /
MPMImplicitDynamicSolver / MPMExplicitSolver as the MPM "solver_types", which
reads as the value to put in ProjectParameters. None of the four is accepted.
The python solvers wrapper matches short labels:

    static | Static
    quasi_static | Quasi-static
    dynamic | Dynamic          (+ time_integration_method: implicit | explicit)

and rejects anything else with

    The requested solver type "MPMImplicitDynamicSolver" is not in the python
    solvers wrapper
    Available options are: "static", "dynamic", "quasi_static"

This fixture calls the wrapper directly, so it never needs a mesh.

Mutation control: T2_MUTATE=1 INVERTS the accept/reject expectation for every
spelling -- it asserts the four class names are accepted and the three real
labels are not. The wrapper call is untouched, so the mutation proves each
verdict comes from a real CreateSolver attempt. Mutated, every accepted[...]
line disagrees with itself and solver_type_mismatches goes 0 -> 7.
"""
from __future__ import annotations

import json
import os
import sys

import KratosMultiphysics as KM
from KratosMultiphysics.MPMApplication import python_solvers_wrapper_mpm

MUTATE = os.environ.get("T2_MUTATE") == "1"

# spelling -> is it a value the wrapper accepts?
CASES = [
    ("static", True),
    ("quasi_static", True),
    ("dynamic", True),
    ("MPMStaticSolver", False),
    ("MPMQuasiStaticSolver", False),
    ("MPMImplicitDynamicSolver", False),
    ("MPMExplicitSolver", False),
]
if MUTATE:
    print("mutation=accept_reject_expectations_inverted")
    CASES = [(name, not ok) for name, ok in CASES]


def wrapper_accepts(solver_type: str) -> bool:
    """True unless the wrapper rejects the label itself.

    Anything past the label check (a missing mdpa, a missing key) counts as
    accepted: this fixture is about the label, not about a complete deck.
    """
    settings = KM.Parameters(json.dumps({
        "solver_settings": {
            "solver_type": solver_type,
            "time_integration_method": "implicit",
            "model_part_name": "MPM_Material",
            "domain_size": 2,
        },
        "problem_data": {"parallel_type": "OpenMP"},
    }))
    try:
        python_solvers_wrapper_mpm.CreateSolver(KM.Model(), settings)
        return True
    except Exception as exc:  # noqa: BLE001 - classifying, not handling
        return "is not in the python solvers wrapper" not in str(exc)


def main() -> int:
    mismatches = 0
    for name, must in CASES:
        got = wrapper_accepts(name)
        print(f"accepted[{name}]={got}_expected={must}")
        if got != must:
            mismatches += 1
            print(f"MISMATCH: {name} accepted={got} expected={must}", file=sys.stderr)

    print(f"solver_type_mismatches={mismatches}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
