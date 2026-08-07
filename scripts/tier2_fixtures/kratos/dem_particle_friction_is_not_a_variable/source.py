"""Tier-2: PARTICLE_FRICTION is not a Kratos variable; friction is STATIC_/DYNAMIC_.

Pitfall (kratos.dem): the OASiS DEM template used to emit PARTICLE_FRICTION into
both the mdpa Properties block and MaterialsDEM.json. The name appears in ZERO
files of the installed Kratos distribution, compiled libraries included. Kratos
DEM spells per-contact friction STATIC_FRICTION and DYNAMIC_FRICTION, and both
live in a `material_relations` entry, not in `materials`.

Observable, executed on DEMApplication 10.4.3:

  KratosGlobals.GetVariable("PARTICLE_FRICTION")
    -> ValueError: Kernel.GetVariable() ERROR: Variable PARTICLE_FRICTION is
       unknown. Maybe you need to import the application where it is defined?

  KratosGlobals.GetVariable("STATIC_FRICTION")   -> resolves
  KratosGlobals.GetVariable("DYNAMIC_FRICTION")  -> resolves

Mutation control: T2_MUTATE=1 swaps the probed name, asserting that
STATIC_FRICTION is the unknown one and PARTICLE_FRICTION the registered one --
the exact inversion of what this build provides. The registry lookup itself is
untouched, so the mutation proves the verdict comes from a real GetVariable call
and not from an echoed table. Mutated, absent_name=PARTICLE_FRICTION and both
present_name lines disappear and the process exits non-zero.
"""
from __future__ import annotations

import os
import sys

import KratosMultiphysics as KM
import KratosMultiphysics.DEMApplication  # noqa: F401  (registers DEM variables)

MUTATE = os.environ.get("T2_MUTATE") == "1"

# The name the catalog used to serve, and the two names Kratos actually uses.
ABSENT = "PARTICLE_FRICTION"
PRESENT = ["STATIC_FRICTION", "DYNAMIC_FRICTION"]

if MUTATE:
    print("mutation=claimed_absent_and_present_names_swapped")
    ABSENT, PRESENT = PRESENT[0], [ABSENT, PRESENT[1]]


def resolves(name: str) -> bool:
    try:
        KM.KratosGlobals.GetVariable(name)
        return True
    except Exception:
        return False


def main() -> int:
    failures = []

    if resolves(ABSENT):
        failures.append(f"{ABSENT} resolved but was claimed unknown")
    else:
        print(f"absent_name={ABSENT} resolves=False")

    for name in PRESENT:
        if resolves(name):
            print(f"present_name={name} resolves=True")
        else:
            failures.append(f"{name} did not resolve but was claimed registered")

    if failures:
        for f in failures:
            print("MISMATCH:", f, file=sys.stderr)
        return 1

    print("dem_friction_variable_check=ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
