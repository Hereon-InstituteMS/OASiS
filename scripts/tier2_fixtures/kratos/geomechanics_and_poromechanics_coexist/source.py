"""Tier-2: UPw lives in GeoMechanics, UPl in Poromechanics.

Two applications own two spellings of the same formulation.
The pitfall in each application's own section reads as a
version-wide fact and is not one. Both modules import, so both
stems are reachable — from different applications.

Mutation control: T2_MUTATE=1 INVERTS every claimed availability flag in CASES, i.e. it
asserts the opposite of what this install actually provides, without touching the
probe itself. Each importable[<app>]=<got>_expected=<must> line then disagrees
with itself and app_import_mismatches rises from 0 to len(CASES). That is the
control this fixture needs: it proves the printed booleans come from a real
importlib.import_module call on this build and that a wrong availability claim is
actually caught, rather than the fixture echoing a hard-coded table.
"""
from __future__ import annotations

import importlib
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")

# Imported unconditionally: with Kratos absent this raises and the
# process exits non-zero, so the fixture cannot report a pass while
# testing nothing.
import KratosMultiphysics as KM

MUTATE = os.environ.get("T2_MUTATE") == "1"
if MUTATE:
    print("mutation=every_claimed_application_availability_flag_inverted")

print(f"kratos_version_present={bool(KM.__file__)}")

# (application_module_suffix, must_import)
CASES = [('GeoMechanicsApplication', True), ('PoromechanicsApplication', True), ('StructuralMechanicsApplication', True), ('NoSuchKratosApplication', False)]
if MUTATE:
    # Pathology injected: claim the opposite availability for every
    # application, leaving the import probe itself untouched.
    CASES = [(app, not must) for app, must in CASES]


def main() -> int:
    bad = 0
    for app, must in CASES:
        try:
            importlib.import_module("KratosMultiphysics." + app)
            got, err = True, ""
        except BaseException as exc:
            got, err = False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:130]}"
        print(f"importable[{app}]={got}_expected={must}")
        if not got:
            print(f"  message: {err}")
        if got != must:
            bad += 1
            print(f"FAIL: {app} importable={got} expected={must} {err}",
                  file=sys.stderr)
    print(f"app_import_mismatches={bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
