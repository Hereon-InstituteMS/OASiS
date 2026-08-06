"""Tier-2: UPw lives in GeoMechanics, UPl in Poromechanics.

Two applications own two spellings of the same formulation.
The pitfall in each application's own section reads as a
version-wide fact and is not one. Both modules import, so both
stems are reachable — from different applications.
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

print(f"kratos_version_present={bool(KM.__file__)}")

# (application_module_suffix, must_import)
CASES = [('GeoMechanicsApplication', True), ('PoromechanicsApplication', True), ('StructuralMechanicsApplication', True), ('NoSuchKratosApplication', False)]


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
